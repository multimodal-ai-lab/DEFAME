from secrets import token_urlsafe
from typing import Any, Optional
from io import BytesIO
import base64

from PIL import Image as PillowImage
from fastapi import HTTPException
from pydantic import BaseModel
from starlette import status

from defame.common import Content, Image, Claim, Label
from defame.helpers.parallelization.pool import FactCheckerPool, Status
from defame.helpers.parallelization.task import Task


class UserQuery(BaseModel):
    """User-submitted, raw content to be fact-checked."""
    data: list[tuple[str, Any]]  # list of text and image bytes


class FactCheckQuery:
    """Keeps all tasks related to the fact-check query in one place. There is one
    task to extract the claims from the content and, additionally, there is one
    task per extracted claim, making 1 + n tasks. The ID of the claim extraction
    task has the same ID as the query."""

    def __init__(self, identifier: str, content: Content, pool: FactCheckerPool):
        self.id = identifier
        self._pool = pool
        self.content_task = Task(content, identifier=identifier, status_message="Scheduled for fact-check.")
        self.claim_tasks: Optional[list[Task]] = None

    @property
    def terminated(self) -> bool:
        return (self.content_task.terminated and
                self.claim_tasks is not None and all([t.terminated for t in self.claim_tasks]))

    @property
    def is_running(self) -> bool:
        return (self.content_task.is_running or
                self.claim_tasks is not None and any([t.is_running for t in self.claim_tasks]))

    @property
    def is_done(self) -> bool:
        return self.claim_tasks is not None and all([t.is_done for t in self.claim_tasks])

    @property
    def failed(self) -> bool:
        return (self.content_task.failed or
                self.claim_tasks is not None and any([t.failed for t in self.claim_tasks]))

    @property
    def status(self) -> Status:
        if self.is_done:
            return Status.DONE
        elif self.failed:
            return Status.FAILED
        elif self.is_running:
            return Status.RUNNING
        else:
            return Status.PENDING

    @property
    def content(self) -> Content:
        return self.content_task.payload

    @property
    def claims(self) -> Optional[list[Claim]]:
        return self.content.claims

    def set_claims(self, claims: list[Claim]):
        """Saves the claims belonging to the querie's content. Also adds the claims
        as tasks to the worker pool."""
        self.content.claims = claims

        # Create new tasks from claims
        self.claim_tasks = []
        for i, claim in enumerate(claims):
            task = Task(claim, identifier=self.content_task.id + f"/{i}")
            self.claim_tasks.append(task)
            self._pool.add_task(task)

    @property
    def n_claims(self) -> Optional[int]:
        claims = self.content.claims
        return len(claims) if claims is not None else None

    @property
    def aggregated_verdict(self) -> Optional[Label]:
        return self.content.verdict

    def get_claim(self, claim_id: int) -> Optional[Claim]:
        claims = self.content.claims
        if claims is not None:
            return claims[claim_id]

    async def update(self) -> dict:
        """Updates all tasks related to this query. Returns a summary of all changes."""
        update_message = dict()

        # Update content and save its message
        content_task_update_message = await self.content_task.update(blocking=False)
        if content_task_update_message is not None:
            update_message["content"] = content_task_update_message
            if "result" in content_task_update_message:
                # Register new claims and add them to pool as new tasks
                claims_dict = content_task_update_message["result"]
                claims = [c for i, c in sorted(claims_dict.items())]
                self.set_claims(claims)

        # Do the same for all claims
        if self.claim_tasks is not None:
            for i, claim_task in enumerate(self.claim_tasks):
                claim_task_update_message = await claim_task.update(blocking=False)
                if claim_task_update_message is not None:
                    claim_task_update_message.update(id=i)
                    update_message["claims"] = update_message.get("claims", []) + [claim_task_update_message]

        return update_message

    async def get_updated_info(self) -> dict:
        """Updates the query and returns its full current status."""
        await self.update()
        return self.__getstate__()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_pool", None)
        state.pop("content_task", None)
        state.pop("claim_tasks", None)
        state["status"] = self.status.value
        state["content"] = self.content_task
        state["claims"] = self.claim_tasks
        return state


class QueryManager:
    def __init__(self, pool: FactCheckerPool):
        self.pool = pool
        self.query_registry: dict[str, FactCheckQuery | None] = dict()  # TODO: Make persistent

    def add_query(self, query: FactCheckQuery | UserQuery) -> str:
        """Adds the query to the query registry and returns the query's ID."""
        if isinstance(query, UserQuery):
            content = process_query(query)
            query_id = self.generate_query_id()
            content.id = query_id
            query = FactCheckQuery(query_id, content, self.pool)
        self.query_registry[query.id] = query
        self.pool.add_task(query.content_task)
        return query.id

    def generate_query_id(self):
        while query_id := token_urlsafe(16):
            if query_id not in self.query_registry:
                self.query_registry[query_id] = None  # register the ID
                return query_id

    def validate_query_id(self, query_id: str):
        if query_id not in self.query_registry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Query with ID {query_id} not found."
            )

    def get_query(self, query_id: str) -> FactCheckQuery:
        self.validate_query_id(query_id)
        return self.query_registry[query_id]

    async def get_latest_query_info(self, query_id: str):
        """Returns the latest info for the specified query. Updates the status of all tasks first."""
        query = self.get_query(query_id)
        return await query.get_updated_info()


def process_query(user_query: UserQuery) -> Content:
    processed = []
    for block in user_query.data:
        block_type, block_content = block
        if block_type.lower() == "text":
            processed.append(block_content)
        elif block_type.lower() == "image":
            img = PillowImage.open(BytesIO(base64.b64decode(block_content)))
            image = Image(pillow_image=img)
            processed.append(image)
        else:
            raise ValueError(f"Invalid block type: {block_type}")
    return Content(processed)


def task_id_to_query_and_claim_id(task_id: str) -> (str, Optional[int]):
    if "/" in task_id:
        query_id, claim_id = task_id.split("/")
        return query_id, int(claim_id)
    else:
        return task_id, None
