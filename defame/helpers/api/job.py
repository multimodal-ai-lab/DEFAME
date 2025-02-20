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
from defame.utils.utils import deep_diff


class UserSubmission(BaseModel):
    """User-submitted, raw content to be fact-checked."""
    data: list[tuple[str, Any]]  # list of text and image bytes


class Job:
    """A fact-checking job. Keeps all tasks related to the fact-check in one place.
    There is one task to extract the claims from the content and, additionally,
    there is one task per extracted claim, making 1 + n tasks. The ID of the claim
    extraction task has the same ID as the query."""

    def __init__(self, identifier: str, content: Content, pool: FactCheckerPool):
        self.id = identifier
        self._pool = pool
        self.content_task = Task(content, identifier=identifier, status_message="Scheduled for extraction.")
        self.claim_tasks: Optional[list[Task]] = None
        self._latest_update = None  # used to track changes

    @property
    def tasks(self) -> list[Task]:
        tasks = [self.content_task]
        if self.claim_tasks is not None:
            tasks.extend(self.claim_tasks)
        return tasks

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
            claim.id = self.content_task.id + f"/{i}"
            task = Task(claim, identifier=claim.id)
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

    def get_status_message(self) -> str:
        match self.status:
            case Status.PENDING:
                return "In queue."
            case Status.RUNNING:
                if self.content_task.is_running:
                    return "Extracting claims."
                else:
                    return "Verifying claims."
            case Status.DONE:
                return "Job completed successfully."
            case Status.FAILED:
                return "Job failed, see the corresponding task for details."

    def get_update(self, report_only_changes: bool = True) -> Optional[dict]:
        """Updates all tasks related to this query. Returns a summary of all changes,
        following the schema for status messages."""
        if self.content.claims is not None and self.claim_tasks is None:
            # Add new claims to pool as new tasks
            self.set_claims(self.content.claims)

        # Gather summary info and compose it into an update message
        job_info = self.get_summary()
        content = self.content.get_summary()
        claims = {c.id: c.get_summary() for c in self.claims} if self.claims is not None else None
        update = dict(job_info=job_info, content=content, claims=claims)

        # Remove unchanged entries
        if report_only_changes:
            message = self.get_changes(update)
        else:
            message = update
        self._latest_update = update

        if message:
            return message

    def get_changes(self, update_message: dict, **kwargs) -> dict:
        """Keeps only the changes in the update message."""
        if self._latest_update is not None:
            difference = deep_diff(self._latest_update, update_message, **kwargs)
        else:
            difference = update_message
        return difference

    def get_summary(self):
        """Used to construct the response message."""
        tasks = {t.id: t.get_summary() for t in self.tasks}
        return dict(job_id=self.id,
                    status=self.status.value,
                    status_message=self.get_status_message(),
                    tasks=tasks)


class JobManager:
    def __init__(self, pool: FactCheckerPool):
        self.pool = pool
        self.job_registry: dict[str, Job | None] = dict()  # TODO: Make persistent

    def add_job(self, job: Job | UserSubmission) -> str:
        """Adds the query to the query registry and returns the query's ID."""
        if isinstance(job, UserSubmission):
            content = process_submission(job)
            job_id = self.generate_job_id()
            content.id = job_id
            job = Job(job_id, content, self.pool)
        self.job_registry[job.id] = job
        self.pool.add_task(job.content_task)
        return job.id

    def generate_job_id(self):
        while query_id := token_urlsafe(16):
            if query_id not in self.job_registry:
                self.job_registry[query_id] = None  # register the ID
                return query_id

    def validate_query_id(self, query_id: str):
        if query_id not in self.job_registry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Query with ID {query_id} not found."
            )

    def get_job(self, job_id: str) -> Job:
        self.validate_query_id(job_id)
        return self.job_registry[job_id]


def process_submission(user_query: UserSubmission) -> Content:
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
    return Content(data=processed)


# def task_id_to_job_and_claim_id(task_id: str) -> (str, Optional[int]):
#     if "/" in task_id:
#         query_id, claim_id = task_id.split("/")
#         return query_id, int(claim_id)
#     else:
#         return task_id, None
