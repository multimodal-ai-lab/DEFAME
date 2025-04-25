from typing import Optional

from ezmm import MultimodalSequence
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from defame.common import Content, Claim, Label
from defame.helpers.api.common import ClaimInfo, get_claim_info, ContentInfo, get_content_info
from defame.helpers.common import Status
from defame.helpers.parallelization.pool import Pool
from defame.helpers.parallelization.task import Task, TaskInfo


class JobInfo(BaseModel):
    job_id: str
    status: str
    status_message: str
    tasks: dict[str, TaskInfo] = Field(examples=[{
        "<task_id>": TaskInfo(task_id="<task_id>",
                              status="PENDING",
                              status_message="In queue.")}])


class StatusResponse(BaseModel):
    job_info: JobInfo
    content: ContentInfo
    claims: Optional[dict[int, ClaimInfo]] = Field(examples=[{
        0: ClaimInfo(claim_id="<job_id>/0",
                     data=[["text", "This is the first claim."]],
                     verdict=None,
                     justification=None),
        1: ClaimInfo(claim_id="<job_id>/1",
                     data=[["text", "This is another claim."]],
                     verdict="SUPPORTED",
                     justification=[["text", "The claim is supported, because the image"],
                                    ["image", "<base64_encoded_image_string>"],
                                    ["text", "shows clear signs of deepfake generation."]])}])


class Job:
    """A fact-checking job. Keeps all tasks related to the fact-check in one place.
    There is one task to extract the claims from the content and, additionally,
    there is one task per extracted claim, making 1 + n tasks. The ID of the claim
    extraction task has the same ID as the query."""

    def __init__(self, identifier: str, content: Content, pool: Pool):
        self.id = identifier
        self._pool = pool
        self.content_task = Task(content,
                                 id=identifier,
                                 status_message="Scheduled for extraction.",
                                 callback=self.register_claims)
        self.claim_tasks: Optional[list[Task]] = None

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

    def register_claims(self, task: Task):
        """Saves the claims belonging to the query's content. Also adds the claims
        as tasks to the worker pool."""
        claims: list[Claim] = task.result["claims"]
        topic: str = task.result["topic"]

        # Create new tasks from claims
        self.claim_tasks = []
        for i, claim in enumerate(claims):
            claim.id = self.id + f"/{i}"
            task = Task(claim, id=claim.id, callback=self.register_verification_results)
            self.claim_tasks.append(task)
            self._pool.add_task(task)

        self.content.claims = claims
        self.content.topic = topic

    def register_verification_results(self, task: Task):
        claim = task.payload
        claim.verdict = task.result["verdict"]
        claim.justification = MultimodalSequence(task.result["justification"])

    @property
    def n_claims(self) -> Optional[int]:
        return len(self.claims) if self.claims is not None else None

    @property
    def aggregated_verdict(self) -> Optional[Label]:
        return self.content.verdict

    def get_claim(self, claim_idx: int) -> Optional[Claim]:
        """Returns the corresponding claim for the given integer index."""
        claims = self.content.claims
        if claims is not None:
            return claims[claim_idx]

    def get_claim_task(self, claim_idx: int) -> Optional[Task]:
        """Returns the corresponding task for the given integer index."""
        if self.claim_tasks is None or claim_idx not in range(self.n_claims):
            return None
        else:
            return self.claim_tasks[claim_idx]

    def get_status_message(self) -> str:
        match self.status:
            case Status.PENDING:
                return "In queue."
            case Status.RUNNING:
                if self.content_task.is_running:
                    return "Extracting claims."
                else:
                    return "Verifying claim(s)."
            case Status.DONE:
                return "Job completed successfully."
            case Status.FAILED:
                return "Job failed, see the corresponding task for details."
        return "Unknown status."

    def get_status(self):
        """Returns a full status report of the job and all (available) results."""
        job_info = self.get_info()
        content = get_content_info(self.content)
        claims = self.get_claims_info()
        response = StatusResponse(job_info=job_info, content=content, claims=claims)
        return jsonable_encoder(response)

    def get_claims_info(self):
        if self.claims is None:
            return None
        claims = dict()
        for claim in self.claims:
            index = int(claim.id.split("/")[-1])
            info = get_claim_info(claim)
            claims[index] = info
        return claims

    def get_info(self) -> JobInfo:
        """Used to construct the response message."""
        tasks = {t.id: t.get_info() for t in self.tasks}
        return JobInfo(job_id=self.id,
                       status=self.status.name,
                       status_message=self.get_status_message(),
                       tasks=tasks)
