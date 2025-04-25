import base64
from datetime import datetime
from io import BytesIO
from secrets import token_urlsafe

from PIL import UnidentifiedImageError, Image as PillowImage
from ezmm import Image
from fastapi import HTTPException
from starlette import status as http_status

from defame.common import Content
from defame.helpers.api.common import UserSubmission
from defame.helpers.api.job import Job
from defame.helpers.parallelization.pool import Pool


class JobManager:
    def __init__(self, pool: Pool):
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
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Query with ID {query_id} not found."
            )

    def get_job(self, job_id: str) -> Job:
        self.validate_query_id(job_id)
        return self.job_registry[job_id]


def process_submission(submission: UserSubmission) -> Content:
    processed = []
    for block in submission.content:
        block_type, block_content = block
        if block_type.lower() == "text":
            processed.append(block_content)
        elif block_type.lower() == "image":
            try:
                img = PillowImage.open(BytesIO(base64.b64decode(block_content)))
            except UnidentifiedImageError:
                raise HTTPException(http_status.HTTP_422_UNPROCESSABLE_ENTITY,
                                    detail="The provided string is not a valid Base64-encoding of an image:\n" +
                                           block_content)
            image = Image(pillow_image=img)
            processed.append(image)
        else:
            raise ValueError(f"Invalid block type: {block_type}")
    date = datetime.strptime(submission.date, "%Y-%m-%d") if submission.date else None
    return Content(processed, author=submission.author, date=date)
