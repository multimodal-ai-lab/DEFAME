import asyncio

from fastapi import FastAPI, status, HTTPException, Depends, Response, WebSocket
from fastapi.security import APIKeyHeader
from websockets import ConnectionClosed
from starlette.websockets import WebSocketDisconnect

from defame.common import logger
from .job_manager import JobManager
from defame.helpers.parallelization.pool import Pool
from .common import UserSubmission
from .job import StatusResponse
from .config import save_dir, fact_checker_kwargs
from .util import ensure_authentication
from defame.utils.utils import deep_diff

title = "DEFAME API"
version = "0.1.1"
description = """This is the API backend of DEFAME, a multimodal AI fact-checker.
The API enables you to run HTTP requests in order to submit fact-checks and retrieve their results.
This documentation is semi-automatically generated via FastAPI.

## Authentication
To submit new content to be fact-checked (via `/verify`), you need to authenticate with an API key.
Ask [Mark Rothermel](mailto:mark.rothermel@tu-darmstadt.de) if you want to get access.

## `/status` Websocket
Besides the API calls below, you can get real-time updates via a websocket available at
`/status/{job_id}`. You can connect to it with
 ```ws://<api_domain>:<api_port>/status/{job_id}```
First, you will receive a full status message. Then, as long as the fact-check is running,
the websocket will send you live updates on (only) *the changes* regarding the fact-check's
status and output results.
"""
tags_metadata = [
    {
        "name": "API Calls",
        "description": "All available API endpoints.",
    },
]

app = FastAPI(title=title, description=description, version=version,
              openapi_tags=tags_metadata,
              contact={
                  "name": "Mark Rothermel",
                  "email": "mark.rothermel@tu-darmstadt.de",
              },
              license_info={
                  "name": "Apache 2.0",
                  "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
              },
              )

header_scheme = APIKeyHeader(name="api-key")

pool = Pool(target_dir=save_dir, n_workers=8, print_log_level="debug", **fact_checker_kwargs)

job_manager = JobManager(pool)

# Initialize logger
logger.set_experiment_dir(save_dir)
logger.info("API backend successfully initialized.")


@app.get("/", summary="Use this to see if the API is running.", tags=["API Calls"])
async def root():
    return "DEFAME API is up and running!"


@app.post("/verify", summary="Submit new content to be decomposed and fact-checked.", tags=["API Calls"])
async def verify(user_submission: UserSubmission, api_key: str = Depends(header_scheme)):
    """Adds the provided content to the fact-checking worker pool. Returns the job's ID
    with which the results can be retrieved. This endpoint requires authentication through an API key.
    """
    ensure_authentication(api_key)
    job_id = job_manager.add_job(user_submission)
    return {"job_id": job_id}


@app.websocket("/status/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """Delivers the current state immediately, followed by real-time updates (containing
    just the changes). Closes automatically when fact-check terminated. Can handle multiple
    connections for the same job concurrently."""
    try:
        job = job_manager.get_job(job_id)

        await websocket.accept()

        # Send full job status
        job_status = job.get_status()
        await websocket.send_json(job_status)

        # Send changes in real-time while the job is running
        while True:
            new_status = job.get_status()

            # Report only the changes
            if update := deep_diff(job_status, new_status):
                await websocket.send_json(update)

            job_status = new_status

            # Check if job has terminated
            if job_status["job_info"]["status"] in ["DONE", "FAILED"]:
                break
            else:
                await asyncio.sleep(0.1)

        await websocket.close()

    except (ConnectionClosed, WebSocketDisconnect):
        logger.log("Websocket connection closed unexpectedly.")


@app.get("/results/{job_id}",
         summary="Get the status/results of a previously submitted fact-checking job.",
         tags=["API Calls"])
async def get_job_results(job_id: str) -> StatusResponse:
    """Returns the full job status incl. any results for the job specified with `job_id`."""
    job = job_manager.get_job(job_id)
    return job.get_status()


@app.get("/results/{job_id}/{claim_id}/report.pdf",
         summary="Download the report PDF of a fact-check.",
         tags=["API Calls"])
async def get_report_pdf(job_id: str, claim_id: int):
    """Downloads the report PDF of the fact-check of the claim specified with `job_id` and
    `claim_id`."""

    job = job_manager.get_job(job_id)

    # Ensure that task is done
    claim_task = job.get_claim_task(claim_id)
    if claim_task is None:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} for job {job_id} not found.")

    if not claim_task.is_done:
        if claim_task.failed:
            detail = f"No report PDF available for claim {job_id}/{claim_id} because the task failed."
        else:
            detail = (f"No report PDF available as the fact-check for "
                      f"claim {job_id}/{claim_id} is not finished yet.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail
        )

    report_path = save_dir / "fact-checks" / job_id / str(claim_id) / "report.pdf"
    with open(report_path, "rb") as f:
        pdf_bytes = f.read()
    headers = {'Content-Disposition': 'inline; filename="report.pdf"'}
    return Response(pdf_bytes, headers=headers, media_type="application/pdf")
