import asyncio
from pathlib import Path

from fastapi import FastAPI, status, HTTPException, Depends, Response, WebSocket
from fastapi.encoders import jsonable_encoder
from fastapi.security import APIKeyHeader

from defame.helpers.api.query import UserQuery, QueryManager, Status
from defame.helpers.parallelization.pool import FactCheckerPool

API_KEY = "sdfg8rtjzuo5we68g54654rehz9tghr65465df414qq"

SAVE_DIR = Path("out/api/")


title = "DEFAME API"
version = "0.0.1"
description = """This is the API backend of DEFAME, a multimodal AI fact-checker.
The API enables you to run HTTP requests in order to submit fact-checks and retrieve their results.
This documentation is semi-automatically generated via FastAPI.

## Authentication
To submit new content to be fact-checked (via `/verify`), you need to authenticate with an API key.
Ask [Mark Rothermel](mailto:mark.rothermel@tu-darmstadt.de) if you want to get access.

## `/status` Websocket
Besides the API calls below, you can get real-time updates via a websocket available at
`/status/{query_id}`. You can connect to it with
 ```ws://<api_domain>:<api_port>/status/{query_id}```
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


def ensure_authentication(api_key: str):
    if not api_key == API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )


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

pool = FactCheckerPool(target_dir=SAVE_DIR, n_workers=8)

query_manager = QueryManager(pool)


@app.get("/", summary="Use this to see if the API is running.", tags=["API Calls"])
async def root():
    return "DEFAME API is up and running!"


@app.post("/verify/", summary="Submit new content to be decomposed and fact-checked.", tags=["API Calls"])
async def verify(user_query: UserQuery, api_key: str = Depends(header_scheme)):
    """Adds the provided content to the fact-checking worker pool. Returns the query's ID
    with which the results can be retrieved. Expects the content to be JSON-formatted, for example:
    ```json
    {
        "data": [
            ("text", "The image"),
            ("image", <base64_encoded_image_string>),
            ("text", "shows the Sahara in 2023 covered with snow!),
        ]
    }
    ```
    This endpoint requires authentication through an API key.
    """
    ensure_authentication(api_key)
    query_id = query_manager.add_query(user_query)
    return {"query_id": query_id}


@app.websocket("/status/{query_id}")
async def websocket_endpoint(websocket: WebSocket, query_id: str):
    """Delivers the current state immediately, followed by real-time updates (containing
    just the changes). Closes automatically when fact-check terminated."""
    await websocket.accept()

    # Send initial query status
    query = query_manager.get_query(query_id)
    query_status = await query.get_updated_info()
    await websocket.send_json(jsonable_encoder(query_status))

    # Send all updates in real time while the query is running
    while not query.terminated:
        update = await query.update()
        if update:
            await websocket.send_json(jsonable_encoder(update))
        await asyncio.sleep(0.1)

    # Send final termination signal
    if query.is_done:
        await websocket.send_json(jsonable_encoder(dict(status=Status.DONE.value)))
    else:
        await websocket.send_json(jsonable_encoder(dict(status=Status.FAILED.value)))

    await websocket.close()


@app.get("/results/{query_id}",
         summary="Get the status/results of a previously submitted fact-checking query.",
         tags=["API Calls"])
async def get_query_results(query_id: str):
    """Returns the fact-checking results for the query specified with `query_id`. Contains
    the query's status and, if available, the corresponding decomposition and verification
    results. If the query is not finished yet, returns just the status."""
    return await query_manager.get_latest_query_info(query_id)


@app.get("/results/{query_id}/{claim_id}/report.pdf",
         summary="Download the report PDF of a fact-check.",
         tags=["API Calls"])
async def get_report_pdf(query_id: str, claim_id: int):
    """Downloads the report PDF of the fact-check of the claim specified with `query_id` and
    `claim_id`."""

    # Ensure that task is done
    query = query_manager.get_query(query_id)
    if not query.is_done:
        if query.failed:
            detail = f"No report PDF available for task {query_id} because the task failed."
        else:
            detail = f"No report PDF available as the task {query_id} is not finished yet."
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail
        )

    report_path = SAVE_DIR / "fact-checks" / query_id / str(claim_id) / "report.pdf"
    with open(report_path, "rb") as f:
        pdf_bytes = f.read()
    headers = {'Content-Disposition': 'inline; filename="report.pdf"'}
    return Response(pdf_bytes, headers=headers, media_type="application/pdf")
