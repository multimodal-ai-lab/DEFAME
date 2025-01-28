import asyncio
from pathlib import Path

from fastapi import FastAPI, status, HTTPException, Depends, Response, WebSocket
from fastapi.encoders import jsonable_encoder
from fastapi.security import APIKeyHeader

from defame.helpers.api.query import UserQuery, QueryManager, Status
from defame.helpers.parallelization.pool import FactCheckerPool

API_KEY = "sdfg8rtjzuo5we68g54654rehz9tghr65465df414qq"

SAVE_DIR = Path("out/api/")


def ensure_authentication(api_key: str):
    if not api_key == API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )


app = FastAPI()

header_scheme = APIKeyHeader(name="api-key")

pool = FactCheckerPool(target_dir=SAVE_DIR, n_workers=8)

query_manager = QueryManager(pool)


@app.get("/")
async def root():
    return "DEFAME API is up and running!"


@app.post("/verify/")
async def verify(user_query: UserQuery, api_key: str = Depends(header_scheme)):
    """Adds the user-submitted content to the fact-checking task pool. Returns the task ID."""
    ensure_authentication(api_key)
    query_id = query_manager.add_query(user_query)
    return {"query_id": query_id}


@app.websocket("/status/{query_id}")
async def websocket_endpoint(websocket: WebSocket, query_id: str):
    """Delivers the current state immediately, followed by real-time updates (containing
    just the changes). Closes automatically when fact-check is done or failed."""
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


@app.get("/results/{query_id}")
async def get_query_results(query_id: str):
    """Returns the fact-checking results for this query among with its current status. If
    the query is not finished yet, returns just the status."""
    return await query_manager.get_latest_query_info(query_id)


@app.get("/results/{query_id}/{claim_id}/report.pdf")
async def get_report_pdf(query_id: str, claim_id: int):
    """For downloading the Report PDF."""

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
