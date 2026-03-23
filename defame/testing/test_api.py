import json

import pytest
import requests
from ezmm import Image
from fastapi.testclient import TestClient
from websockets.sync.client import connect

from defame.helpers.api.config import api_key, host, port

TEST_DEPLOYED_API = False  # use the existing, running API instead of creating a temporary one

job_id = None  # may be set to an existing job


class DeployedClient:
    def __init__(self, url: str):
        self.url = url

    def get(self, path: str, **kwargs):
        return requests.get(self.url + path, **kwargs)

    def post(self, path: str, **kwargs):
        return requests.post(self.url + path, **kwargs)


host_url = f"http://{host}:{port}"

if TEST_DEPLOYED_API:
    client = DeployedClient(host_url)
else:
    from defame.helpers.api.main import app

    client = TestClient(app)


def test_is_running():
    response = client.get("/")
    assert response.status_code == 200
    assert response.text == "\"DEFAME API is up and running!\""


def test_verify():
    headers = {"api-key": api_key, "Content-Type": "application/json"}
    body = dict(
        content=[
            ("text", "The image"),
            ("image", Image("in/example/sahara.webp").get_base64_encoded()),
            ("text", "shows the Sahara in 2023 covered with snow! Did you know, Obi-Wan got killed by Anakin?")
        ],
        author="Some social media user.",
        date="2025-02-21"
    )
    response = client.post("/verify", json=body, headers=headers)
    assert response.status_code == 200
    assert response.json() is not None

    global job_id
    job_id = response.json()["job_id"]
    assert isinstance(job_id, str)
    print("\njob_id:", job_id)


@pytest.mark.timeout(90)
def test_websocket_endpoint():
    if TEST_DEPLOYED_API:
        with connect(f"ws://{host}:{port}/status/{job_id}") as websocket:
            while True:
                msg = websocket.recv()
                msg = json.loads(msg)
                print(json.dumps(msg, indent=4))
                if "job_info" in msg:
                    job_info = msg["job_info"]
                    if job_info.get("status") == "DONE":
                        return True
                    elif job_info.get("status") == "FAILED":
                        raise AssertionError(f"Fact-check failed!")
    else:
        with client.websocket_connect(f"/status/{job_id}") as websocket:
            while True:
                msg = websocket.receive_json()
                print(json.dumps(msg, indent=4))
                if "job_info" in msg:
                    job_info = msg["job_info"]
                    if job_info.get("status") == "DONE":
                        return True
                    elif job_info.get("status") == "FAILED":
                        raise AssertionError(f"Fact-check failed!")


def test_get_results():
    response = client.get(f"/results/{job_id}")

    assert response.status_code == 200
    assert response.json() is not None
    data = response.json()

    print("\nJob results:", json.dumps(data, indent=4))

    # Check job info
    assert "job_info" in data
    job_info = data["job_info"]
    assert "job_id" in job_info
    assert "status" in job_info
    assert job_info["status"] == "DONE"
    assert "status_message" in job_info
    assert "tasks" in job_info
    assert len(job_info["tasks"]) > 0

    assert "content" in data
    assert "claims" in data
    assert len(data["claims"]) > 0
    assert "verdict" in list(data["claims"].values())[0]
    assert "justification" in list(data["claims"].values())[0]


def test_get_report_pdf():
    claim_id = 0
    response = client.get(f"/results/{job_id}/{claim_id}/report.pdf")
    assert response.status_code == 200
