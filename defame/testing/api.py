import pytest
from fastapi.testclient import TestClient

from defame.helpers.api.main import app, API_KEY
from defame.common import Image

client = TestClient(app)
query_id = None


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.text == "\"DEFAME API is up and running!\""


def test_verify():
    headers = {"api-key": API_KEY, "Content-Type": "application/json"}
    body = dict(
        data=[
            ("text", "The image"),
            ("image", Image("in/example/sahara.webp").get_base64_encoded()),
            ("text", "shows the Sahara in 2023 covered with snow!")
        ]
    )
    response = client.post("/verify", json=body, headers=headers)
    assert response.status_code == 200
    assert response.json() is not None

    global query_id
    query_id = response.json()["query_id"]
    assert isinstance(query_id, str)
    print("\nquery_id:", query_id)


@pytest.mark.timeout(90)
def test_websocket_endpoint():
    with client.websocket_connect(f"/status/{query_id}") as websocket:
        while True:
            msg = websocket.receive_json()
            print(msg)
            if "status" in msg:
                if msg["status"] == "done":
                    break
                elif msg["status"] == "failed":
                    raise AssertionError(f"Fact-check failed!")


def test_get_task_results():
    response = client.get(f"/results/{query_id}")

    assert response.status_code == 200
    assert response.json() is not None
    data = response.json()

    # Check overall query info
    assert "id" in data
    assert data["id"] == query_id
    assert "status" in data
    assert data["status"] == "done"

    # Check content info
    assert "content" in data
    content = data["content"]
    assert "status" in content
    assert content["status"] == "done"

    # Check claims infos
    assert "claims" in data
    claims = data["claims"]
    for i, claim in enumerate(claims):
        assert "id" in claim
        assert claim["id"] == f"{query_id}/{i}"
        assert "status" in claim
        assert claim["status"] == "done"

    print("\nquery results:", data)


def test_get_report_pdf():
    claim_id = 0
    response = client.get(f"/results/{query_id}/{claim_id}/report.pdf")
    assert response.status_code == 200
    assert response.stream is not None
