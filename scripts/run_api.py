"""Starts DEFAME's API server, offering to run DEFAME fact-checks via
HTTP requests."""

import uvicorn
from defame.helpers.api.main import app


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3003)
