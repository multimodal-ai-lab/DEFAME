"""Starts DEFAME's API server, offering to run DEFAME fact-checks via
HTTP requests."""

import uvicorn
from defame.helpers.api.main import app
from defame.helpers.api.config import host, port


if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port)
