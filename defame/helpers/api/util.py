from fastapi import HTTPException
from starlette import status

from defame.helpers.api.config import api_key


def ensure_authentication(presented_key: str):
    if not presented_key == api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
