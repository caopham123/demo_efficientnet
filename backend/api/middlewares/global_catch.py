import logging
from fastapi import status, Request, Response

logger = logging.getLogger(__name__)


async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.exception(e)
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)