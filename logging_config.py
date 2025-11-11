import logging
from datetime import datetime
import os
from starlette.middleware.base import BaseHTTPMiddleware
import time

def get_logger(name,log_dir='./logs'):
    os.makedirs(log_dir,exist_ok=True)
    log_file=os.path.join(log_dir,f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log")

    logger=logging.getLogger(name)
    logger.setLevel(logging.INFO)

    file_handler=logging.FileHandler(log_file)
    stream_handler=logging.StreamHandler()

    formatter=logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Logs the latency for every HTTP request."""
    async def dispatch(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        response.headers["X-Process-Time-ms"] = f"{process_time:.2f}"
        logging.getLogger("ml_api").info(
            f"{request.method} {request.url.path} completed in {process_time:.2f}ms"
        )
        return response