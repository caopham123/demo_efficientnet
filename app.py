from fastapi import FastAPI
import os
from fastapi.middleware.cors import CORSMiddleware
from api.middlewares.global_catch import catch_exceptions_middleware
from api.controller import *

os.environ["LOG_LEVEL"] = "WARNING"
os.environ["LOG_LEVEL_LIBRARIES"] = "WARNING"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

app = FastAPI(
    title = 'FACE RECOGNIZE',
    description = "This API was built with FastAPI",
    version = "1.0.0",
)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware('http')(catch_exceptions_middleware)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5055)