from fastapi import FastAPI
from app.routes.upload import router as upload_router
from app.routes.query import router as query_router

app = FastAPI()
app.include_router(upload_router)
app.include_router(query_router)