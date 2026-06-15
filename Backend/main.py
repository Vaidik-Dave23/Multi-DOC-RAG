from fastapi import FastAPI
from app.routes.upload import router as upload_router
from app.routes.query import router as query_router
from app.routes.extract import router as extract_router
from app.routes.compare import router as compare_router
from app.routes.feedback import router as feedback_router
from app.store import load_store, save_store
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.include_router(upload_router)
app.include_router(query_router)
app.include_router(extract_router)
app.include_router(compare_router)
app.include_router(feedback_router)


@app.on_event("startup")
async def startup():
    load_store()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)