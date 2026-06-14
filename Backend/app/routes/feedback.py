from datetime import datetime
from app.store import feedback_log, responses_log
from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter()

class FeedbackInput(BaseModel):
    response_id: str
    helpful: bool

@router.post("/feedback")
async def feedback(fb: FeedbackInput):
    if fb.response_id not in responses_log:
        return {"error": "response_id not found"}
    
    feedback_log.append({
        "response_id": fb.response_id,
        "helpful": fb.helpful,
        "timestamp": datetime.now().isoformat(),
        "original_query": responses_log[fb.response_id]["query"]
    })
    
    return {
        "message": "Feedback recorded",
        "response_id": fb.response_id,
        "helpful": fb.helpful
    }