# main.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from agents import TutorAgent, BioUser

# Initialize FastAPI
app = FastAPI(title="Bioinformatics Tutor API")

# Initialize agents
user_agent = BioUser()
tutor_agent = TutorAgent()

# Request model
class QueryRequest(BaseModel):
    question: str

# Response model
class QueryResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=QueryResponse)
def ask_tutor(request: QueryRequest):
    """
    Ask the Bioinformatics Tutor a question.
    """
    answer = tutor_agent.process_query(request.question)
    return QueryResponse(answer=answer)

@app.get("/")
def root():
    return {"message": "Bioinformatics Tutor API is running."}

