from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import coach_routes, upload_routes

app = FastAPI(title="Interview Practice Listener API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_routes.router, prefix="/upload", tags=["Upload"])
app.include_router(coach_routes.router, prefix="/coach", tags=["Coach"])

@app.get("/")
def health_check():
    return {"status": "ok", "app": "Interview Practice Listener"}
