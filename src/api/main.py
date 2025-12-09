from fastapi import FastAPI
from src.api.endpoints import router

app = FastAPI(title="ECG Classification API")
app.include_router(router, prefix="/api")
