from fastapi import FastAPI
from app.routers import predictions

app = FastAPI(
    title="AI Services API",
    description="API pour les prédictions et l'inférence IA.",
    version="1.0.0",
)

# Inclure les routes
app.include_router(predictions.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Services API"}
