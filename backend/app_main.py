"""
Minimal FastAPI app exposing just the personalization routes.

This bypasses the broken pgx/narrative imports and serves the core
digital twin API on port 8000.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="BioDigitalTwin - Personalization API", version="6.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and mount personalization router
from app.personalization import personalization_router  # noqa: E402
app.include_router(personalization_router)

# Try to include phase 5 router
try:
    from app.personalization import phase5_router  # noqa: E402
    app.include_router(phase5_router)
except Exception as e:
    print(f"Phase 5 router not loaded: {e}")


@app.get("/")
def root():
    return {
        "service": "BioDigitalTwin Platform",
        "version": "6.0.0",
        "status": "running",
        "endpoints": [
            "/docs - Swagger UI",
            "/redoc - ReDoc",
            "/openapi.json - OpenAPI schema",
            "/personalization - Personalization routes",
            "/phase5 - Phase 5 routes",
        ],
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
