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


# Alias endpoints — resolve 404s for un-prefixed routes
@app.get("/mimic/summary")
async def get_mimic_summary_alias():
    """Alias for /phase17/mimic/summary."""
    try:
        from app.mimic import get_mimic_connector
        connector = get_mimic_connector()
        return await connector.get_mimic_summary_stats()
    except Exception as e:
        return {"status": "unavailable", "message": str(e)}


@app.get("/personalization/validation/summary")
async def get_validation_summary_alias():
    """Alias for /personalization/v4/validation/results."""
    try:
        from app.personalization.phase5.validation_framework import ValidationFrameworkV2
        vf = ValidationFrameworkV2()
        vf.run_all()
        return vf.get_report()
    except Exception as e:
        return {"status": "unavailable", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
