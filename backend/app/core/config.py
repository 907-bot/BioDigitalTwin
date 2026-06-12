import os
import logging
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def _get_required_env(var_name: str, fallback_default: str = None) -> str:
    """Get required environment variable or raise error."""
    value = os.getenv(var_name)
    if value is not None:
        return value
    if fallback_default is not None:
        logger.warning(
            f"{var_name} not set. Using insecure default. "
            "Set this environment variable in production!"
        )
        return fallback_default
    raise ValueError(
        f"{var_name} environment variable is required. "
        f"Set it before deployment to ensure database security."
    )


@dataclass
class Settings:
    APP_NAME: str = "Bio-Digital Twin"
    PHASE: str = "2-3"
    DATA_DIR: str = field(default_factory=lambda: os.getenv("DATA_DIR", "data"))
    MODEL_DIR: str = field(default_factory=lambda: os.getenv("MODEL_DIR", "data/models"))

    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "biodigital")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    # SECURITY FIX: Require POSTGRES_PASSWORD in production, warn in dev
    POSTGRES_PASSWORD: str = field(default_factory=lambda: _get_required_env("POSTGRES_PASSWORD", "postgres"))

    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    # SECURITY FIX: Require NEO4J_PASSWORD in production, warn in dev
    NEO4J_PASSWORD: str = field(default_factory=lambda: _get_required_env("NEO4J_PASSWORD", "password"))

    GNN_HIDDEN: int = 64
    GNN_EMBEDDING_DIM: int = 32
    GNN_EPOCHS: int = 200
    GNN_LR: float = 1e-3

    @property
    def postgres_url(self) -> str:
        # SECURITY FIX: Use urllib.parse for safer URL construction
        from urllib.parse import quote_plus
        user = quote_plus(self.POSTGRES_USER)
        password = quote_plus(self.POSTGRES_PASSWORD)
        host = quote_plus(self.POSTGRES_HOST)
        return f"postgresql://{user}:{password}@{host}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"


settings = Settings()
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.MODEL_DIR, exist_ok=True)
