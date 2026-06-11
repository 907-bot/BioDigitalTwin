"""Shared configuration loaded from environment."""
from dataclasses import dataclass, field
import os


@dataclass
class Settings:
    postgres_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/biodigital"))
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://neo4j:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password"))
    redis_host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "redis"))
    redis_port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://host.docker.internal:11434"))
    service_port: int = field(default_factory=lambda: int(os.getenv("SERVICE_PORT", "8000")))

    @classmethod
    def from_env(cls) -> "Settings":
        return cls()
