"""
VITImeasure API — configuration from environment variables.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = ""

    # Auth
    jwt_secret: str = "CHANGE_ME_IN_PRODUCTION"
    jwt_expires_in_days: int = 7

    # Server
    port: int = 3000
    allowed_origins: str = "*"

    # OpenAI
    openai_api_key: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def cors_origins(self) -> list[str]:
        if self.allowed_origins == "*":
            return ["*"]
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def async_database_url(self) -> str:
        """Convert postgresql:// to postgresql+asyncpg:// for async SQLAlchemy."""
        url = self.database_url
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url


settings = Settings()
