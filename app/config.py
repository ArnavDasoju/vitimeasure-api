"""
VITImeasure API — configuration from environment variables.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Cosmos DB
    cosmos_db_connection_string: str = ""
    cosmos_db_database_name: str = "vitiligo_db"

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


settings = Settings()
