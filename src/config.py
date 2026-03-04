from functools import lru_cache
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# -----------------------------
# Database Configuration (PLAIN MODEL)
# -----------------------------
class DatabaseConfig(BaseModel):
    host: str
    port: int
    name: str
    user: str
    password: str

    @property
    def url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


# -----------------------------
# Gemini Configuration (PLAIN MODEL)
# -----------------------------
class GeminiConfig(BaseModel):
    api_key: str


# -----------------------------
# Global App Settings (ONLY ONE BaseSettings)
# -----------------------------
class Settings(BaseSettings):
    # Database
    db_host: str = Field(alias="DB_HOST")
    db_port: int = Field(alias="DB_PORT")
    db_name: str = Field(alias="DB_NAME")
    db_user: str = Field(alias="DB_USER")
    db_password: str = Field(alias="DB_PASSWORD")

    # Gemini
    gemini_api_key: str = Field(alias="GEMINI_API_KEY")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "forbid",
    }

    @property
    def database(self) -> DatabaseConfig:
        return DatabaseConfig(
            host=self.db_host,
            port=self.db_port,
            name=self.db_name,
            user=self.db_user,
            password=self.db_password,
        )

    @property
    def gemini(self) -> GeminiConfig:
        return GeminiConfig(api_key=self.gemini_api_key)


# -----------------------------
# Cached Settings Loader
# -----------------------------
@lru_cache
def get_settings() -> Settings:
    return Settings()