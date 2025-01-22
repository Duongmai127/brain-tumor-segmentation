from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    model_device: str = "cuda"

    class Config:
        env_file = ".env"


settings = Settings()
