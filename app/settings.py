import typing as T
from functools import lru_cache
from pydantic import BaseSettings


class ModelSettings(BaseSettings):
    JAMO_MODEL_PATH = "model_store/model.onnx"
    JAMO_MODEL_SIZE = "small"


class MicroBatchSettings(BaseSettings):
    MB_BATCH_SIZE = 64
    MB_MAX_LATENCY = 0.2 
    MB_WORKER_NUM = 1


class DeviceSettings(BaseSettings):
    DEVICE = "cpu"


class Settings(
    ModelSettings,
    MicroBatchSettings,
    DeviceSettings,
):
    CORS_ALLOW_ORIGINS: T.List[str] = [
        "*",
    ]


@lru_cache()
def get_settings():
    setting = Settings()
    return setting