import typing as T
from functools import lru_cache

import torch
from service_streamer import ManagedModel, Streamer
from fastapi.logger import logger
from ..settings import get_settings
from ..jamo import JAMO

env = get_settings()


class JamoModelManager(ManagedModel):
    def init_model(self):
        self.jamo = JAMO.from_pretrained(env.JAMO_MODEL_SIZE, env.JAMO_MODEL_PATH, env.DEVICE)

    @torch.inference_mode()
    def predict(self, inputs: T.List[torch.Tensor]) -> T.List[torch.Tensor]:
        logger.info(f"batch size: {len(inputs)}")
        results = []
        try:
            batch = torch.cat(inputs, 0).to(env.DEVICE)
            print("batch_size:", batch.shape)
            pred = self.classifier(batch)
            prob = torch.softmax(pred, dim=1)
            prob = prob.cpu().numpy()
            results = [output for output in prob]
        except Exception as e:
            logger.error(f"Error {self.__class__.__name__}: {e}")
        return results


@lru_cache(maxsize=1)
def get_jamo_streamer():
    streamer = Streamer(
        JamoModelManager,
        batch_size=env.MB_BATCH_SIZE,
        max_latency=env.MB_MAX_LATENCY,
        worker_num=env.MB_WORKER_NUM,
    )
    return streamer
