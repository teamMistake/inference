import typing as T
from functools import lru_cache

import torch
from fastapi.logger import logger
from ..settings import get_settings
from ..jamo import JAMO

env = get_settings()

class JamoModelManager():
    def __init__(self): self.init_model()

    def init_model(self):
        self.model = JAMO.from_pretrained(env.JAMO_MODEL_SIZE, env.JAMO_MODEL_PATH, env.DEVICE)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, input: torch.Tensor, max_seq_length: int, input_pos: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
        next_token_idx = 0
        try:
            logits = self.model(input, max_seq_length, input_pos)
            logits = logits[0, -1] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
        except Exception as e:
            logger.error(f"Error {self.__class__.__name__}: {e}")
        return next_token_idx

    def clean_cache(self):
        self.model.reset_cache()

@lru_cache(maxsize=1)
def get_jamo_manager():
    manager = JamoModelManager()

    return manager