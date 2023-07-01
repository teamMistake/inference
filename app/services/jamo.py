import typing as T
import torch
from fastapi import Depends
from fastapi.logger import logger
from ..settings import get_settings
from ..managers import (
    get_jamo_streamer
)
from ..schema import ChatResponseProtocol

env = get_settings()


class JamoService:
    def __init__(
        self,
        jamo_streamer=Depends(get_jamo_streamer),
    ):
        logger.info(f"DI: {self.__class__.__name__}")
        self.model = jamo_streamer

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        temperature: float=1.0,
        top_k=None,
        eos_id=None
    ) -> T.List[list]:
        T = idx.size(0)
        max_seq_length = self.model.config.block_size

        device, dtype = idx.device, idx.dtype

        # generate max_new_tokens tokens
        for _ in range(max_new_tokens):
            # forward
            logits = self.model(x)
            logits = logits[0, -1] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)
            
            # advance
            input_pos = input_pos[-1:] + 1
            # concatenate the new generation
            idx = idx.index_copy(0, input_pos, idx_next)
            return idx[:input_pos] 

        embedding = self.embedding_streamer.predict([image])[0]
        embedding = torch.unsqueeze(embedding, dim=0)
        prob = self.classifier_streamer.predict([embedding])[0]
        top_prob, top_catid = torch.topk(prob, k)
        top_prob = [prob.item() for prob in top_prob]
        results = [
            ChatResponseProtocol(name=self.classes[index], prob=prob)
            for prob, index in zip(top_prob, top_catid)
        ]
        return results
