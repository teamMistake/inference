import typing as T
from fastapi import (
    Depends,
    HTTPException,
    status,
    Body,
)
from fastapi_restful.cbv import cbv

from fastapi_restful.inferring_router import InferringRouter
from fastapi.logger import logger
from PIL import Image

from transformers import AutoTokenizer, GPT2TokenizerFast

from ..settings import get_settings
from ..services import JamoService
from ..schema import ChatRequestProtocol, ChatResponseProtocol


router = InferringRouter()
setting = get_settings()


@cbv(router)
class Jamo:
    jamo: JamoService = Depends()
    tokenizer: GPT2TokenizerFast = Depends()

    @router.post("/generate", response_model=T.List[ChatResponseProtocol])
    def predict_tag(
        self,
        request: ChatRequestProtocol = ChatRequestProtocol(prompt="")
    ):
        logger.info("------------- Generation Start -----------")
        # image = self.imread(image)
        answer = jamo.
        logger.info("------------- Generation Done -----------")

    @staticmethod
    def tokenize(text):
        try:
            token = 
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail=f"Please Check the tokenization process.",
            )
        return token