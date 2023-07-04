import typing as T
from functools import lru_cache, partial
from fastapi import (
    Depends,
    HTTPException,
    status,
    Header
)

from fastapi.responses import StreamingResponse, JSONResponse
import json
from fastapi_restful.cbv import cbv

from fastapi_restful.inferring_router import InferringRouter
from fastapi.logger import logger

import torch
from transformers import AutoTokenizer, GPT2TokenizerFast

from ..services import JamoService
from ..schema import ChatRequestProtocol, ChatResponseProtocol

router = InferringRouter()

@lru_cache(maxsize=1)
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("hg_tokenizer")
    return tokenizer


@cbv(router)
class JamoRouter:
    jamo: JamoService = Depends()
    tokenizer: GPT2TokenizerFast = Depends(get_tokenizer)

    @router.post("/generate")
    def generate(
        self,
        req_id: str | None = Header(default=None, convert_underscores=False),
        target_model: str | None = Header(default=None, convert_underscores=False),
        request: ChatRequestProtocol = ChatRequestProtocol(req="", context="", stream=False, max_token=0)
    ):
        EOS_TOKEN = "<s>"
        req_prompt = request.req
        parsed_prompt = JamoRouter.parsing_prmopt(req_prompt)
        parsed_prompt = f"{EOS_TOKEN} {parsed_prompt}"
        max_token = request.max_token
        streaming = request.stream 
        context = request.context

        prompt_idx = self.encode(parsed_prompt).squeeze(0)
        kwgs = {"idx":prompt_idx, "max_token":max_token}

        headers = {"req_id": req_id, "seq_id": "0"}
        if streaming:
            
            def chat_streaming():
                nonlocal headers
                seq_id = 0

                cur = len(parsed_prompt)
                full_answer = ""

                try:
                    for predicted_idx in self.jamo.streaming_generate_idx(**kwgs):
                        if predicted_idx == None:
                            raise StopIteration
                        target = self.decode(predicted_idx)
                        target = target[:-1]
                        full_answer = JamoRouter.clean_response(target)
                        new = target[cur:]
                        cur = len(target)
                        yield json.dumps({"resp_partial":new, "resp_full":full_answer, "eos":False}) 

                        seq_id += 1
                        headers["seq_id"] = str(seq_id)

                except StopIteration:
                    yield json.dumps({"resp_partial":"", "resp_full":full_answer, "eos":True})

            return StreamingResponse(chat_streaming(), media_type="application/x-ndjson", headers=headers)
        
        predicted_idx = self.jamo.generate_idx(**kwgs)
        predicted_text = self.decode(predicted_idx)
        answer = JamoRouter.clean_response(predicted_text)

        content = ChatResponseProtocol(resp_partial="", resp_full=answer, eos=True).json()

        return JSONResponse(content=content, headers=headers)

    def encode(self, text) -> torch.Tensor:
        try:
            token = self.tokenizer.encode(text, return_tensors="pt")    
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail=f"Please Check the tokenize encode process.",
            )
        return token
    
    def decode(self, idx) -> str:
        try:
            text = self.tokenizer.decode(idx)    
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail=f"Please Check the tokenize decode process.",
            )
        return text
    
    @staticmethod
    def parsing_prmopt(instruction):
        chat_parser = (
            "명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
            "### 명령어:\n{instruction}\n\n### 응답:\n"
        )

        parsed_prompt = chat_parser.format_map({"instruction":instruction})

        return parsed_prompt

    @staticmethod
    def clean_response(answer):
        splitted = answer.strip().split("### 응답:")
        cleaned_answer = splitted[-1].rstrip().strip() if len(splitted) > 1 else ""

        return cleaned_answer