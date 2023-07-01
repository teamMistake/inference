from pydantic import BaseModel


class ChatRequestProtocol(BaseModel):
    prompt: str

class ChatResponseProtocol(BaseModel):
    prompt: str
    answer: str
    response_time: float