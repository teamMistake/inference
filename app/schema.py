from pydantic import BaseModel

## Chat Request Protocol
## Header: "req_id": "to pass back"
class ChatRequestProtocol(BaseModel):
    req: str # request prompt
    context: str # previous dialogue
    stream: bool # streaming creation
    max_token: int # max token in generating step

## Chat Response Protocol
## Header: "req_id": "req id from earlier"
##         "seq_id": "incrementing integer per req_id"
class ChatResponseProtocol(BaseModel):
    resp_partial: str # for the streaming
    resp_full: str # full text
    eos: bool # depending on if this is end of stream