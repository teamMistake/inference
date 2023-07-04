FROM tiangolo/uvicorn-gunicorn:python3.10

COPY ./requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

COPY ./app /app
COPY ./hg_tokenizer /hg_tokenizer
COPY ./model_store /model_store