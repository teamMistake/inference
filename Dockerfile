FROM tiangolo/uvicorn-gunicorn:python3.10

WORKDIR /app/

COPY ./requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

COPY ./app /app/app
COPY ./hg_tokenizer /app/hg_tokenizer
COPY ./model_store /app/model_store

