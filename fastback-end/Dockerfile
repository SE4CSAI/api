FROM python:3.8

RUN pip install --upgrade pip
WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./src ./src/
COPY ./main.py ./main.py
COPY ./src/app/app.py ./app.py

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "src.app.app:app"]