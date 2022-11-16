FROM python:3.8

WORKDIR /code

COPY ../requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./src ./src/
COPY main.py ./main.py
COPY app/app.py ./app.py

CMD ["python", "main.py"]