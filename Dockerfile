FROM python:latest
COPY . /app
COPY ./datasets /app/datasets
RUN pip install -r app/requirements.txt
ENTRYPOINT ["python", "app/src/index.py", "housing.csv"]
