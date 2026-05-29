FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install mlflow pandas scikit-learn

CMD ["python", "modelling.py"]