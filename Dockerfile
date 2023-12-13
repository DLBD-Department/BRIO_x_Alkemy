FROM python:3.10
RUN apt-get update && apt-get install -y

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
ENV PYTHONPATH=/app
WORKDIR /app/frontend

EXPOSE 5000

CMD ["python", "app.py"]
