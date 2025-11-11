FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y build-essential
RUN apt-get update && apt-get install -y gfortran
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["python", "main.py"]
