FROM python:3.11-alpine

RUN apt-get update && apt-get install -y --no-install-recommends gcc

WORKDIR /app

COPY requirements.txt .

RUN python3 -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python3", "app.py"]