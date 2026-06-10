FROM python:3.11-slim

# poppler-utils: PDF -> image rendering for pdf2image
RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
