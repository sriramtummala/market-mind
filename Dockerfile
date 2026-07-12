FROM python:3.14-slim

WORKDIR /app

# Day 8 document intelligence needs the Tesseract binary (not just the
# Python wrapper) for OCR fallback on scanned documents.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the vector index at image build time so the container is
# ready to serve immediately (swap this for a startup-time build if
# your document corpus updates more often than your deploys).
RUN python3 src/ingest.py

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
