FROM python:3.11-slim

# poppler-utils: PDF -> image rendering for pdf2image
# openjdk + audiveris: local transcription for /analyze-structure —
#   the remote audiveris service returns MusicXML only, and the
#   structure pass needs the .omr project file
# tesseract-ocr: number/word OCR for the structure pass
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        poppler-utils git libgl1 libglib2.0-0 \
        tesseract-ocr wget \
        libgtk-3-0 libxext6 libxrender1 libxtst6 libxi6 \
    && rm -rf /var/lib/apt/lists/*

# Audiveris 5.11 (same version as the validated local bench). The deb's
# post-install script fails on slim images (desktop hooks), so extract
# the payload directly; it bundles its own Java runtime and tesseract
# JNI, needing only the GTK/X11 libs above even in -batch mode.
RUN wget -q https://github.com/Audiveris/audiveris/releases/download/5.11.0/Audiveris-5.11.0-ubuntu22.04-x86_64.deb \
    && dpkg-deb -x Audiveris-5.11.0-ubuntu22.04-x86_64.deb / \
    && rm Audiveris-5.11.0-ubuntu22.04-x86_64.deb

# Audiveris runs tesseract in LEGACY mode: it needs the FULL traineddata
# from tesseract-ocr/tessdata (the apt/LSTM-only files fail silently and
# every title/tempo text is dropped). Never register `heb` here —
# Audiveris 5.11's WordScanner dies on RTL and takes all TEXTS with it.
RUN mkdir -p /usr/local/share/audiveris-tessdata \
    && wget -q -O /usr/local/share/audiveris-tessdata/eng.traineddata \
        https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PORT=8080
ENV AUDIVERIS_BIN=/opt/audiveris/bin/Audiveris
ENV TESSDATA_DIR=/usr/local/share/audiveris-tessdata
ENV TESSERACT_BIN=/usr/bin/tesseract
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
