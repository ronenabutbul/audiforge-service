FROM python:3.11-slim

# poppler-utils: PDF -> image rendering for pdf2image
# openjdk + audiveris: local transcription for /analyze-structure —
#   the remote audiveris service returns MusicXML only, and the
#   structure pass needs the .omr project file
# tesseract-ocr: number/word OCR for the structure pass and the chart
#   readers (its eng/heb data is replaced below)
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

# Our own OCR (fix_* modules and the chart readers) reads the SYSTEM tessdata
# (TESSDATA_DIR below), not Audiveris'. The apt package's eng is the small
# LSTM-only model; the bench reads with the full one, and with it the drum
# and chord readers give the same file in this image as on the Mac (with
# the apt model, rehearsal letters, ending numbers and tempo marks read
# differently on 6 of 7 drum charts). Same files as the bench, checked by
# hash. heb is for Hebrew titles (fix_titles reads eng+heb); it goes in this
# directory only, never in Audiveris' (see above).
RUN cp /usr/local/share/audiveris-tessdata/eng.traineddata /usr/share/tesseract-ocr/5/tessdata/eng.traineddata \
    && wget -q -O /usr/share/tesseract-ocr/5/tessdata/heb.traineddata \
        https://github.com/tesseract-ocr/tessdata/raw/main/heb.traineddata \
    && echo "007b522901a665bc2037428602d4d527f5ead7ed  /usr/share/tesseract-ocr/5/tessdata/eng.traineddata" | sha1sum -c - \
    && echo "c6be68caca2343dc3884e596ac620deed06e4400  /usr/share/tesseract-ocr/5/tessdata/heb.traineddata" | sha1sum -c -

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
# The chart readers (drums, chord charts, scores) and their entry point.
# They live in local_bench/ beside app/, as in the repository: the readers
# find app/ as ../app, and the service runs them as child processes through
# convert_any.py (app/chart_routing.py). One file list, kept in step with
# .dockerignore; tests/test_routing.py checks both against the imports.
COPY local_bench/convert_any.py local_bench/chart_kind.py local_bench/drum_grid.py local_bench/drum_ink.py local_bench/drum_overlay.py local_bench/chord_chart.py local_bench/partitura.py ./local_bench/

ENV PORT=8080
ENV AUDIVERIS_BIN=/opt/audiveris/bin/Audiveris
# Audiveris runs tesseract in legacy mode and needs the full data;
# our own OCR needs the system tessdata, whose configs/ directory is
# what makes tesseract's TSV output work.
ENV AUDIVERIS_TESSDATA=/usr/local/share/audiveris-tessdata
ENV TESSDATA_DIR=/usr/share/tesseract-ocr/5/tessdata
ENV TESSERACT_BIN=/usr/bin/tesseract
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
