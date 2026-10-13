# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS dependencies
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY requirements-runtime.txt ./
RUN python -m venv /opt/venv && /opt/venv/bin/pip install -r requirements-runtime.txt

FROM python:3.12-slim-bookworm AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 ca-certificates locales \
    && sed -i 's/^# *en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 radnote && useradd --uid 10001 --gid radnote --create-home radnote
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 RADNOTE_CPU_THREADS=2 RADNOTE_ENABLE_OCR=1 RADNOTE_MODEL_PRECISION=int8 \
    RADNOTE_OCR_MODEL_DIR=/tmp/radnote-ocr/models RADNOTE_OCR_USER_DIR=/tmp/radnote-ocr/network \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
WORKDIR /app
COPY --from=dependencies /opt/venv /opt/venv
COPY api/ api/
COPY radnote/ radnote/
COPY client/ client/
COPY web/ web/
COPY models/ models/
COPY samples/ samples/
COPY scripts/ scripts/
USER radnote
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=90s --retries=12 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4)" || exit 1
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]

FROM python:3.12-slim-bookworm AS client
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME=/tmp
WORKDIR /app
RUN groupadd --gid 10001 radnote && useradd --uid 10001 --gid radnote --create-home radnote
COPY client/ client/
COPY web/ web/
COPY samples/ocr/images/ samples/ocr/images/
USER radnote
EXPOSE 8765
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=12 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/ready', timeout=4)" || exit 1
CMD ["python", "client/web_client.py", "--server", "http://backend:8000", "--host", "0.0.0.0", "--port", "8765"]
