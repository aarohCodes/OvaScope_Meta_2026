# OvaScope API + worker image (CPU). Research prototype; local-only by default.
FROM python:3.11.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OVASCOPE_DATA_DIR=/data

RUN useradd --create-home --uid 10001 ovascope && mkdir -p /data && chown ovascope /data
WORKDIR /app

# Hash-pinned dependencies (CPU torch wheels from the PyTorch index).
COPY requirements/lock.txt requirements/lock.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/lock.txt

COPY pyproject.toml README.md ./
COPY ml ml
COPY services services
COPY configs configs
RUN pip install --no-deps -e .

USER ovascope
EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["uvicorn", "ovascope_api.main:get_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
