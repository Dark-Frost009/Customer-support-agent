# Customer support agent API.
#
#   docker build -t support-agent .
#   docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... -v support-data:/data support-agent
#
# Databases live in /data (a volume), so they survive container restarts.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SUPPORT_DB_PATH=/data/support.db \
    POLICY_INDEX_PATH=/data/policy_index.db

WORKDIR /srv

# Install dependencies first so code changes don't invalidate this layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ app/
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Run as an unprivileged user; it owns only the data directory.
# (sed strips Windows line endings in case the script was edited on Windows.)
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data && chown appuser:appuser /data \
    && sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh
USER appuser
VOLUME ["/data"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "main:app", "--app-dir", "app", "--host", "0.0.0.0", "--port", "8000"]
