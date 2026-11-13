FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    TZ=Asia/Seoul \
    OMP_NUM_THREADS=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 libgl1 libglu1-mesa tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY deploy/requirements*.txt /app/deploy/
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/deploy/requirements.txt \
    && python -c "import sumo,subprocess; from pathlib import Path; subprocess.run([str(Path(sumo.SUMO_HOME)/'bin'/'sumo'),'--version'],check=True)"

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /app/results \
    && chown app:app /app/results
COPY --chown=app:app CODE/ /app/CODE/
COPY --chown=app:app web/ /app/web/
COPY --chown=app:app DATASET/ /app/DATASET/
COPY --chown=app:app SUMO/ /app/SUMO/
COPY --chown=app:app deploy/entrypoint.py /app/deploy/entrypoint.py
COPY --chown=app:app deploy/seed-results/ /app/seed-results/

USER app
EXPOSE 8000
ENTRYPOINT ["python", "-B", "/app/deploy/entrypoint.py"]
CMD ["python", "-B", "-m", "uvicorn", "dashboard_asgi:app", "--app-dir", "/app/CODE", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "64", "--timeout-keep-alive", "10"]
