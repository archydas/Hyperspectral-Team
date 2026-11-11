# DS3 Library — web application + API in one container (the database runs in the `db` container, see docker-compose.yml)
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements-server.txt pyproject.toml README.md ./
COPY src ./src
RUN pip install -r requirements-server.txt
COPY material_library ./material_library
ENV PYTHONPATH=/app/material_library/backend:/app/src \
    AUTO_IMPORT_DIR=/libraries
EXPOSE 8000
WORKDIR /app/material_library/backend
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
