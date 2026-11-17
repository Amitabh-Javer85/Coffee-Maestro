FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt \
    && useradd --create-home appuser \
    && install -d -o appuser -g appuser /app/Logs
COPY --chown=appuser:appuser backend/ ./backend/
COPY --from=frontend /build/dist ./frontend/dist
USER appuser
WORKDIR /app/backend
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
