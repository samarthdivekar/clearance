FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
RUN pip install --no-cache-dir -e ".[ml]"

# The index is built outside the image and mounted at /app/data:
#   docker run -p 8000:8000 -v $(pwd)/data:/app/data -e ANTHROPIC_API_KEY=... clearance
VOLUME ["/app/data"]
EXPOSE 8000
CMD ["clearance", "serve", "--host", "0.0.0.0", "--port", "8000"]
