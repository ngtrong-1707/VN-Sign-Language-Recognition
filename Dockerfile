FROM python:3.12-slim

# Prevent python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for OpenCV & font rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU directly first to keep image lightweight
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy project specification & install requirements
COPY pyproject.toml .env.example ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Copy application assets, templates, static files, and checkpoints
COPY assets/ ./assets/
COPY models/ ./models/
COPY static/ ./static/
COPY templates/ ./templates/

EXPOSE 8000

CMD ["uvicorn", "vsl.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
