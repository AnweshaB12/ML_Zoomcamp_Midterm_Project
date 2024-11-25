FROM python:3.10-slim

WORKDIR /app

COPY . /app

# Install system dependencies (if required for your Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in a virtual environment
RUN python -m venv venv

# Install the Python packages and Gunicorn in the virtual environment
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt && \
    /app/venv/bin/pip install gunicorn

EXPOSE 8080

# Set the environment variables to use the virtual environment
ENV VIRTUAL_ENV=/app/venv
ENV PATH="/app/venv/bin:$PATH"

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8080", "predict:app"] 