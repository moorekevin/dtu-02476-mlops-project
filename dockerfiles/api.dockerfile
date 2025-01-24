# Use Python 3.11-slim as the base image
FROM python:3.11-slim AS base

# Install necessary dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy model files
COPY models/ models/

# Copy the source code
COPY src/ src/

# Copy the other necessary files
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

# Install required Python packages
RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

# Set the entry point for the container
ENTRYPOINT ["uvicorn", "src.final_project.api:app", "--host", "0.0.0.0", "--port", "8000"]
