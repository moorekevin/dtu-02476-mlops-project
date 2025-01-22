# Base image
# FROM python:3.11-slim AS base

# RUN apt update && \
#     apt install --no-install-recommends -y build-essential gcc && \
#     apt clean && rm -rf /var/lib/apt/lists/*

# COPY src src/
# COPY requirements.txt requirements.txt
# COPY requirements_dev.txt requirements_dev.txt
# COPY README.md README.md
# COPY pyproject.toml pyproject.toml

# WORKDIR /
# RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
# RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_dev.txt
# RUN pip install . --no-deps --no-cache-dir

# ENTRYPOINT ["python", "-u", "src/mlops/train.py"]

# Base image
FROM python:3.11-slim AS base


# Install system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy project files
COPY src src/
COPY data data/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY .git .git
COPY .dvc .dvc
COPY .dvcignore .dvcignore

# Install Python dependencies
RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose


# Run dvc pull to fetch data
RUN dvc pull

# Add entrypoint to run scripts
# ENTRYPOINT ["python", "-u", "src/final_project/data.py"]
ENTRYPOINT ["sh", "-c", " python src/final_project/data.py && python src/final_project/train.py && python src/final_project/evaluate.py"]
