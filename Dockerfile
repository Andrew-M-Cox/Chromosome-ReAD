FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# we probably need build tools?
RUN apt-get update \
    && apt-get install --yes \
    gcc \
    g++ \
    build-essential \
    python3-dev

WORKDIR /app
COPY requirements.txt requirements.txt

# COPY packages.txt packages.txt
# RUN xargs -a packages.txt apt-get install --yes

# RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install --upgrade -r requirements.txt

EXPOSE 8501

COPY . .

CMD ["streamlit", "run", "streamlit_app.py"]

# Build image:
# docker build --progress=plain --tag read:latest .
# Run container:
# docker run -ti -p 8501:8501 --rm read:latest
# Run container with mounted volume:
# docker run -ti -p 8501:8501 -v ${pwd}:/app --rm read:latest
# Run container with shell:
# docker run -ti -p 8501:8501 --rm read:latest /bin/bash
# Run container with mounted volume and interactive shell:
# docker run -ti -p 8501:8501 -v ${pwd}:/app --rm read:latest /bin/bash
