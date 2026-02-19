FROM python:3.10-slim-bullseye

WORKDIR /app

# Environment variables for Python optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install minimal system dependencies (dlib pre-compiled, no build needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install pre-compiled dlib binary (no compilation needed)
RUN pip install --upgrade pip setuptools wheel

RUN pip install dlib==19.24.2 --extra-index-url https://pypi.org/simple/

RUN pip install face_recognition==1.3.0

RUN pip install fastapi==0.104.1 uvicorn==0.24.0 python-multipart==0.0.6 numpy==1.24.3 Pillow==10.0.0 pymongo==4.6.0 python-dotenv==1.0.0

RUN pip cache purge

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "120"]
