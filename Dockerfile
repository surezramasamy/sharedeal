# Use slim base for smaller size
FROM python:3.11-slim

# Install system deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements and install (excluding torch)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install CPU-only PyTorch separately (this works reliably)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy app code
COPY . .

# Expose port
EXPOSE 8000

# Run app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]