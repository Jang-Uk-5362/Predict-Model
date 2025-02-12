# Base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy requirement file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Command to run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

