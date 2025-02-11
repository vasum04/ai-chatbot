# Use Python 3.10 for compatibility
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the source code
COPY . .

# Expose the API port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

