# Use a base image
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure required system dependencies are installed for PostgreSQL

RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

# Download necessary NLTK data
RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/usr/local/nltk_data')"

ENV NLTK_DATA="/usr/local/nltk_data"

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

