# Use a specific, stable Python base image (slim keeps the image small)
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file first to leverage Docker caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
     -f https://download.pytorch.org/whl/torch_stable.html


# Copy the rest of the project files (main.py, model files, styles folder, etc.)
COPY . .

# Expose the port where FastAPI will run
EXPOSE 8080

# Command to run the FastAPI app with Uvicorn
# --host 0.0.0.0 allows external access from your host machine
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
