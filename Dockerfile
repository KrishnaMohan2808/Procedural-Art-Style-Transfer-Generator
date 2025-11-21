# Use the stable, lightweight base Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# --- FIX: Robust Dependency Installation ---
# 1. Install Numpy and Pillow first (Core dependencies)
RUN pip install --no-cache-dir numpy==1.26.4 Pillow

# 2. Install PyTorch CPU versions explicitly
# We use --index-url to force looking at the PyTorch CPU wheel repository
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# 3. Install remaining dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (main.py, model files, styles folder, etc.)
COPY . .

# Expose the port where FastAPI will run (Cloud Run expects 8080)
EXPOSE 8080

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]