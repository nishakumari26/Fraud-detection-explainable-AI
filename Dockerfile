# =============================================================
# Dockerfile for Fraud Detection with Explainable AI
# =============================================================
#
# WHAT IS DOCKER?
# ---------------
# Docker packages your entire project (code + dependencies)
# into a "container" that can run on ANY computer.
# Think of it as a portable box that has everything needed
# to run your project — no "it works on my machine" issues.
#
# HOW TO USE:
# docker build -t fraud-detection .
# docker run -p 8501:8501 -p 8000:8000 fraud-detection
#
# Then visit:
#   Dashboard: http://localhost:8501
#   API:       http://localhost:8000/docs
# =============================================================

# Use Python 3.11 slim image (smaller, faster)
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first (Docker caches this layer)
# This means if you change code but NOT requirements,
# Docker won't reinstall packages (saves time!)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose ports:
# 8501 = Streamlit dashboard
# 8000 = FastAPI
EXPOSE 8501 8000

# Default command: run the Streamlit dashboard
# --server.address=0.0.0.0 makes it accessible from outside
CMD ["streamlit", "run", "dashboard/dashboard.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
