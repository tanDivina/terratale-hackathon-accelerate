# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend application code into the container
COPY ./backend ./backend

# Expose the port the app runs on
EXPOSE 8080

# The command to run the application
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}
