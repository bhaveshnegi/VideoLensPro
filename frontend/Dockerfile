# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the frontend files
COPY . .

# Expose the port
EXPOSE 9000

# Start a simple HTTP server
CMD ["python", "-m", "http.server", "9000"]
