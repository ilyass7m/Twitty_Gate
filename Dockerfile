# Use the official Python image from the Docker Hub
FROM python:3.9

# Set working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application to the container
COPY . /app/

# Expose the port where the Dash app will run
EXPOSE 8050

# Define the command to start the Dash application
CMD ["python", "app.py"]
