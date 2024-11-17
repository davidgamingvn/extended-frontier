# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory to /server
WORKDIR /app

# Copy the current directory contents into the container at /server
COPY ./server/  /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y curl && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=2.py

# Run app.py when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]