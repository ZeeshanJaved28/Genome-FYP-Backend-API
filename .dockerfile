# Use Python base image
FROM python:3.10-slim

# Set workdir
WORKDIR /code

# Copy app files
COPY . /code

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (Hugging Face uses port 7860)
EXPOSE 7860

# Command to run your Flask app via gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "genome-app:app"]
