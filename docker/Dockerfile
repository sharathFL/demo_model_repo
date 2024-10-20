FROM python:3.9-slim

WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y git

# Install DVC
RUN pip install dvc

CMD ["tail", "-f", "/dev/null"]  # Keeps the container running

