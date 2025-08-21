FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
RUN pip install -r requirements.txt
COPY . .

CMD ['uvicorn', 'main:app', '--reload', '--host 0.0.0.0', '--port 8000']