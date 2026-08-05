FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python scripts/verify_assets.py
EXPOSE 8765
CMD ["python", "launch.py", "--host", "0.0.0.0", "--port", "8765", "--no-browser"]
