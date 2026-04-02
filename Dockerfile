FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Chromium + its system-level deps for Playwright
RUN playwright install --with-deps chromium

# Copy application code
COPY . .

EXPOSE 8080

CMD ["python", "bot.py"]
