FROM python:3.12-slim

WORKDIR /app

COPY pyrfr-0.9.0-cp312-cp312-linux_x86_64.whl requirements.txt ./
RUN pip install --no-cache-dir pyrfr-0.9.0-cp312-cp312-linux_x86_64.whl && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "run.py", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
