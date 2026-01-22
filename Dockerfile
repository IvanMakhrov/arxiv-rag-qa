FROM python:3.12-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -e .[api]

EXPOSE 8000

CMD ["uvicorn", "arxiv_rag_qa.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
