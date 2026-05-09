# ArXiv RAG QA

**RAG-система для анализа научных статей из ArXiv**

Данный проект предоставляет исследователям возможность развернуть полноценный
пайплайн Retrieval-Augmented Generation (RAG). Система позволяет загружать
PDF-статьи с ArXiv, парсить их, разбивать на чанки, создавать векторные
эмбеддинги и выполнять поиск по содержимому статей с использованием LLM.

Проект включает в себя полный цикл обработки данных — от загрузки и индексации
документов до генерации ответов и оценки качества пайплайна. Вся инфраструктура
запускается через Docker Compose, что делает развёртывание простым и
воспроизводимым.

### Ключевые возможности:

- Загрузка и парсинг PDF-статей с ArXiv;
- Гибкая стратегия чанкинга;
- Генерация эмбеддингов и индексация в векторной БД Qdrant;
- Гибридный поиск (dense + sparse retrieval);
- Генерация ответов через LLM с RAG-контекстом;
- Автоматическая генерация тестовых данных и оценка качества ретривера и
  генератора;
- Оркестрация пайплайнов через Airflow;
- Логирование экспериментов в MLflow;
- Мониторинг метрик через Grafana + Prometheus;
- UI для взаимодействия с системой.

## 1. Настройка проекта

Копируем проект и настраиваем окружение

```bash
git clone https://github.com/IvanMakhrov/arxiv-rag-qa.git
cd arxiv-rag-qa

python3 -m venv .venv
source .venv/bin/activate
```

Устанавливаем uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Устанавливаем зависимости

```bash
uv pip install -e ".[dev]"
```

Устанавливаем pre-commit для качества кода

```bash
pre-commit install
```

## 2. Настройка .env файла

Структура файла .env:<br>

- `AIRFLOW_UID` - user ID в airflow контейнере
- `_AIRFLOW_WWW_USER_USERNAME` - логин в airflow
- `_AIRFLOW_WWW_USER_PASSWORD` - пароль в airflow
- `AIRFLOW__CORE__FERNET_KEY` - ключ для кодирования паролей

- `AWS_ACCESS_KEY_ID` - логин в minio
- `AWS_SECRET_ACCESS_KEY` - пароль в minio

- `POSTGRES_USER` - пользователь в postgres
- `POSTGRES_PASSWORD` - пароль в postgres
- `POSTGRES_DB` - имя БД в postgres

- `GF_SECURITY_ADMIN_USER` - пользователь в Grafana
- `GF_SECURITY_ADMIN_PASSWORD` - пароль в Grafana

Для получения FERNET_KEY:<br>

```python
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

## 3. Запускаем в Docker

```bash
docker compose up
```

## 4. Настройка в AirFlow

В AirFlow UI заходим в Admin - Connections:<br>

- Connection_id - rag_service. http_conn_id в HTTPOperator
- Connection Type - HTTP
- Host - http://rag-service:8000. Имя сервиса и порт в docker-compose.yml

## Стэк

- Оркестратор: AirFlow
- Логирование экспериментов: MlFlow
- БД: postgres
- Векторная БД: Qdrant
- S3: MinIO
- Кеширование: Redis
- Контроль качества кода: Ruff
- Hooks: pre-commit
- Гиперпараметры: Hydra
- Очередь задач: Celery
- Мониторинг: Grafana
- Real-time метрики: Prometheus
- Сбор логов: Promtail
- Хранение логов: Loki

## UI:

- MinIO UI: http://localhost:9001
- AirFlow: http://localhost:8080
- MLFlow: http://localhost:5050
- Api: http://localhost:8000
- Qdrant: http://localhost:6333/dashboard
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- RAG UI: http://localhost:80

## Routers:

- POST - /download-papers - Загрузка pdf
- POST - /parse-pdf - Парсинг pdf в json
- POST - /chunking - Чанкинг
- POST - /embeddings - Создание эмбеддингов
- POST - /qdrant-setup - Настройка qdrant и добавление данных
- POST - /generate-test-data - Генерация тестовых данных
- POST - /retriever-eval - Оценка качества ретривера
- POST - /generator-eval - Оценка качества генератора
- POST - /get-response-json - Получение ответа от RAG
- GET - /tasks/{task_id} - Получение статуса задачи
- GET - /tasks - Получение списка задач
- DELETE - /tasks/{task_id} - Удаление задачи
- GET - /tasks/stats - Статистика задач
- GET - / - UI RAG системы
- GET - /task/{task_id} - Получить статус задачи в UI
- GET - /health - Состояние api
- GET - /metrics - Метрики Latency и Throughput
