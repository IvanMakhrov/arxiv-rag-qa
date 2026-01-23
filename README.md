## Стэк

- MinIO для S3 в Docker
- pre-commit для контроля качества кода
- DVC для контроля данных
- Hydra для гиперпараметров
- AirFlow
- MLFlow

## Запускаем MinIO:

Запускаем MinIO

```bash
docker-compose up -d
```

Файлы pdf хранятся в src/data/raw<br> С помощью dvc push загружаем файлы в S3

MinIO UI:<br> http://localhost:9001

AirFlow:<br> http://localhost:8080

MLFlow:<br> http://localhost:5050

Api:<br> http://localhost:8000

Qdrant:<br> http://localhost:6333<br> http://localhost:6333/dashboard<br>

## Пушим pdf в S3

```bash
export $(grep -v '^#' .env | xargs)
dvc add data/raw
dvc push
```

## Настройка AirFlow для Api

В AirFlow UI заходим в Admin - Connections<br>

- Connection_id - http_conn_id в HTTPOperator
- Connection Type - HTTP
- Host - http://chunking-service:8000. Имя сервиса и порт в docker-compose.yml
