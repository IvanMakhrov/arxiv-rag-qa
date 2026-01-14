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

## Пушим pdf в S3

```bash
export $(grep -v '^#' .env | xargs)
dvc push
```
