## k8s (Minikube) запуск

### 1) Перейти в директорию проекта

```bash
cd /path/to/arxiv-rag-qa
```

### 2) Запустить Minikube

```bash
minikube start
```

### 3) Собрать образы (локально)

```bash
docker build -t rag-service:latest -f Dockerfile .
docker build -t celery-worker:latest -f Dockerfile.celery .
docker build -t nginx-rag:latest -f Dockerfile.nginx .
docker build -t mlflow-custom:latest -f Dockerfile.mlflow .
```

### 4) Загрузить образы в Minikube

```bash
minikube image load rag-service:latest
minikube image load celery-worker:latest
minikube image load nginx-rag:latest
minikube image load mlflow-custom:latest
```

### 5) Создать namespace

```bash
kubectl apply -f k8s/namespace/namespace.yaml
```

### 6) Создать PersistentVolumeClaims

```bash
kubectl apply -f k8s/storage/rag-data-pvc.yaml
kubectl apply -f k8s/storage/redis-data-pvc.yaml
```

### 7) Создать секреты из `.env`

`app-secrets` создаём напрямую из `.env`:

```bash
kubectl apply -f k8s/secrets/app-secrets.yaml
```

Docker Registry secret (если тянете образы из private registry):

```bash
kubectl create secret docker-registry docker-registry-secret \
  --docker-server=ghcr.io \
  --docker-username=YOUR_USERNAME \
  --docker-password=YOUR_TOKEN \
  -n rag-project
```

Basic Auth secret для Ingress:

```bash
htpasswd -c auth airflow
kubectl create secret generic basic-auth-secret --from-file=auth -n rag-project
```

### 8) Создать ConfigMaps

```bash
kubectl apply -f k8s/config/
```

### 9) Применить базы данных

```bash
kubectl apply -f k8s/database/postgresql-headless-service.yaml
kubectl apply -f k8s/database/postgresql-service.yaml
kubectl apply -f k8s/database/postgresql-statefulset.yaml
kubectl apply -f k8s/database/redis-deployment.yaml
kubectl apply -f k8s/database/redis-service.yaml
kubectl apply -f k8s/database/qdrant-statefulset.yaml
kubectl apply -f k8s/database/qdrant-service.yaml
```

### 10) Запустить миграцию (один раз)

```bash
kubectl apply -f k8s/jobs/db-migration-job.yaml
kubectl wait --for=condition=complete job/db-migration -n rag-project --timeout=600s
```

### 11) Применить остальные сервисы

```bash
# Storage
kubectl apply -f k8s/storage/minio-service.yaml
kubectl apply -f k8s/storage/minio-statefulset.yaml

# ML Infrastructure
kubectl apply -f k8s/ml/mlflow-deployment.yaml
kubectl apply -f k8s/ml/mlflow-service.yaml

# Application services
kubectl apply -f k8s/app/

# Monitoring
kubectl apply -f k8s/monitoring/

# Background jobs
kubectl apply -f k8s/jobs/daily-cleanup-cronjob.yaml
```

Ingress:

```bash
kubectl apply -f k8s/network/ingress.yaml
minikube addons enable ingress
```

### 12) Проверить статус

```bash
kubectl get pods -n rag-project -w
kubectl get svc -n rag-project
```

### 13) Доступ к сервисам

Запустить проброс портов:

```bash
./k8s.sh start
```

Дальше открывать по url:

- RAG UI: http://localhost:8000
- Airflow: http://localhost:8081
- MLflow: http://localhost:5000
- Qdrant: http://localhost:6333
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- MinIO: http://localhost:9001
