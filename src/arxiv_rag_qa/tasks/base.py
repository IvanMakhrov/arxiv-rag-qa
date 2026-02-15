import os

from celery import Task
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from arxiv_rag_qa.db.models import Base, TaskStatus

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://airflow:airflow@postgres/airflow")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

Base.metadata.create_all(bind=engine)


class DatabaseTask(Task):
    _session = None

    @property
    def session(self):
        if self._session is None:
            self._session = SessionLocal()
        return self._session

    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        """Вызывается после завершения задачи"""
        if self._session:
            self._session.close()
            self._session = None

    def update_task_status(self, task_id, **kwargs):
        """Обновление статуса задачи в БД"""
        try:
            task = self.session.query(TaskStatus).filter(TaskStatus.id == task_id).first()
            if task:
                for key, value in kwargs.items():
                    setattr(task, key, value)
                self.session.commit()
        except Exception as e:
            print(f"Error updating task status: {e}")
            self.session.rollback()
