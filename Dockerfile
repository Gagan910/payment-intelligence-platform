FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY pyproject.toml ./

COPY src ./src
COPY models ./models
COPY configs ./configs
COPY db ./db

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir .

EXPOSE 8001

CMD ["uvicorn", "payment_platform.api.app:app", "--host", "0.0.0.0", "--port", "8001"]