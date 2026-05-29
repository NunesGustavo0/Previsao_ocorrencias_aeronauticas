FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . .

# Os metadados locais do MLflow foram gerados com este caminho absoluto.
# O link mantem o registry funcional dentro do container sem duplicar arquivos.
RUN mkdir -p /home/gustavo/codigos \
    && ln -s /app /home/gustavo/codigos/ocorrencias_aeronauticas

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
