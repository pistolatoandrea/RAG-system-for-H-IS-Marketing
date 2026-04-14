# 1. Usiamo Python 3.11 slim per un ottimo bilanciamento tra peso e performance
FROM python:3.11-slim

# 2. Installiamo le dipendenze di sistema
# Aggiunto 'build-essential' per permettere la compilazione di numba/numpy
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic-dev \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Impostiamo la cartella di lavoro
WORKDIR /app

# 4. IL FIX CRUCIALE: Installiamo numpy prima di ogni altra cosa
# Questo risolve l'errore "ModuleNotFoundError: No module named 'numpy'" durante il build
RUN pip install --no-cache-dir numpy==1.24.3

# 5. Copiamo e installiamo il resto delle dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copiamo il resto del codice
COPY . .

# 7. Esposizione porta e configurazione porta dinamica per Cloud Run
ENV PORT 8080
EXPOSE 8080

# 8. Comando di avvio
# Usiamo la forma "python main.py" perché nel tuo file main.py abbiamo già configurato 
# uvicorn internamente con la gestione corretta delle porte.
CMD ["python", "main.py"]