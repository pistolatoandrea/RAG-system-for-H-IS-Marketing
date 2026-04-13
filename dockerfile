# Usa un'immagine Python ufficiale leggera
FROM python:3.10-slim

# Installiamo le dipendenze di sistema necessarie per processare i PDF e le immagini
# libmagic-dev è fondamentale per identificare i tipi di file
# poppler-utils e tesseract-ocr servono a Unstructured per leggere i PDF
RUN apt-get update && apt-get install -y \
    libmagic-dev \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Impostiamo la cartella di lavoro dentro il container
WORKDIR /app

# Copiamo il file delle dipendenze
COPY requirements.txt .

# Installiamo le librerie Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiamo tutto il resto del codice
# Nota: se i PDF non sono nella repo, dovrai caricarli via Cloud Storage (vedremo dopo)
COPY . .

# Cloud Run usa la porta 8080 di default
EXPOSE 8080

# Comando per avviare l'app usando la variabile d'ambiente PORT assegnata da Google
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}