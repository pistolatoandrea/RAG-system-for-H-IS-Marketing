# H-IS Marketing AI Admissions Bot (RAG System)

This project implements an AI-powered virtual assistant for the **H-FARM International School** Admissions Office. The system utilizes a **RAG (Retrieval-Augmented Generation)** architecture to provide accurate responses based on official documents, internal procedures, and website content.

## 🚀 Features
- **Multi-source Knowledge Base**: Extracts information from web URLs, PDF brochures, and text files.
- **Cloud-Native Integration**: Automatic document synchronization via Google Cloud Storage.
- **Clean Output**: Generates plain-text responses (no Markdown) optimized for messaging platforms and webhooks.
- **Startup Efficiency**: The vector database is built once at container startup to maximize response speed and minimize API costs.

## 🛠 Tech Stack
- **Language**: Python 3.11+
- **Web Framework**: FastAPI & Uvicorn
- **AI Framework**: LangChain
- **LLM**: OpenAI GPT-4o
- **Vector Store**: DocArray (In-memory)
- **Infrastructure**: Google Cloud Run, Google Cloud Storage, Secret Manager

## 🏗 System Architecture

### 1. Data Ingestion
Upon startup, the service performs the following operations:
1. Connects to the Google Cloud Storage bucket `h-is-marketing-knowledge-base`.
2. Downloads the PDF brochure and procedure files locally.
3. Scrapes content from the official H-FARM School websites.

### 2. Processing & RAG
Documents are split into chunks using `RecursiveCharacterTextSplitter` and converted into embeddings via OpenAI. When a user asks a question:
- The system retrieves the most relevant chunks (Top-K: 8).
- Context and question are sent to GPT-4o with a custom-tailored prompt.
- The response is stripped of Markdown characters for a professional, plain-text output.

## ⚙️ Infrastructure Configuration

### Environment Variables (Secret Manager)
The service requires the following variable configured in Google Cloud Secret Manager:
- `OPENAI_API_KEY`: Valid API key for OpenAI models.

### Cloud Run Resources
For optimal performance, the container should be configured with:
- **Memory**: 2 GiB (minimum recommended for PDF parsing).
- **CPU**: 1 vCPU.
- **CPU Allocation**: "CPU is only allocated during request processing" for cost optimization.

### IAM Permissions (Service Account)
The Cloud Run service account must hold the following roles:
1. `Storage Object Viewer`: To download documents from the Bucket.
2. `Secret Manager Secret Accessor`: To read the OpenAI API Key.
3. `Cloud Run Invoker`: To allow external invocations (e.g., from Make.com).

## 🔌 API Endpoints

### `POST /ask`
Main query endpoint.
**JSON Payload**:
```json
{
  "question": "What are the deadlines for DP enrollment?",
  "program": "DP"
}
```
**Response**
```
{
  "answer": "The deadlines for the Diploma Programme enrollment are..."
}
```
## 📂 Knowledge Base Maintenance
To update the bot's information without changing the code:

1. Upload the new version of brochure.pdf or procedure.txt to the Google Cloud Storage bucket.

2. Restart the Cloud Run service or deploy a new revision.

3. The bot will automatically download and index the new files upon restart.

*Developed for H-FARM International School - MAC Department.*