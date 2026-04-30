# H-IS Marketing AI Admissions Bot (RAG System)

This repository contains a high-performance **RAG (Retrieval-Augmented Generation)** web service designed for the **H-FARM International School** Admissions Office. It serves as an intelligent middle-layer, transforming raw student inquiries into precise, context-aware responses.

## 🚀 Overview
The system bridges the gap between prospective families and the Admissions Office by automating the first level of inquiry. It leverages a custom knowledge base—including official brochures, internal procedures, and web content—to deliver human-like, accurate answers.

## 🏗 Global Automation Workflow
This service is a core component of a wider automation ecosystem:
1. **Intake**: Lead inquiries are captured via **Zoho Forms**.
2. **Orchestration**: **Make.com** triggers the workflow, sanitizing the input and calling this RAG service.
3. **Intelligence**: This service processes the RAG logic via **OpenAI GPT-4o** and returns a sanitized plain-text response.
4. **Closing**: Make.com updates **Zoho CRM** and triggers personalized follow-up emails.

### 🎯 Focus of this Repository
**This repository specifically implements Node #3 (Intelligence/RAG) of the workflow described above.** While the automation logic is managed by Make.com and the data resides in Zoho, this web service acts as the "brain" of the system. It is responsible for receiving the structured data, performing the vector search within the H-FARM knowledge base, and returning a human-like, plain-text answer that is ready for CRM insertion and email delivery.

## 🛠 Tech Stack
- **Framework**: FastAPI (Python 3.11+)
- **Orchestration & AI**: LangChain
- **LLM**: OpenAI GPT-4o
- **Vector Store**: DocArray (In-memory for low-latency startup)
- **Infrastructure**: Google Cloud Run, Cloud Storage, Secret Manager
- **Integration**: Webhooks via Make.com & Zoho Suite

## ⚙️ Key Technical Features
- **Multi-source Ingestion**: Hybrid data loading from GCS buckets (PDFs/TXT) and real-time web scraping.
- **Auto-Indexing**: The vector database is re-initialized at container startup, ensuring the bot always uses the latest version of the uploaded documents.
- **Robust Parsing**: Specifically engineered to handle complex JSON payloads with special characters and line breaks (common in long form inquiries).
- **Clean Response Engine**: Custom output filtering to strip Markdown, ensuring compatibility with CRM text fields and email templates.

## 🔌 API Reference

### `POST /ask`
Processes an inquiry based on the selected academic program.

**JSON Payload**:
```json
{
  "question": "What are the scholarship options for the IB Diploma Programme?",
  "program": "DP",
  "boarding": "false",
  "language": "English"
}
```

### Successful Response
```json
{
  "answer": "H-FARM International School offers merit-based scholarships for the DP program... [Plain Text]"
}
```
## 📂 Knowledge Base Management

Updating the bot's knowledge requires zero code changes:

Upload the updated brochure.pdf or procedures.txt to the h-is-marketing-knowledge-base bucket.

Redeploy or Restart the Cloud Run service.

The system will automatically re-index the new data upon the next startup.

## 🛡 Security & Deployment
**Authentication**: Secured via Cloud Run IAM Invoker roles.

**Secrets**: API Keys are never hardcoded; they are fetched at runtime from Google Secret Manager.

**Resources**: Optimized for 2 GiB RAM / 1 vCPU to balance PDF processing power and cost-efficiency.