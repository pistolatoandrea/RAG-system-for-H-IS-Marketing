import os
import json
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv
from google.cloud import storage

# --- SETUP AMBIENTE ---
load_dotenv()
os.environ["USER_AGENT"] = "AdmissionsBot/1.0"

# Import LangChain e utility
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader, UnstructuredPDFLoader, TextLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.run_nables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

# --- CONFIGURAZIONE ---
BUCKET_NAME = "h-is-marketing-knowledge-base"
MY_KNOWLEDGE_BASE_URLS = [
    "https://schools.h-farm.com/venice/", 
    "https://schoolnews.h-farm.com"
]
PDF_FILE_PATH = "brochure.pdf"
EXTRA_DATA_PATH = "procedure.txt"

# Variabile globale per il retriever
retriever = None

# --- FUNZIONI DI SUPPORTO ---

def download_knowledge_base():
    """Scarica i file dal Google Cloud Storage Bucket"""
    print("--- 1. Download file dal Bucket... ---")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        files = [PDF_FILE_PATH, EXTRA_DATA_PATH]

        for file_name in files:
            blob = bucket.blob(file_name)
            blob.download_to_filename(file_name)
            print(f"✅ Scaricato: {file_name}")
    except Exception as e:
        print(f"❌ Errore durante il download dal Bucket: {e}")

def initialize_vector_store():
    """Carica i documenti e crea il Vector Store (Eseguito solo all'avvio)"""
    global retriever
    print("--- 2. Inizializzazione Vector Store... ---")
    all_docs = []

    # A. Web Scraping
    try:
        web_loader = WebBaseLoader(MY_KNOWLEDGE_BASE_URLS)
        all_docs.extend(web_loader.load())
        print(f"✅ Web scraping completato ({len(MY_KNOWLEDGE_BASE_URLS)} URL)")
    except Exception as e:
        print(f"⚠️ Web Scraping Error: {e}")

    # B. PDF (Usiamo strategy='fast' per risparmiare RAM su Cloud Run)
    if os.path.exists(PDF_FILE_PATH):
        try:
            pdf_loader = UnstructuredPDFLoader(PDF_FILE_PATH, strategy="fast")
            all_docs.extend(pdf_loader.load())
            print("✅ PDF caricato correttamente")
        except Exception as e:
            print(f"⚠️ PDF Loader Error: {e}")

    # C. Dati Extra (TXT)
    if os.path.exists(EXTRA_DATA_PATH):
        try:
            txt_loader = TextLoader(EXTRA_DATA_PATH)
            all_docs.extend(txt_loader.load())
            print("✅ File procedure.txt caricato")
        except Exception as e:
            print(f"⚠️ Text Loader Error: {e}")

    if not all_docs:
        print("❌ ERRORE: Nessun documento caricato. Il bot non avrà base di conoscenza.")
        return

    # Splitting & Vector Store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    splits = text_splitter.split_documents(all_docs)
    
    vectorstore = DocArrayInMemorySearch.from_documents(
        splits, 
        embedding=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    print("🚀 SISTEMA PRONTO: Vector Store creato con successo!")

def clean_text_for_json(text):
    """Rimuove caratteri speciali e Markdown per output pulito"""
    if not text: return ""
    clean = "".join(char for char in text if ord(char) >= 32 or char in "\n\r\t")
    clean = clean.replace("**", "").replace("*", "").replace("#", "")
    return clean.strip()

# --- AVVIO PROCESSI ---
# Questi girano una volta sola quando il container si accende
download_knowledge_base()
initialize_vector_store()

# --- API ENDPOINTS ---

@app.post("/ask")
async def ask_ai(request: Request):
    if not retriever:
        raise HTTPException(status_code=503, detail="Knowledge base non ancora pronta")
    
    try:
        payload = await request.json()
        user_question = payload.get("question")
        selected_program = payload.get("program", "Not specified")
        # Nuova variabile: di default impostiamo False se non passata
        is_boarding = payload.get("boarding", False)

        if not user_question:
            raise HTTPException(status_code=400, detail="Missing question")

        # Prepariamo l'informazione sul boarding per il prompt
        boarding_info = "The user IS interested in the boarding school service." if is_boarding else "The user is NOT interested in boarding; they are a day student."

        # Prompt Aggiornato con variabile {boarding_context}
        template = """You are the official Admissions Assistant for H-FARM International School. 
        Use the provided context to respond professionally.

        USER INTERESTS:
        - Academic Program: {program_info} 
        - Boarding Status: {boarding_context}

        STYLE RULES:
        1. Respond in the SAME LANGUAGE as the question.
        2. NO MARKDOWN (no stars, no hashtags).
        3. NO BOLD OR ITALICS.
        4. TONE: Professional and elegant.
        5. LISTS: Use simple dashes "-" or numbers "1.".
        6. Always conclude with: admissions@h-farm.com.
                
        CONTENT GUIDELINES:
        - If Boarding Status is "interested", include relevant details about boarding life or facilities if found in the context.
        - If Boarding Status is "not interested", focus only on school/academic aspects.
        - Do not invent information not present in the context.

        Context:
        {context}
        
        Question: {question}

        Response:"""

        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(model_name="gpt-4o", temperature=0)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Chain aggiornata con il nuovo campo
        chain = (
            {
                "context": retriever | format_docs, 
                "question": RunnablePassthrough(), 
                "program_info": lambda x: selected_program,
                "boarding_context": lambda x: boarding_info # Passiamo la stringa creata sopra
            }
            | prompt
            | model
            | StrOutputParser()
        )

        raw_answer = chain.invoke(user_question)
        final_answer = clean_text_for_json(raw_answer)

        return {"answer": final_answer}

    except Exception as e:
        print(f"SYSTEM ERROR: {str(e)}")
        return {"answer": "I encountered a technical error. Please contact admissions@h-farm.com."}
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)