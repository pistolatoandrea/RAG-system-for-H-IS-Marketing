import os
import json
import re
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv

# Setup Ambiente
os.environ["USER_AGENT"] = "AdmissionsBot/1.0"
load_dotenv()

# Import LangChain e utility
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader, UnstructuredPDFLoader, TextLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

# --- CONFIGURAZIONE ---
MY_KNOWLEDGE_BASE_URLS = [
    "https://schools.h-farm.com/venice/", 
    "https://schoolnews.h-farm.com"
]
PDF_FILE_PATH = "brochure.pdf"
EXTRA_DATA_PATH = "procedure.txt"
# ----------------------

def clean_text_for_json(text):
    """
    Rimuove caratteri di controllo ASCII (0-31) che rompono il JSON, 
    ma mantiene i ritorni a capo standard.
    """
    if not text:
        return ""
    # Rimuove caratteri non stampabili tranne \n, \r, \t
    clean = "".join(char for char in text if ord(char) >= 32 or char in "\n\r\t")
    # Rimuove eventuali asterischi rimasti per errore (doppia sicurezza)
    clean = clean.replace("**", "").replace("*", "")
    return clean.strip()

@app.post("/ask")
async def ask_ai(request: Request):
    try:
        # 1. Ricezione e validazione Payload
        payload = await request.json()
        user_question = payload.get("question")
        selected_program = payload.get("program", "Not specified") # EYU, PYP, MYP, DP

        if not user_question:
            raise HTTPException(status_code=400, detail="Missing question")

        print(f"\n--- Processing: {user_question} ---")
        print(f"--- Program Interest: {selected_program} ---")

        # 2. Caricamento Documenti (Knowledge Base)
        all_docs = []

        # A. Web Scraping
        try:
            web_loader = WebBaseLoader(MY_KNOWLEDGE_BASE_URLS)
            all_docs.extend(web_loader.load())
        except Exception as e:
            print(f"Web Scraping Error: {e}")

        # B. PDF Intelligente (Unstructured)
        if os.path.exists(PDF_FILE_PATH):
            try:
                pdf_loader = UnstructuredPDFLoader(PDF_FILE_PATH, strategy="hi_res")
                all_docs.extend(pdf_loader.load())
            except Exception as e:
                print(f"PDF Loader Error: {e}")

        # C. Dati Extra (TXT)
        if os.path.exists(EXTRA_DATA_PATH):
            txt_loader = TextLoader(EXTRA_DATA_PATH)
            all_docs.extend(txt_loader.load())

        if not all_docs:
            return {"answer": "Error: Knowledge base is empty."}

        # 3. Splitting & Vector Store
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
        splits = text_splitter.split_documents(all_docs)
        
        vectorstore = DocArrayInMemorySearch.from_documents(
            splits, 
            embedding=OpenAIEmbeddings()
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

        # 4. Prompt Multilingua e Context-Aware
        template = """You are the official Admissions Assistant for H-FARM International School. 
        Use the provided context to respond professionally.

        USER INTERESTS:
        The user is specifically interested in: {program_info} 
        (Note: EYU=Early Years, PYP=Primary Years, MYP=Middle Years, DP=Diploma Programme).

        STYLE RULES:
        1. LANGUAGE: Respond in the SAME LANGUAGE as the user's question (e.g., if Italian, reply in Italian).
        2. NO MARKDOWN: Do not use asterisks (**), hashtags (#), or any other markdown symbols.
        3. NO BOLD OR ITALICS: Provide plain text only.
        4. TONE: Professional, cordial, and elegant.
        5. STRUCTURE: Clear paragraphs with a blank line between them.
        6. LISTS: Use simple dashes "-" or numbers "1.".
        7. CLOSING: Always conclude by inviting them to contact admissions@h-farm.com for more details.
                
        CONTENT GUIDELINES:
        - Prioritize information related to the selected program ({program_info}).
        - Describe admission tests, meetings, or documents clearly if mentioned in the context.
        - If the information is not in the context, do NOT invent it. Instead, refer them to the Admissions Office.

        Context:
        {context}
        
        Question: {question}

        Detailed Response:"""

        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(model_name="gpt-4o", temperature=0)

        # 5. RAG Chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Creiamo un dizionario di input per la chain
        chain_input = {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "program_info": lambda x: selected_program
        }

        rag_chain = chain_input | prompt | model | StrOutputParser()

        # 6. Generazione e Pulizia Finale
        raw_answer = rag_chain.invoke(user_question)
        final_answer = clean_text_for_json(raw_answer)

        print(f"RESPONSE GENERATED ({len(final_answer)} chars)")
        return {"answer": final_answer}

    except Exception as e:
        print(f"SYSTEM ERROR: {str(e)}")
        # Restituiamo comunque un JSON valido anche in caso di errore
        return {"answer": "I am sorry, but I encountered a technical error. Please contact admissions@h-farm.com directly."}

if __name__ == "__main__":
    # Avvio server sulla porta 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)