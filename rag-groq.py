import os
import tempfile
import gradio as gr

from groq import Groq
from langchain_huggingface import (
    HuggingFaceEndpointEmbeddings
)

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import (
    Chroma
)

# ==========================
# CONFIG
# ==========================
GROQ_MODEL = (
    "llama-3.3-70b-versatile"
)

groq_client = Groq(
    api_key=os.getenv(
        "GROQ_API_KEY"
    )
)

embedding = (
    HuggingFaceEndpointEmbeddings(
        model=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        huggingfacehub_api_token=
        os.getenv(
            "HF_TOKEN"
        )
    )
)

db = None


# ==========================
# PDF PROCESSING
# ==========================
def process_pdf(pdf_file):

    global db

    if pdf_file is None:
        return (
            "Upload PDF dulu."
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        with open(
            pdf_file.name,
            "rb"
        ) as f:

            tmp_file.write(
                f.read()
            )

        temp_path = (
            tmp_file.name
        )

    loader = PyPDFLoader(
        temp_path
    )

    documents = loader.load()

    splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
    )

    docs = (
        splitter
        .split_documents(
            documents
        )
    )

    db = (
        Chroma
        .from_documents(
            docs,
            embedding=
            embedding
        )
    )

    return (
        "✅ Dokumen siap!"
    )


# ==========================
# CHAT
# ==========================
def chat(message, history):

    global db

    if db is None:
        return (
            "Upload PDF dulu."
        )

    retrieved_docs = (
        db.similarity_search(
            message,
            k=3
        )
    )

    context = "\n".join(
        [
            doc.page_content
            for doc
            in retrieved_docs
        ]
    )

    prompt = f"""
Anda adalah AI pencari informasi dokumen.

ATURAN:
1. Jawab HANYA dari context
2. Jangan gunakan pengetahuan pribadi
3. Jika informasi tidak ada, jawab:
"Informasi tidak ditemukan di dokumen."

Context:
{context}

Question:
{message}
"""

    response = (
        groq_client
        .chat
        .completions
        .create(
            model=
            GROQ_MODEL,
            messages=[
                {
                    "role":
                    "user",
                    "content":
                    prompt
                }
            ]
        )
    )

    return (
        response
        .choices[0]
        .message.content
    )


# ==========================
# UI
# ==========================
with gr.Blocks() as demo:

    gr.Markdown(
        "# 🤖 RAG PDF Chatbot"
    )

    gr.Markdown(
        "Upload PDF lalu tanya isi dokumen."
    )

    pdf_file = gr.File(
        label="Upload PDF",
        file_types=[".pdf"]
    )

    upload_btn = gr.Button(
        "Process PDF"
    )

    status = gr.Textbox(
        label="Status"
    )

    chatbot = gr.ChatInterface(
        fn=chat,
        type="messages"
    )

    upload_btn.click(
        fn=process_pdf,
        inputs=pdf_file,
        outputs=status
    )

demo.launch()