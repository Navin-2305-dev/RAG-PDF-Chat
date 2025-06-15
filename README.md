# 🧠 RAG-PDF-Chat: Retrieval-Augmented Generation for PDFs

This project enables users to **chat with their uploaded PDF documents** using the power of **RAG (Retrieval-Augmented Generation)** combined with **LLMs (Large Language Models)**. Built with **Flask**, **Weaviate**, and **LangChain**, it processes PDFs, stores their vectorized embeddings, and generates contextual answers to user queries.

## 🚀 Features

- Upload and process PDF documents via web UI
- Extract and chunk document text using LangChain
- Generate embeddings with HuggingFace models
- Store embeddings in Weaviate vector database
- Ask natural language queries and receive AI-generated answers via Mistral (Ollama)
- Flask-powered backend with RESTful routes
- Supports multiple uploads and re-indexing

## 📽️ Demo

Check out a video demo of the project in action!  
*(Attach video link or embed if available)*

## 🔧 Tech Stack

- **Backend**: Python + Flask
- **RAG Framework**: LangChain
- **LLM**: Ollama (Mistral model)
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **Vector Store**: Weaviate
- **Document Parsing**: PyPDFLoader
- **Text Splitting**: RecursiveCharacterTextSplitter
- **Package Manager**: `uv`
- **Environment Handling**: python-dotenv

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/Navin-2305-dev/RAG-PDF-Chat.git
cd RAG-PDF-Chat
```

2. **Set up environment variables**
Create a `.env` file in the root directory and add:
```
WEAVIATE_URL=your_weaviate_instance_url
WEAVIATE_API_KEY=your_api_key
```

3. **Install dependencies using uv**
```bash
uv pip install -r requirements.txt
```

4. **Run the app**
```bash
python app.py
```

## 📂 Project Flow

```
Upload PDF -> Chunk & Embed -> Store in Weaviate -> User Query -> Vector Search -> Context Passed to LLM -> Generate Answer
```
