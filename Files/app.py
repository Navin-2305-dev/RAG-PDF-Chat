from flask import Flask, request, render_template, jsonify
import weaviate
from weaviate.auth import AuthApiKey
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Weaviate
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api = os.getenv("WEAVIATE_API_KEY")

client = weaviate.Client(
    url=weaviate_url,
    auth_client_secret=AuthApiKey(api_key=weaviate_api),
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

llm = ChatOllama(model="mistral", temperature=0)
template = """You are a helpful assistant. Given the following context from uploaded documents, answer the question in a concise manner.
Context: {context}
Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm | StrOutputParser()

vector_db = None
class_name = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_vector_store():
    global vector_db, class_name
    try:
        schema_classes = client.schema.get().get('classes', [])
        for class_obj in schema_classes:
            if class_obj['class'].startswith('LangChain_'):
                print(f"Deleting old class {class_obj['class']} to avoid conflicts.")
                client.schema.delete_class(class_obj['class'])

        documents = []
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if allowed_file(filename):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())

        if not documents:
            print("No PDF files found in upload folder.")
            return False

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=800)
        text = text_splitter.split_documents(documents)

        vector_db = Weaviate.from_documents(
            text, embeddings, client=client, by_text=False
        )

        time.sleep(2)

        schema = client.schema.get()
        for class_obj in schema['classes']:
            if class_obj['class'].startswith('LangChain_'):
                class_name = class_obj['class']
                break
        if class_name is None:
            raise ValueError("No LangChain_ class found in Weaviate schema after indexing")

        vector_db = Weaviate(client=client, index_name=class_name, text_key="text", embedding=embeddings, by_text=False)
        return True
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        return False

initialize_vector_store()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    global vector_db, class_name
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400

        files = request.files.getlist('files')
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        if not initialize_vector_store():
            return jsonify({'error': 'Failed to process uploaded files'}), 500

        return jsonify({'message': 'Files uploaded and processed successfully'})
    
    except Exception as e:
        return jsonify({'error': f'Error uploading files: {str(e)}'}), 500

@app.route('/query', methods=['POST'])
def query():
    try:
        user_query = request.form['query']
        if not user_query:
            return jsonify({'error': 'Please enter a query'}), 400
        
        if vector_db is None:
            return jsonify({'error': 'No documents have been processed yet'}), 400

        results = vector_db.similarity_search(user_query, k=3)
        context = "\n".join([doc.page_content for doc in results])
        
        response = chain.invoke({"context": context, "question": user_query})
        return jsonify({'response': response, 'query': user_query})
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)