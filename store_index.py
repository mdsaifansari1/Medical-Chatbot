import os
from dotenv import load_dotenv
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables from .env file
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY') 

# Setting up environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# 1. Extract and Process PDF Data
extracted_data = load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

# 2. Download Embeddings Model
embeddings = download_hugging_face_embeddings()

# 3. Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"  # Updated index name to match your jupyter trials

# 4. Create Pinecone Index if it doesn't exist
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # Matching HuggingFace MiniLM embeddings dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to the existing/new index
index = pc.Index(index_name)

# 5. Push Chunks to Pinecone Vector Store
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)

print("Data ingestion and indexing completed successfully!")