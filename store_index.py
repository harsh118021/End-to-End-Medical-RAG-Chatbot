from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_pinecone import Pinecone
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Read API key (already in .env)
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Load and process data
extracted_data = load_pdf_file(data="Data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone CLIENT (no api_key argument)
pc = PineconeClient()

index_name = "medicalbot"

# Create index if it does not exist
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Store embeddings in Pinecone using LangChain
docsearch = Pinecone.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

print("✅ Pinecone index created and documents indexed successfully")
