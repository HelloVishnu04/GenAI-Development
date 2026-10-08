from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize the embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"  # or "models/gemini-embedding-2"
)

# 1. Embed a single search query
query_vector = embeddings.embed_query("Delhi is the capital of india")
print(f"Query embedding length: {len(query_vector)}")

# 2. Embed a list of documents
docs = [
    "LangChain provides integrations for Google Gemini models.",
    "Embeddings convert text into dense semantic vectors.",
    "Vector databases perform cosine similarity searches."
]
doc_vectors = embeddings.embed_documents(docs)
print(f"Indexed {len(doc_vectors)} documents, dim: {len(doc_vectors[0])}")