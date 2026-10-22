from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Actually execute load_dotenv() to load GOOGLE_API_KEY from your .env file
load_dotenv()

# 2. Load the document (specify utf-8 encoding to avoid OS decoding errors)
loader = TextLoader("Lec 6- Runnables in langchain/docs.txt", encoding="utf-8")
documents = loader.load()

# 3. Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 4. Use GoogleGenerativeAIEmbeddings instead of the retired GooglePalmEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever()

# 5. Use .invoke() instead of the deprecated .get_relevant_documents()
query = "What are the key takeaways from the document?"
retrieved_docs = retriever.invoke(query)

retrieved_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

# 6. Use a valid Gemini model identifier
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.7)

prompt = f"Based on the following text, answer the question: {query}\n\n{retrieved_text}"
answer = model.invoke(prompt)
print(answer.content)