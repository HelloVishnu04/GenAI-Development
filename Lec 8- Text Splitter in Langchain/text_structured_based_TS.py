from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.document_loaders import PyMuPDFLoader


loader = PyMuPDFLoader('Lec 7- Document loader in Langchain\Vishnu Kumar (3).pdf')
text = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500
)

result = splitter.split_documents(text)

print(result)

