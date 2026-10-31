from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_classic.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader('Lec 7- Document loader in Langchain\Vishnu Kumar (3).pdf')
text = loader.load()
splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator = ''
)

result = splitter.split_documents(text)

print(result[2].page_content)
