from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

loader = DirectoryLoader(
    path='Lec 7- Document loader in Langchain/resumes',
    glob ='*.pdf',
    loader_cls = PyMuPDFLoader
)

docs = loader.lazy_load() # ----- > Use for the large number of documents 

for document in docs:
    print(document.metadata)

