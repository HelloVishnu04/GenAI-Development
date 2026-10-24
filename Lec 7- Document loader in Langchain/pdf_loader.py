from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser


loader = PyMuPDFLoader('Lec 7- Document loader in Langchain\Vishnu Kumar (3).pdf')
pdf = loader.load()

print(len(pdf))
print(pdf[0].page_content)


