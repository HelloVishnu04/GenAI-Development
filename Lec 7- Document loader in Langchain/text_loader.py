from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

loader = TextLoader('Lec 7- Document loader in Langchain/docs.txt', encoding='utf-8')
Docx = loader.load()

model = ChatGoogleGenerativeAI(model = 'gemini-3.5-flash-lite', temprature= 0.7)

prompt = PromptTemplate.from_template("Write a summary for the following poem - \n {poem}")

parser = StrOutputParser()

chain = prompt | model | parser

output = chain.invoke({'poem':Docx[0].page_content})

print(output)
