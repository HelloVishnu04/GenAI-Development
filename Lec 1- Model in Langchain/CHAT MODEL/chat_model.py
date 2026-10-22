import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite', temperature=0.7)
()
chain = model | StrOutputParser

result = chain.invoke("Write 5 lines poem on cricket ")
print(result)