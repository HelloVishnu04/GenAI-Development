from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-3.5-flash-lite', temperature = 0.7)

chain = model | StrOutputParser()

messages = [
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about langchain')
]
response = chain.invoke(messages)

messages.append(AIMessage(content=response))
print(messages)