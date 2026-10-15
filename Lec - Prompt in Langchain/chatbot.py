from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite', temperature=0.7)

chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]



chain = model | StrOutputParser()
while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    response = chain.invoke(chat_history)
    chat_history.append(AIMessage(content=response))
    print("AI : ",response)
print(chat_history)