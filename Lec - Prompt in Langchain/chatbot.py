from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite', temperature=0.7)

chat_history = []


chain = model | StrOutputParser()
while True:
    user_input = input('You: ')
    chat_history.append(user_input)
    if user_input == 'exit':
        break
    response = chain.invoke(chat_history)
    chat_history.append(response)
    print("AI : ",response)
print(chat_history)