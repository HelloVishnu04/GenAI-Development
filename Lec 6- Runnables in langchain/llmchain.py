from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_classic.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-3.5-flash-lite')

prompt = PromptTemplate(
    template = "Suggest a catchy blog title {topic}",
    input_variables=['topic']
)

chain = model | prompt
topic = "Boxing"
output = chain.invoke(topic)

print(output)

