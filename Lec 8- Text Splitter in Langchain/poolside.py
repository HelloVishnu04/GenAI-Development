import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()

llm = ChatOpenAI(
    model="poolside/laguna-s-2.1",  # Replace with your Poolside model name
    openai_api_key=os.getenv("POOLSIDE_API_KEY"),
    openai_api_base=os.getenv("POOLSIDE_BASE_URL", "https://inference.poolside.ai/v1"),
    temperature=0.2,
)
messages = "how are you"

result = llm.invoke(messages)
print(result.content)