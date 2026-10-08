import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load variables from .env into os.environ
load_dotenv()

# Initialize the model (auto-picks GOOGLE_API_KEY from environment)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.7,
)

# Test invocation
response = llm.invoke("What are three key principles of clean code?")
print(response.content)