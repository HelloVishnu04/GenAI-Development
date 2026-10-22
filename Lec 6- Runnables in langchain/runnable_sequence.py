from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.schema.runnable import RunnableSequence

load_dotenv()

# 1. Define prompts
prompt1 = PromptTemplate.from_template("Write a joke about {topic}")
prompt2 = PromptTemplate.from_template("Explain the following joke - {text}")

# 2. Setup model and string parser
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.7)
parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

# 4. Stream output token-by-token in real time
print("AI Response: ", end="", flush=True)
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)

print()  