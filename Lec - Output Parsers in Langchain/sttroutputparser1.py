from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Use a valid model name with hyphens, e.g., 'gemini-2.5-flash' or 'gemini-1.5-flash'
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.7)

# Prompt 1
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"],
)

# Prompt 2: corrected input_variables to match '{text}'
template2 = PromptTemplate(
    template="Write a 5 line summary of the following text:\n\n{text}",
    input_variables=["text"],
)

# create a parser
parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)
