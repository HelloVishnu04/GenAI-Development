from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

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

# First run: Generate report
prompt1 = template1.invoke({"topic": "black hole"})
result = model.invoke(prompt1)

# Second run: Summarize report
prompt2 = template2.invoke({"text": result.content})
result1 = model.invoke(prompt2)

print(result1.content)