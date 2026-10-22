from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Use a valid model name with hyphens, e.g., 'gemini-2.5-flash' or 'gemini-1.5-flash'
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.7)


# parser
parser = JsonOutputParser()
# Prompt 1
template1 = PromptTemplate(
    template="Give me the name, age and city of a fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()} 
)

chain = template1 | model | parser
result = chain.invoke({})
print(result)