from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate 5 pointer summary for the following text \n {text}",
    input_variables=['text']
)

model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser 

result = chain.invoke({'topic':'unemployement in india '})

print(result)

chain.get_graph().print_ascii()