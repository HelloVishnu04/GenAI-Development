from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers.structured import StructuredOutputParser, ResponseSchema

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.7)

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
    
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = " Give 3 facts about the {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions}
)

prompt = template.invoke({'topic':'black hole'})

result = model.invoke(prompt)
final_result = parser.invoke(result)
print(final_result)