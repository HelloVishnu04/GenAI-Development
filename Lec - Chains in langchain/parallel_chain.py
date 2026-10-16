from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")

model2 = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")

prompt1 = PromptTemplate(
    template= "Generate sort and simple notes from the following text \n {text}",
    input_variables=['text'],
)

prompt2 = PromptTemplate(
    template= "Generate 5 sort question answer from the following text \n {text}",
    input_variables=['text'],
)

prompt3 = PromptTemplate(
    template = "merge the provided notes and quiz into a single document \n {notes} and {quiz}",
    input_variables=['notes', 'quiz']
)


parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})

merge_chian = prompt3 | model1 | parser



chain = parallel_chain | merge_chian 
text = """"""

result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()