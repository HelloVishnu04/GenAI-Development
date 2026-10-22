from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-3.5-flash-lite')

prompt1 = PromptTemplate.from_template("Generate a topic about {topic}")
prompt2 = PromptTemplate.from_template("Generate a Linkedin Post about {topic}")

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'Linkedin': RunnableSequence(prompt2, model, parser)
})

parallel_chain.invoke({'topic':'AI'})

print(parallel_chain)



