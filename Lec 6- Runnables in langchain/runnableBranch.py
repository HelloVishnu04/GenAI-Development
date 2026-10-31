from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

prompt1 = PromptTemplate.from_template("Write a detailed report on {topic}")
prompt2 = PromptTemplate.from_template("Summarize the following text \n {text}")

model1 = ChatGoogleGenerativeAI(model = 'gemini-3.5-flash-lite')

parser = StrOutputParser()

report_gen_chain = prompt1 | model1 | parser 

branch_chain = RunnableBranch(
    (lambda x:len(x.split())>500, prompt2 | model1 | parser),
    RunnablePassthrough()
)

final_chain = report_gen_chain | branch_chain

result = final_chain.invoke({'topic':'Russia vs Ukraine'})

print(result)
