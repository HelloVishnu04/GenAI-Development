from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-3.5-flash-lite')

prompt1 = PromptTemplate.from_template("Generate a joke about {topic}")
prompt2 = PromptTemplate.from_template("Explain the following joke - {text}")

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'Explain': RunnableSequence(prompt2, model, parser)
    }
)
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

final_chain.invoke({'topic':'AI'})
print(final_chain)



