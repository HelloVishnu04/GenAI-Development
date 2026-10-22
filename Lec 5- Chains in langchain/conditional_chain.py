from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model1 = ChatGoogleGenerativeAI(model = 'gemini-3.5-flash-lite')

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)




prompt1 = PromptTemplate(
    template = 'Classify the sentiment of the following feedback text into positive or negative. \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions}
)



classifier_chain = prompt1 | model1 | parser2
prompt2 = PromptTemplate(
    template = "Write an appopriate response to this positive feedback \n {feedback}",
    input_variables = ['feedback']
)
prompt3 = PromptTemplate(
    template = "Write an appopriate response to this Negative feedback \n {feedback}",
    input_variables = ['feedback']
)

branch_chain = RunnableBranch(
    # (condition1, chain),
    # (condition2, chain),
    # default chain
    (lambda x: x.sentiment == 'positive', prompt2 | model1 | parser), # -------> chain 1
    (lambda x: x.sentiment == 'negative', prompt3 | model1 | parser),  # -------> chain 2
    RunnableLambda(lambda x:"Could not find sentiment")  # ----> Default chain ( We have to convert it into a Runnable to work)
)  

final_chain = classifier_chain | branch_chain

print(final_chain.invoke({'feedback':'this is a beautiful phone'}))


final_chain.get_graph().print_ascii()



