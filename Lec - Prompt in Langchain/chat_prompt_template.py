from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

chat_template = ChatPromptTemplate([
    SystemMessage(content='You are a help {domain} expert'),
    HumanMessage(content='Explain in simple terms, What is {topic}')
        
])

prompt = chat_template.invoke({'domain':'cricket', 'topic':'Dusra'})
print(prompt)