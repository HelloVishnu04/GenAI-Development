from langchain_community.document_loaders import TextLoader, WebBaseLoader
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()
url = 'https://github.com/HelloVishnu04'
loader = WebBaseLoader(url)
docs = loader.load()

model = ChatGoogleGenerativeAI(model = 'gemini-3.5-flash-lite')
prompt = PromptTemplate.from_template("Answer the following question \n {question} from following text - \n {text}")

chain = prompt | model | StrOutputParser()

print(chain.invoke({'question':'What is the key skills this programmer have', 'text':docs[0].page_content}))




