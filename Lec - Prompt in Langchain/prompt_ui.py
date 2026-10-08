from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt
import streamlit as st

load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite', temperature=0.7)
chain = model | StrOutputParser

st.header('Reasearch Tool')

paper_input = st.selectbox("Select Research Paper Name", ["Attention is all you need", "BART: Pre-training of deep Bidirectional transformers ", "GPT-3: Language Models are Few-Shot Learners ", "Diffusion Models Beat GANs on Image Synthesis"])
style_input = st.selectbox("Select Explaination Style", ['Biginner-Friendly', "Technical", "Code- Oriented", "Methematical"])
length_input = st.selectbox("Select Explaination length", ["Short (1-2 paragraphs)", "Medium (3 - 5 paragraphs)", "Long (Detailed Explaiantion)"])


template = load_prompt('.\template.json')

prompt = template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
})
if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)
