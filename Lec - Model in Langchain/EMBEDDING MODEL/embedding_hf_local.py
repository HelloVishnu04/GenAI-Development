from langchain_huggingface import HuggingFaceEndpointEmbeddings

embedding = HuggingFaceEndpointEmbeddings(model_name = '--')

text = "Delhi is the capital on India"

vector = embedding.embed_query(text)

print(str(vector))