import pandas as pd
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()


faqs_path = Path(__file__).parent/ "resources/faq_data.csv"
chroma_client = chromadb.Client()
collection_name_faq = 'faqs'
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)


def ingest_faq_data(path):
    if collection_name_faq not in [c.name for c in chroma_client.list_collections()]:
        print("Ingesting FAQ data into ChromaDB")
        collection = chroma_client.get_or_create_collection(
            name=collection_name_faq,
            embedding_function=ef
        )

        df = pd.read_csv(path)
        docs = df['question'].to_list()
        metadata = [{'answer': ans} for ans in df['answer'].to_list()]
        ids = [f"id_{i}" for i in range(len(docs))]
        
        collection.add(
            documents=docs,
            metadatas=metadata,
            ids = ids 
        )
        print(f"FAQ Data is successfully ingested into Chroma Collection {collection_name_faq}")
    else:
        print(f"Collection {collection_name_faq} already exists!")


def get_relevant_qa(query):
    collection = chroma_client.get_collection(name=collection_name_faq)
    result = collection.query(
        query_texts=[query],
        n_results=2
    )
    return result

def faq_chain(query):
    result = get_relevant_qa(query)

    context = ''.join([r.get('answer') for r in result['metadatas'][0]])

    prompt = f''' Question and Context are provided below. Generate the answer based on the context provided only. 
    In case you don't find the answer in the context provided, then reply "I don't know". 
    Don't hallucinate and make things up. 
    QUESTION: {query}
    CONTEXT: {context}
    '''
    client = Groq()
    completion = client.chat.completions.create(
        model=os.environ['GROQ_MODEL'],
        messages=[
        {
            "role": "user",
            "content": prompt
        }],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None
    )
    answer = ""
    for chunk in completion:
        answer += chunk.choices[0].delta.content or ""
    return answer

if __name__ == "__main__":
    ingest_faq_data(faqs_path)
    query = "what if I use cash for paying?"
    answer = faq_chain(query)
    #print(answer)