import chromadb
from sentence_transformers import SentenceTransformer
import json
import numpy as np

db_client = chromadb.PersistentClient('./resources')
resources_collection = db_client.get_collection('resources')

with open('data.json') as file:
    documents = json.load(file)

model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')

metadatas = []
ids = []
texts = []

for doc in documents:
    metadata = {'title': doc['title'], 'category': doc['category']}
    metadatas.append(metadata)
    ids.append(doc['id'])
    texts.append(doc['text'])

embeddings = model.encode(texts)

resources_collection.upsert(
    ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings
)
