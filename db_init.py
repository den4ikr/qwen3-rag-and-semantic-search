import chromadb

db_client = chromadb.PersistentClient('./resources')

db_client.create_collection('resources')