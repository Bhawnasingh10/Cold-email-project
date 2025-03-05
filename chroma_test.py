
import chromadb
client = chromadb.Client()
collection = client.create_collection(name="my_collection")
collection.add(
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ],
    ids=["id1", "id2"]
)
all_docs=collection.get()
print(all_docs)
documents=collection.get(ids=["id1"])
print(documents)

results=collection.query(
    query_texts=['Query is about Nagpur'],
    n_results=2
)
print(results)