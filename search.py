import qdrant_client
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient

model_name = "amberoad/bert-multilingual-passage-reranking-msmarco"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
client = QdrantClient(path="qdrant/")
collection_name = "MyCollection"
qdrant = Qdrant(client, collection_name, hf)
search_result = qdrant.similarity_search(
    query="What are limitations of biomedical relationship extraction??",k=10
)

print(search_result)
print(len(search_result))
for res in search_result:
    print(res.metadata.get("path"))
    print(res.page_content)