import qdrant_client
import PyPDF2
from os import listdir
from os.path import isfile, join,isdir
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
import sys
from langchain_text_splitters import TokenTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def get_files(dir):
    file_list = []
    for f in listdir(dir):
        if isfile(join(dir,f)):
            file_list.append(join(dir,f))
        elif isdir(join(dir,f)):
            file_list= file_list + get_files(join(dir,f))
    return file_list

def main_indexing(mypath):
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
    client.create_collection(collection_name,vectors_config=VectorParams(size=768, distance=Distance.COSINE))
    qdrant = Qdrant(client, collection_name, hf)
    print("Indexing...")
    onlyfiles = get_files("TestFolder")
    file_content = ""
    for file in onlyfiles:
        print("indexing "+file)
        reader = PyPDF2.PdfReader(file)
        for i in range(0,len(reader.pages)):
            file_content = file_content + " "+reader.pages[i].extract_text()
        text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_text(file_content)
        metadata = []
        for i in range(0,len(texts)):
            metadata.append({"path":file})
        embeddings = hf.embed_documents(texts)
        qdrant.add_texts(texts,metadatas=metadata)
        len(texts)
    print(onlyfiles)
    print("Finished indexing!")

if __name__ == "__main__":
    arguments = sys.argv
    if len(arguments)>1:
        main_indexing(arguments[1])
    else:
        print("You need to provide a path to folder with documents to index as command line argument")