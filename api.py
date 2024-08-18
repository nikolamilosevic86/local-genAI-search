from fastapi import FastAPI
#import qdrant_client
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_core.messages import HumanMessage
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import environment_var
import os
from openai import OpenAI
#from langchain_community.llms import Ollama
#from langgraph.graph import END, MessageGraph

class Item(BaseModel):
    query: str
    def __init__(self, query: str) -> None:
        super().__init__(query=query)

#model_name = "amberoad/bert-multilingual-passage-reranking-msmarco"
model_name = "sentence-transformers/msmarco-bert-base-dot-v5"
model_kwargs = {'device': 'cuda'} # changed by pdchristian to 'cuda'
#model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

os.environ["HF_TOKEN"] = environment_var.hf_token
use_nvidia_api = False
use_quantized = True
if environment_var.nvidia_key !="":
    client_ai = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=environment_var.nvidia_key
#        base_url="http://localhost:11434", #pdchristian tried to connect to local ollama server
#        api_key="ollama" #pdchristian tried to connect to local ollama server
#        base_url="http://localhost:1234", #pdchristian tried to connect to local LM-Studio server
#        api_key=environment_var.nvidia_key #pdchristian tried to connect to local LM-Studio server
    )
    use_nvidia_api = True
elif use_quantized:
    model_id = "Kameshr/LLAMA-3-Quantized"
#    model_id = "llama3.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
else:
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
#    model_id = "llama3.1"
#    model_id = "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

client = QdrantClient(path="qdrant/")
collection_name = "MyCollection"
qdrant = Qdrant(client, collection_name, hf)



app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/search")
def search(Item:Item):
    query = Item.query
    search_result = qdrant.similarity_search(
        query=query, k=20
    )
    i = 0
    list_res = []
    for res in search_result:
        list_res.append({"id":i,"path":res.metadata.get("path"),"content":res.page_content})
    return list_res

@app.post("/ask_localai")
async def ask_localai(Item:Item):
    query = Item.query
    search_result = qdrant.similarity_search(
        query=query, k=20
    )
    i = 0
    list_res = []
    context = ""
    mappings = {}
    i = 0
    for res in search_result:
        context = context + str(i)+"\n"+res.page_content+"\n\n"
        mappings[i] = res.metadata.get("path")
        list_res.append({"id":i,"path":res.metadata.get("path"),"content":res.page_content})
        i = i +1

    rolemsg = {"role": "system",
               "content": "Answer user's question using documents given in the context. Formulate all answers in German. In the context are documents that should contain an answer. Please always reference document id (in squere brackets, for example [0],[1]) of the document that was used to make a claim. Use as many citations and documents as it is necessary to answer question."}
    #           "content": "Answer user's question using documents given in the context. In the context are documents that should contain an answer. Please always reference document id (in squere brackets, for example [0],[1]) of the document that was used to make a claim. Use as many citations and documents as it is necessary to answer question."}
    messages = [
        rolemsg,
        {"role": "user", "content": "Documents:\n"+context+"\n\nQuestion: "+query},
    ]
    if use_nvidia_api:
        completion = client_ai.chat.completions.create(
            model="meta/llama3-70b-instruct",
            messages=messages,
            temperature=0.5,
            top_p=1,
            max_tokens=1024,
            stream=False
        )
        response = completion.choices[0].message.content
    else:
        input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)


        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
        )
        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:])
    return {"context":list_res,"answer":response}
