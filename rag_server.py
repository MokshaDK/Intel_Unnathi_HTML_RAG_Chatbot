from optimum.intel.openvino import OVModelForCausalLM
import os
import time
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from fastapi import FastAPI
from transformers import AutoTokenizer, pipeline

#loading information from .env
load_dotenv(verbose=True)
cache_dir         = os.environ['CACHE_DIR']
model_vendor      = os.environ['MODEL_VENDOR']
model_name        = os.environ['MODEL_NAME']
model_precision   = os.environ['MODEL_PRECISION']
inference_device  = os.environ['INFERENCE_DEVICE']
document_dir      = os.environ['DOCUMENT_DIR']
vectorstore_dir   = os.environ['VECTOR_DB_DIR']
vector_db_postfix = os.environ['VECTOR_DB_POSTFIX']
num_max_tokens    = int(os.environ['NUM_MAX_TOKENS'])
embeddings_model  = os.environ['MODEL_EMBEDDINGS']
rag_chain_type    = os.environ['RAG_CHAIN_TYPE']
ov_config         = {"PERFORMANCE_HINT":"LATENCY", "NUM_STREAMS":"1", "CACHE_DIR":cache_dir}

from transformers import AutoModel
model = AutoModel.from_pretrained(embeddings_model, trust_remote_code=True) 

embeddings = HuggingFaceEmbeddings(
    model_name = embeddings_model,
    model_kwargs = {'device':'cpu'},
    encode_kwargs = {'normalize_embeddings':True}
)

#retrieving response from vectorstore based on context
vectorstore_dir = f'{vectorstore_dir}_{vector_db_postfix}'
vectorstore = Chroma(persist_directory=vectorstore_dir, embedding_function=embeddings)
retriever = vectorstore.as_retriever()

print(f'Vector store : {vectorstore_dir}')

model_id = f'{model_vendor}/{model_name}'
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
ov_model_path = f'./{model_name}/{model_precision}'
model = OVModelForCausalLM.from_pretrained(model_id=ov_model_path, device=inference_device, ov_config=ov_config, cache_dir=cache_dir)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=num_max_tokens)
llm = HuggingFacePipeline(pipeline=pipe)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type=rag_chain_type, retriever=retriever)

#generating answer from llm inference
def run_generation(text_user_en):
    ans = qa_chain.run(text_user_en)
    return ans

app = FastAPI()

@app.get('/chatbot/{item_id}')
async def root(item_id:int, query:str=None):
    if query:
        stime = time.time()
        ans = run_generation(query)
        etime = time.time()
        wc = len(ans.split())
        process_time = etime - stime
        words_per_sec = wc / process_time
        return {'response':f'{ans} \r\n\r\nProcessing Time: {process_time:6.1f} seconds'}
    return {'response':''}
