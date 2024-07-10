import os
from langchain_core.documents.base import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import SpacyTextSplitter
from langchain.vectorstores import Chroma
from bs4 import BeautifulSoup
import glob
from dotenv import load_dotenv
import pickle
from tqdm import tqdm

#loading information from .env
load_dotenv(verbose=True)
document_dir = os.environ['DOCUMENT_DIR']
vectorstore_dir = os.environ['VECTOR_DB_DIR']
embeddings_model = os.environ['MODEL_EMBEDDINGS']

from transformers import AutoModel
model = AutoModel.from_pretrained(embeddings_model, trust_remote_code=True) 

def generate_documents_from_html(glob_pattern:str, max_doc_count:int=-1) -> list[Document]:  #creating a docs from HTML pages
    doc_count = 0

    html_files = glob.glob(glob_pattern, recursive=True)

    documents = []
    for html_file in tqdm(html_files):
        with open(html_file, 'rt', encoding='utf-8') as f:
            html_contents = ''.join(f.readlines())

        soup = BeautifulSoup(html_contents, 'html.parser')
        main_section = soup.find('main')
        if main_section is not None:
            text = main_section.get_text()
            text = ''.join([line+'\n' for line in text.splitlines() if line != ''])
            doc = Document(page_content=text, metadata={'source':html_file})
            documents.append(doc)

            doc_count += 1
            if max_doc_count != -1 and doc_count >= max_doc_count:
                break

    return documents

def generate_vectorstore_from_documents( #creating vector db from docs
        docs             :list[Document],
        vectorstore_dir  :str  = './db',
        chunk_size       :int  = 300,           
        chunk_overlap    :int  = 0,
        normalize_emb    :bool = False,
        embeddings_model :str  = 'sentence-transformers/all-mpnet-base-v2',
        pipeline         :str  = 'en_core_web_sm'
    ) -> None:
    print('Splitting docs into chunks')
    print(f'Chunk size : {chunk_size}, Chunk overlap : {chunk_overlap}')
    splitter = SpacyTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        pipeline=pipeline
    )
    splitted_docs = splitter.split_documents(docs)

    print('Generate embedding and storing splitted docs into vector store')
    embeddings = HuggingFaceEmbeddings(
        model_name = embeddings_model,
        model_kwargs = {'device':'cpu'},
        encode_kwargs = {'normalize_embeddings':normalize_emb}
    )

    vectorstore = Chroma(
        persist_directory=f'{vectorstore_dir}_{chunk_size}_{chunk_overlap}', 
        embedding_function=embeddings
    )

    for doc in tqdm(splitted_docs):
        vectorstore.add_documents([doc])

pickle_file = './doc_obj.pickle'
if not os.path.exists(pickle_file):
    print('Reading HTML and generating docs')
    docs = generate_documents_from_html(f'{document_dir}/**/*.html')
    with open(pickle_file, 'wb') as f:
        pickle.dump(docs, file=f)
else:
    print(f'Reading docs from a pickled file ({pickle_file})')
    with open(pickle_file, 'rb') as f:
        docs = pickle.load(file=f)

print('Converting docs into embeddings and creating a vector store')
generate_vectorstore_from_documents(docs, vectorstore_dir,  300,   0, True, embeddings_model)
