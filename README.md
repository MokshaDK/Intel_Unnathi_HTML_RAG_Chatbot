# Intel_Unnathi_HTML_RAG_Chatbot
A chatbot capable of answering questions related to AI, ML and DL. It utilizes a RAG model to retrieve information from a vector database. The vector database is created using relevant HTML documents sourced from Wikipedia. 

## Files & Folders
`.env`- Contains environment variables for configuring the application.</br>
`requirements.txt`- Requirements file.</br>
`llm_model_downloader.py`- Downloads the `llama2-7b-chat-hf model` and converts it into OpenVINO IR models.</br>
`html_wikipedia`- Folder containing `Wikipedia` pages related to AI, ML and DL in `HTML` format.</br>
`html_vectordb.py`- Converts the HTML files in `html_wikipedia` to into a vector database.
`huggingface_login.py`- Used to log in to the HuggingFace hub.</br>
`rag_server.py`- Chatbot server.</br>
`rag_client.py`- Chatbot client.</br>
`.streamlit`- Contains theme configurations for the Streamlit UI.</br>

## How to Run
>[!NOTE]
>These installation steps apply to Windows devices.</br>
>Run these commands on Windows Powershell.</br>
>Activate the virtual environment, `venv/Scripts/activate`, each time you wish to run the client and server.</br>

### Setting up the server
>[!NOTE]
>You only need to run the steps in this section once
```
git clone https://github.com/MokshaDK/Intel_Unnathi_HTML_RAG_Chatbot.git
cd Intel_Unnati_HTML_RAG_CHatbot
```
Clone this repository and go to the directory on your device.
```
python -m venv venv
venv/Scripts/activate
python -m pip install -U pip
pip install -U setuptools wheel
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
Install the prerequisites.

You may add/remove files from the 'html_wikipedia' folder based on your requirements, but ensure that the new files added are or HTML format.
```
python html_vectordb.py
```
Stores the documents objects in `doc_obj.pickle` and generates a vector database `.vectorstore_300_0` with chunk size 300 and overlap 0.
```
python llm_model_downloader.py
```
Downloads `llama2-7b-chat-hf`. You may change the model from the `.env` file.
>[!NOTE]
>If you choose to use `llama2-7b-chat-hf`, you will require a token with credentials from the HuggingFace Hub for authentication.

### Running the server
```
uvicorn rag-_server:app --host 0.0.0.0
```
Ensure you are in the correct directory and have activated the virtual environment before running the server.

### Running the client
```
streamlit run rag_client.py
```
Run this in another Windows PowerShell window.
Ensure you are in the correct directory and have activated the virtual environment (in the new window too) before running the client.

## Sample Output
![image](https://github.com/MokshaDK/Intel_Unnathi_HTML_RAG_Chatbot/assets/141493495/cedd4b60-364b-46a7-91d5-924d9007f71d)

## Demo Video
[Demo Video Link](https://drive.google.com/file/d/1HLiFXQnD8EjVPJXKbCodGmn7SZW9sH9d/view?usp=sharing)


