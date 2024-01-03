import streamlit as st
from langchain.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler


st.set_page_config(page_title="PrivateGPT", page_icon="🛩️")


if "messages" not in st.session_state:
    st.session_state["messages"] = []


@st.cache_data(show_spinner="Embedding the file . . .")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/private_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OllamaEmbeddings(model="mistral:latest")
    cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)
def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_template(
    """
    Answer the question using ONLY the following context.
    If you don't know the answer, just say you don't know.
    DON'T make anything up.
    
    Context: {context}
    Question: {question}
    """
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""
    
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
    
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
    
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


st.title("PrivateGPT")
st.markdown("""
    ### Welcome to PrivateGPT!
    PrivateGPT enables users to upload their documents to local storage so that users can upload their documents without any concerns about AI servers being aware of the contents of their documents.
""")
st.divider()
st.subheader("AI Model")
mode = st.selectbox("Choose a model you want", options=["Mistral", "Falcon", "Llama2"])
st.write("###")
st.subheader("File Uploading")
file = st.file_uploader("Upload your document. Available types: .txt .pdf .docx", type=["pdf", "txt", "docx"])
st.divider()

if mode == "Mistral":
    model = "mistral:latest"
elif mode == "Falcon":
    model = "falcon:latest"
elif mode == "Llama2":
    model = "llama2:latest"


llm = ChatOllama(
    model=model,
    temperature=0.7,
    streaming=True,
    callbacks=[ChatCallbackHandler()]
)

if file:
    st.subheader(f"Chatting - {mode}")
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your documents!")
    if message:
        send_message(message, "human")
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        } | prompt | llm
        with st.chat_message("ai"):
            response = chain.invoke(message)
else:
    st.session_state["messages"] = []
    