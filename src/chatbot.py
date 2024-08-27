from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

# Path to the FAISS vector store
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Provide the custom prompt template for QA retrieval(A example is given below)
CUSTOM_PROMPT_TEMPLATE = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


# Creates a prompt template for QA retrieval.
def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=['context', 'question'])


# Creates a RetrievalQA chain using LLM, prompt, and vector store.
def retrieval_qa_chain(llm, prompt, db):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )


# Loads the language model.
def load_llm():
    return CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGUF", # Provide the model name that you want to use here.
        model_type="llama",
        max_new_tokens=512, # Maximum number of tokens that can be generated by the model.
        temperature=0.5 # Temperature for sampling from the model.
    )


# Initializes QA bot with embeddings, vector store, and QA chain.
def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return retrieval_qa_chain(load_llm(), set_custom_prompt(), db)


    # Retrieves final result for a query using QA bot.
def final_result(query):
    return qa_bot()({'query': query})


# Chainlit integration
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    await msg.update(content="Hi, Welcome to RAG Bot. What is your query?")
    cl.user_session.set("chain", chain)


# Handles incoming messages and provides responses using the QA bot.
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    answer += f"\nSources: {sources}" if sources else "\nNo sources found"
    await cl.Message(content=answer).send()