from flask import Flask, render_template, request, jsonify 
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import torch


DB_FAISS_PATH = 'vectorstore/db_faiss'




demo_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful and Caring answer:
"""




custom_prompt_template = """

<s>[INST] <<SYS>>
Act as a AI expert. Use the following information to answer the question at the end.
<</SYS>>

{context}

{question} [/INST]

"""




def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(

        model="llama-2-13b-chat.ggmlv3.q2_K.bin",
        #model="llama-2-13b.ggmlv3.q5_0.bin",
        model_type="llama",
        stream=True,

        config={'max_new_tokens': 600,
        'temperature': 0.01,
        'context_length': 1024}


        #max_new_tokens = 1024,
        #temperature = 0.5,
        #context_length=600
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response



app = Flask(__name__) 





@app.route("/", methods=['POST', 'GET'])  
def query_view(): 
    if request.method == 'POST':

        prompt = request.form['prompt'] 

        out=final_result(prompt)

        response=out['result']

        return jsonify({'response': response}) 
    return render_template('index.html') 


if __name__ == "__main__":
    app.run(debug=True) 
