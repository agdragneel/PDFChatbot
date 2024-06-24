import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyMuPDFLoader


import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

def extract_from_sentence(full_text, sentence):
    try:
        # Find the starting index of the sentence in the full text
        start_index = full_text.index(sentence)
        # Extract the substring from the start index to the end
        result = full_text[start_index:]
        return result
    except ValueError:
        # If the sentence is not found, return an empty string or a message
        return "Sentence not found in the text."

def process_pdf(pdf_file):
    loader = PyMuPDFLoader(pdf_file.name)
    document = loader.load()
    embeddings = HuggingFaceEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    text = text_splitter.split_documents(document)
    db =  Chroma.from_documents(text,embeddings)
    llm = HuggingFaceHub(repo_id="OpenAssistant/oasst-sft-1-pythia-12b",model_kwargs={"temperature": 1.0, "max_length": 256})
    global chain
    chain = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=db.as_retriever())
    return "Document has been loaded successfully"

def answer_query(query):
    answer = chain.run(query)
    short_answer = extract_from_sentence(answer,query)
    return "Question:"+short_answer




# Gradio Block to Upload File
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    pdf_file=gr.File(label='Upload your File',file_types=['.pdf'])

    with gr.Row():

        load_pdf = gr.Button("Load a PDF")
        status = gr.Textbox(label="File Loading Status",placeholder="",interactive='False')
    
    with gr.Row():
        input= gr.Textbox(label="Type in your Query:")
        output=gr.Textbox(label="Output")
    
    submit=gr.Button("Submit")
    submit.click(answer_query,input,output)
    
    
    

    

    load_pdf.click(process_pdf,pdf_file,status)


demo.launch()