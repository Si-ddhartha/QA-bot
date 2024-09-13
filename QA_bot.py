import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

import google.generativeai as genai

from dotenv import load_dotenv

import io
import os
import shutil

load_dotenv()
GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
genai.configure(api_key=GOOGLE_API_KEY)

# Configuration
CHROMA_PATH = 'chroma'
EMBEDDING_MODEL = "models/text-embedding-004"
CHAT_MODEL = "models/gemini-1.5-flash-001"

def load_documents(pdf_doc):
    temp_file = "./temp.pdf"
    with open(temp_file, "wb") as file:
        file.write(pdf_doc.getvalue())
        file_name = pdf_doc.name

    loader = PyPDFLoader(temp_file)
    docs = loader.load()

    return docs

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 150,
        length_function = len,
        add_start_index = True
    )

    chunks = text_splitter.split_documents(documents)

    return chunks

def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        try:
            db = Chroma(persist_directory=CHROMA_PATH)
            db = None
            shutil.rmtree(CHROMA_PATH)
        except Exception as e:
            st.error(f"Error while removing Chroma directory: {e}")
            return 1

    embedding_model = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        task_type="RETRIEVAL_DOCUMENT"
    )

    db = Chroma.from_documents(
        chunks, embedding_model, persist_directory=CHROMA_PATH
    )

def create_prompt_template():
    
    return PromptTemplate.from_template(''' 
        Answer the question as detailed as possible based on the provided context. Make sure to provide all the details. If the answer is not in the provided context just say, "No answer available." Don't give a wrong answer.

        Context:
        {context}

        Question:
        {question}

        Answer:
    ''')

def generate_answer(model, prompt):
    human_message = HumanMessage(content=prompt)
    response = model.invoke([human_message])
    
    return response.content

def process_input(ques: str):
    embedding_function = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        task_type="RETRIEVAL_QUERY"
    )

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(ques)
    context = "\n".join([result.page_content for result, score in results])

    prompt_template = create_prompt_template()
    formatted_prompt = prompt_template.format(context=context, question=ques)

    model = ChatGoogleGenerativeAI(model=CHAT_MODEL)
    answer = generate_answer(model, formatted_prompt)

    st.write(f"Reply: {answer}")

def main():
    st.set_page_config("QA Bot")
    st.header("Chat with PDF")

    user_question = st.text_input("Ask any question related to the pdf file.")

    if user_question:
        process_input(user_question)

    with st.sidebar:
        st.title("Upload File:")

        pdf_doc = st.file_uploader("Upload your PDF file and click on the Submit button.", accept_multiple_files=False)
        
        if st.button("Submit"):
            with st.spinner("Processing..."):
                raw_text = load_documents(pdf_doc)
                text_chunks = split_text(raw_text)
                save_to_chroma(text_chunks)

                st.success("Done")


if __name__ == '__main__':
    main()