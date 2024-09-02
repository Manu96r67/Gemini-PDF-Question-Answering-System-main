import streamlit as st
import os
import fitz  # PyMuPDFc
# import pytesseract
# from PIL import Image
# from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
# pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    """
    Extracts text directly from a list of PDF documents without using OCR.

    Args:
        pdf_docs: A list of PDF documents as UploadedFile objects.

    Returns:
        A string containing the extracted text from all PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        if pdf is not None:
            pdf_bytes = pdf.read()
            if pdf_bytes:
                try:
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        text += page.get_text()  # Extract text directly from the page
                except Exception as e:
                    st.error(f"Error processing file: {e}")
                    text += f"Error processing file: {e}\n"
            else:
                st.error("Uploaded file is empty.")
                text += "Uploaded file is empty.\n"
        else:
            st.error("File not found.")
            text += "File not found.\n"
    return text





def get_text_chunks(text):
    """
    Splits a large text into smaller chunks for efficient processing.

    Args:
        text: A string containing the text to be split.

    Returns:
        A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """
    Creates a vector store from a list of text chunks.

    Args:
        text_chunks: A list of text chunks.

    Returns:
        A FAISS vector store.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faisss_index")
    return vector_store


def get_conversational_chain():
    """
    Creates a conversational chain for question answering.

    Returns:
        A LangChain question-answering chain.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not available in the context, just say, "answer is not available in the context", don't provide the wrong answer.

    Context:
    {context}?

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question, processed_pdf_text):
    """
    Processes user input and generates a response using the conversational chain,
    providing both the user's question and the processed PDF text as context.

    Args:
        user_question: The user's question.
        processed_pdf_text: The processed text extracted from the uploaded PDF files.

    Returns:
        The generated response from the conversational chain.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faisss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    # Combine user question and processed PDF text as context
    context = f"{processed_pdf_text}\n\nQuestion: {user_question}"

    response = chain({"input_documents": docs, "question": user_question, "context": context}, return_only_outputs=True)

    st.write("Reply: ", response["output_text"])


def main():
    """
    Main function for the Streamlit app.
    """
    st.set_page_config("Chat With Multiple PDF")
    st.header("Chat with PDFs powered by CSPL 🙋‍♂️")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        if st.session_state.get("pdf_docs"):
            st.write("Processing PDFs...")
            processed_pdf_text = get_pdf_text(st.session_state["pdf_docs"])  # Use get_pdf_text instead of OCR
            user_input(user_question, processed_pdf_text)
        else:
            st.error("Please upload PDF files first.")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload Files & Click Submit to Proceed", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if pdf_docs:
                    st.session_state["pdf_docs"] = pdf_docs
                    st.write(f"Uploaded {len(pdf_docs)} files.")
                    raw_text = get_pdf_text(pdf_docs)  # Use get_pdf_text instead of OCR
                    st.write(f"Extracted text: {raw_text[:500]}...")  # Display the first 500 characters
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    chain = get_conversational_chain()
                    st.session_state["text_chunks"] = text_chunks
                    st.session_state["vector_store"] = vector_store
                    st.session_state["chain"] = chain
                    st.success("PDFs processed successfully!")
                else:
                    st.error("No files uploaded.")
                    
        if st.button("Reset"):
            st.session_state["pdf_docs"] = []
            st.session_state["text_chunks"] = []
            st.session_state["vector_store"] = None
            st.session_state["chain"] = None
            st.experimental_rerun()

        if st.session_state.get("pdf_docs"):
            st.subheader("Uploaded Files:")
            for i, pdf_doc in enumerate(st.session_state["pdf_docs"]):
                st.write(f"{i+1}. {pdf_doc.name}")

if __name__ == "__main__":
    main()

