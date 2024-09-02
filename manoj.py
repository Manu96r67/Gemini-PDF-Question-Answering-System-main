# import os
# import fitz  # PyMuPDF
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# # Configure the Google API
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# def get_pdf_text(pdf_paths):
#     """
#     Extracts text directly from a list of PDF file paths.

#     Args:
#         pdf_paths: A list of file paths to the PDF documents.

#     Returns:
#         A string containing the extracted text from all PDF documents.
#     """
#     text = ""
#     for pdf_path in pdf_paths:
#         if os.path.exists(pdf_path):
#             try:
#                 with open(pdf_path, "rb") as pdf_file:
#                     doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
#                     for page_num in range(len(doc)):
#                         page = doc.load_page(page_num)
#                         text += page.get_text()  # Extract text directly from the page
#             except Exception as e:
#                 print(f"Error processing file {pdf_path}: {e}")
#                 text += f"Error processing file {pdf_path}: {e}\n"
#         else:
#             print(f"File not found: {pdf_path}")
#             text += f"File not found: {pdf_path}\n"
#     return text


# def get_text_chunks(text):
#     """
#     Splits a large text into smaller chunks for efficient processing.

#     Args:
#         text: A string containing the text to be split.

#     Returns:
#         A list of text chunks.
#     """
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks


# def get_vector_store(text_chunks):
#     """
#     Creates a vector store from a list of text chunks.

#     Args:
#         text_chunks: A list of text chunks.

#     Returns:
#         A FAISS vector store.
#     """
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faisss_index")
#     return vector_store


# def get_conversational_chain():
#     """
#     Creates a conversational chain for question answering.

#     Returns:
#         A LangChain question-answering chain.
#     """
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not available in the context, just say, "answer is not available in the context", don't provide the wrong answer.

#     Context:
#     {context}?

#     Question:
#     {question}

#     Answer:
#     """

#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain


# def user_input(user_question, pdf_paths):
#     """
#     Processes user input and generates a response using the conversational chain,
#     providing both the user's question and the processed PDF text as context.

#     Args:
#         user_question: The user's question.
#         pdf_paths: List of file paths to the PDF documents.

#     Returns:
#         The generated response from the conversational chain.
#     """
#     # Extract text from PDFs each time a question is asked
#     processed_pdf_text = get_pdf_text([path.strip() for path in pdf_paths])

#     # Split the processed text into chunks
#     text_chunks = get_text_chunks(processed_pdf_text)

#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
#     # Load existing FAISS index or create a new one
#     try:
#         new_db = FAISS.load_local("faisss_index", embeddings, allow_dangerous_deserialization=True)
#         # Update FAISS index with new PDF content
#         new_db.add_texts(text_chunks)
#     except Exception as e:
#         print(f"Error loading FAISS index or creating a new one: {e}")
#         new_db = get_vector_store(text_chunks)
    
#     # Save the updated FAISS index
#     new_db.save_local("faisss_index")

#     # Perform similarity search
#     docs = new_db.similarity_search(user_question)

#     chain = get_conversational_chain()

#     # Combine user question and processed PDF text as context
#     context = f"{processed_pdf_text}\n\nQuestion: {user_question}"

#     response = chain({"input_documents": docs, "question": user_question, "context": context}, return_only_outputs=True)

#     print("Reply: ", response["output_text"])



# def main():
#     """
#     Main function to run the program in the terminal.
#     """
#     # Get PDF file paths from the user
#     pdf_paths = input("Enter the paths to the PDF files, separated by commas: ").split(",")

#     # Ask the user's question
#     while True:
#         # Ask the user's question
#         user_question = input("Ask a question about the PDF content (or type 'exit' to quit): ")

#         if user_question.lower() == "exit":
#             break

#         if user_question:
#             print("Processing PDFs...")
#             user_input(user_question, pdf_paths)
#         else:
#             print("No question provided.")


# if __name__ == "__main__":
#     main()
import os
import fitz  # PyMuPDF
import speech_recognition as sr
from gtts import gTTS
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import tempfile
import playsound 

# Configure the Google API
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# language_code = 'kn'
def get_pdf_text(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            try:
                with open(pdf_path, "rb") as pdf_file:
                    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        text += page.get_text()
            except Exception as e:
                print(f"Error processing file {pdf_path}: {e}")
                text += f"Error processing file {pdf_path}: {e}\n"
        else:
            print(f"File not found: {pdf_path}")
            text += f"File not found: {pdf_path}\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faisss_index")
    return vector_store

def get_conversational_chain():
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

def user_input(user_question, pdf_paths):
    processed_pdf_text = get_pdf_text([path.strip() for path in pdf_paths])
    text_chunks = get_text_chunks(processed_pdf_text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local("faisss_index", embeddings, allow_dangerous_deserialization=True)
        new_db.add_texts(text_chunks)
    except Exception as e:
        print(f"Error loading FAISS index or creating a new one: {e}")
        new_db = get_vector_store(text_chunks)
    new_db.save_local("faisss_index")
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    context = f"{processed_pdf_text}\n\nQuestion: {user_question}"
    response = chain({"input_documents": docs, "question": user_question, "context": context}, return_only_outputs=True)
    return response["output_text"]

def speak_text(text, language):
    tts = gTTS(text=text, lang=language)
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts.save(f"{fp.name}.mp3")
        playsound.playsound(f"{fp.name}.mp3")

def listen_for_question(language):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"Listening for your question in {language}...")
        audio = recognizer.listen(source)
        try:
            question = recognizer.recognize_google(audio, language=f"{language}-IN")
            print(f"You asked: {question}")
            return question
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            print("Sorry, my speech service is down.")
            return None

def main():
    # Get the desired language from the user
    language_choice = input("Enter the language code ('en' for English, 'kn' for Kannada, 'ta' for Tamil, 'te' for Telugu, 'gu' for Gujarati): ").strip().lower()

    # Set the language code based on the user's input
    if language_choice not in ['en', 'kn', 'ta', 'te', 'gu','or']:
        print("Invalid language code. Defaulting to English.")
        language_choice = 'en'

    pdf_paths = input("Enter the paths to the PDF files, separated by commas: ").split(",")

    while True:
        user_question = listen_for_question(language_choice)
        if user_question and user_question.lower() != "exit":
            print("Processing PDFs...")
            answer = user_input(user_question, pdf_paths)
            print("Reply: ", answer)
            speak_text(answer, language_choice)
        elif user_question and user_question.lower() == "exit":
            break
        else:
            print("No question provided.")

if __name__ == "__main__":
    main()

