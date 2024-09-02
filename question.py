import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
import random

# Configure the Google API
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

from langdetect import detect

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"  # Default to English if detection fails


def get_pdf_text(pdf_paths):
    """
    Extracts text directly from a list of PDF file paths.
    """
    text = ""
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            try:
                with open(pdf_path, "rb") as pdf_file:
                    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        text += page.get_text("text")  # Extract text directly from the page
            except Exception as e:
                print(f"Error processing file {pdf_path}: {e}")
                text += f"Error processing file {pdf_path}: {e}\n"
        else:
            print(f"File not found: {pdf_path}")
            text += f"File not found: {pdf_path}\n"
    return text

def get_text_chunks(text):
    """
    Splits a large text into smaller chunks for efficient processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

def get_conversational_chain(language='en'):
    """
    Creates a conversational chain for generating questions in the specified language.
    """
    prompt_template = f"""
    Based on the provided context, generate a meaningful multiple-choice question in {language}.
    Provide the question, one correct answer, and three incorrect but plausible distractor answers.

    
    Context:
    {{context}}

    Question:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

import random

def generate_questions(text_chunks, num_questions=10, language_choice='en'):
    """
    Generates a list of multiple-choice questions from the provided text chunks.
    """
    chain = get_conversational_chain(language=language_choice)
    questions = []

    for i, chunk in enumerate(text_chunks):
        if len(questions) >= num_questions:
            break
        context = chunk
        print(f"Processing chunk {i+1}/{len(text_chunks)}")
        try:
            response = chain({"input_documents": [Document(page_content=context)], "context": context}, return_only_outputs=True)
            lines = [line.strip() for line in response['output_text'].strip().split('\n') if line.strip()]

            question = ""
            correct_answer = ""
            incorrect_answers = []

            for i, line in enumerate(lines):
                if "**Question:**" in line:
                    question = lines[i + 1].strip()
                elif "**Correct Answer:**" in line:
                    correct_answer = lines[i + 1].strip().lstrip('*').strip()
                elif "**Incorrect Answers:**" in line or "**Incorrect Distractor Answers:**" in line or "**Incorrect Distractors:**" in line:
                    j = i + 1
                    while j < len(lines) and not lines[j].startswith("**"):
                        incorrect_answers.append(lines[j].strip().lstrip('*').strip())
                        j += 1

            # Ensure we have a valid question set
            if question and correct_answer and len(incorrect_answers) >= 1:
                options = incorrect_answers + [correct_answer]
                options = list(set(options))  # Remove duplicates
                random.shuffle(options)  # Shuffle to randomize answer positions

                # Store options as a list
                questions.append({
                    'question': question,
                    'options': options,
                    'correct_answer': correct_answer
                })

            else:
                print(f"Skipping invalid question set: {lines}")

        except Exception as e:
            print(f"Error generating question {i+1}: {e}")
    
    return questions




def main():
    """
    Main function to run the program in the terminal.
    """
    # Get PDF file paths from the user
    pdf_paths = input("Enter the paths to the PDF files, separated by commas: ").split(",")

    # Number of questions to generate
    num_questions = int(input("Enter the number of questions you want to generate: "))

    print("Processing PDFs...")
    processed_pdf_text = get_pdf_text([path.strip() for path in pdf_paths])

    if processed_pdf_text:
        detected_language = detect_language(processed_pdf_text)
        print(f"Detected Language: {detected_language}")

        text_chunks = get_text_chunks(processed_pdf_text)
        print(f"Total chunks generated: {len(text_chunks)}")  # Debugging output
        questions = generate_questions(text_chunks, num_questions, language_choice=detected_language)

        if questions:
            print("Generated Questions:")
            for i, question in enumerate(questions, 1):
                print(f"{i}. {question['question']}")
                for j, option in enumerate(question['options'], 1):
                    print(f"   {chr(64+j)}. {option}")
                correct_answer_index = question['options'].index(question['correct_answer']) + 1
                print(f"Correct Answer: {chr(64+correct_answer_index)}\n")

        else:
            print("No questions could be generated from the text.")
    else:
        print("No text was extracted from the PDFs.")

if __name__ == "__main__":
    main()

# from flask import Flask, request, render_template, jsonify
# import os
# import fitz  # PyMuPDF
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from langchain_core.documents import Document
# from dotenv import load_dotenv
# import random
# from langdetect import detect

# # Initialize the Flask app
# app = Flask(__name__)

# # Load Google API key
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# def detect_language(text):
#     try:
#         return detect(text)
#     except:
#         return "en"  # Default to English if detection fails

# def get_pdf_text(pdf_paths):
#     text = ""
#     for pdf_path in pdf_paths:
#         if os.path.exists(pdf_path):
#             try:
#                 with open(pdf_path, "rb") as pdf_file:
#                     doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
#                     for page_num in range(len(doc)):
#                         page = doc.load_page(page_num)
#                         text += page.get_text("text")
#             except Exception as e:
#                 print(f"Error processing file {pdf_path}: {e}")
#                 text += f"Error processing file {pdf_path}: {e}\n"
#         else:
#             print(f"File not found: {pdf_path}")
#             text += f"File not found: {pdf_path}\n"
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_conversational_chain(language='en'):
#     prompt_template = f"""
#     Based on the provided context, generate a meaningful multiple-choice question in {language}.
#     Provide the question, one correct answer, and three incorrect but plausible distractor answers.

#     Context:
#     {{context}}

#     Question:
#     """

#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# def generate_questions(text_chunks, num_questions=10, language_choice='en'):
#     chain = get_conversational_chain(language=language_choice)
#     questions = []

#     for i, chunk in enumerate(text_chunks):
#         if len(questions) >= num_questions:
#             break
#         context = chunk
#         try:
#             response = chain({"input_documents": [Document(page_content=context)], "context": context}, return_only_outputs=True)
#             lines = [line.strip() for line in response['output_text'].strip().split('\n') if line.strip()]

#             question = ""
#             correct_answer = ""
#             incorrect_answers = []

#             for i, line in enumerate(lines):
#                 if "**Question:**" in line:
#                     question = lines[i + 1].strip()
#                 elif "**Correct Answer:**" in line:
#                     correct_answer = lines[i + 1].strip().lstrip('*').strip()
#                 elif "**Incorrect Answers:**" in line or "**Incorrect Distractor Answers:**" in line or "**Incorrect Distractors:**" in line:
#                     j = i + 1
#                     while j < len(lines) and not lines[j].startswith("**"):
#                         incorrect_answers.append(lines[j].strip().lstrip('*').strip())
#                         j += 1

#             if question and correct_answer and len(incorrect_answers) >= 1:
#                 options = incorrect_answers + [correct_answer]
#                 options = list(set(options))  # Remove duplicates
#                 random.shuffle(options)  # Shuffle to randomize answer positions

#                 questions.append({
#                     'question': question,
#                     'options': options,
#                     'correct_answer': correct_answer
#                 })

#         except Exception as e:
#             print(f"Error generating question {i+1}: {e}")
    
#     return questions

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_pdf():
#     if 'pdf_files' not in request.files:
#         return jsonify({"error": "No files provided"}), 400

#     files = request.files.getlist('pdf_files')
#     file_paths = []

#     for file in files:
#         file_path = os.path.join('uploads', file.filename)
#         file.save(file_path)
#         file_paths.append(file_path)

#     processed_pdf_text = get_pdf_text(file_paths)

#     if processed_pdf_text:
#         detected_language = detect_language(processed_pdf_text)
#         text_chunks = get_text_chunks(processed_pdf_text)
#         questions = generate_questions(text_chunks, language_choice=detected_language)

#         return jsonify({"questions": questions, "language": detected_language})
#     else:
#         return jsonify({"error": "No text could be extracted from the PDFs"}), 500

# if __name__ == "__main__":
#     app.run(debug=True)



