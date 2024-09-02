import os
import fitz  # PyMuPDF
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from google.api_core.exceptions import InternalServerError

# Load environment variables
load_dotenv()

# Configure the Google API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the ChatGoogleGenerativeAI model
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

# Define the prompt template for extracting key topics
topic_prompt_template = """
Extract key topics from the following text. Provide a list of key topics that are relevant and important.

Text:
{text}
"""
def get_pdf_text(pdf_path, page_numbers):
    """
    Extracts text directly from specific pages of a PDF file.

    Args:
        pdf_path: The file path to the PDF document.
        page_numbers: A list of page numbers to extract text from.

    Returns:
        A string containing the extracted text from the specified pages.
    """
    text = ""
    if os.path.exists(pdf_path):
        try:
            with open(pdf_path, "rb") as pdf_file:
                doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
                for page_num in page_numbers:
                    if page_num <= len(doc):
                        page = doc.load_page(page_num - 1)  # Pages are zero-indexed
                        text += page.get_text()  # Extract text directly from the page
                    else:
                        print(f"Page {page_num} is out of range for {pdf_path}")
        except Exception as e:
            print(f"Error processing file {pdf_path}: {e}")
    else:
        print(f"File not found: {pdf_path}")
    return text

from google.api_core.exceptions import InternalServerError

def extract_key_topics(text):
    """
    Extracts key topics from the provided text using the model.

    Args:
        text: A string of text from which to extract key topics.

    Returns:
        A list of key topics.
    """
    prompt = "Extract key topics from the following text:\n\n{text}\n\nImportant Topics:"
    formatted_prompt = prompt.format(text=text)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": formatted_prompt}
    ]

    try:
        response = model.invoke(messages)
        response_content = response.get('choices', [{}])[0].get('message', {}).get('content', "")
        print("Type of response_content:", type(response_content))  # Debugging line
        print("Response content:", response_content)  # Debugging line

        if isinstance(response_content, str):
            topics = response_content.split("\n")  # Adjust this based on the actual response format
            return list(set(topics))  # Return unique topics
        else:
            print("Unexpected format of response_content.")
    except InternalServerError as e:
        print(f"Internal server error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return []

def generate_study_plan(pdf_path, start_page, end_page):
    """
    Generates a study plan based on the key topics extracted from a range of pages.

    Args:
        pdf_path: The file path to the PDF document.
        start_page: The starting page number.
        end_page: The ending page number.

    Returns:
        A tuple containing:
            - A list of important topics.
            - A detailed study plan.
    """
    study_plan = {}
    combined_text = ""

    # Collect text from all pages in the range
    for page_num in range(start_page, end_page + 1):
        text = get_pdf_text(pdf_path, [page_num])  # Pass a list with a single page number
        if text:
            combined_text += text + "\n"

    if combined_text:
        response_content = extract_key_topics(combined_text)
        important_topics, study_plan_details = parse_response(response_content)
        return important_topics, study_plan_details
    else:
        return ["No significant topics found."], "No study plan available."


def parse_response(response_content):
    """
    Parses the response content into important topics and a study plan.

    Args:
        response_content: The content from the model response.

    Returns:
        A tuple containing:
            - A list of important topics.
            - A detailed study plan.
    """
    # Adjust parsing logic based on the actual response format
    if isinstance(response_content, list):
        important_topics = []
        study_plan = [] 

        for item in response_content:
            # Assuming each item is a line or a part of the response
            if 'Important Topics:' in item:
                # Process item to extract topics
                important_topics = item.split('\n')[1:]  # Example logic
            elif 'Study Plan:' in item:
                # Process item to extract study plan
                study_plan = item.split('\n')[1:]  # Example logic

        return list(set(important_topics)), '\n'.join(study_plan)
    else:
        print("Unexpected format of response_content.")
        return [], "No study plan available."




def user_input(pdf_path, start_page, end_page):
    """
    Processes user input and generates a study plan for the specified PDF pages.

    Args:
        pdf_path: The file path to the PDF document.
        start_page: The starting page number.
        end_page: The ending page number.
    """
    print("Generating study plan...")
    important_topics, study_plan = generate_study_plan(pdf_path, start_page, end_page)
    
    print("\nImportant Topics:")
    for topic in important_topics:
        print(f"- {topic}")

    print("\nStudy Plan:")
    print(study_plan)


def main():
    """
    Main function to run the program in the terminal.
    """
    # Get PDF file path from the user
    pdf_path = input("Enter the path to the PDF file: ").strip()

    # Get page range from the user
    start_page = int(input("Enter the starting page number (inclusive): ").strip())
    end_page = int(input("Enter the ending page number (inclusive): ").strip())

    # Process the user input
    if pdf_path and start_page > 0 and end_page >= start_page:
        user_input(pdf_path, start_page, end_page)
    else:
        print("Invalid input. Please ensure that page numbers are valid.")

if __name__ == "__main__":
    main()
# import os
# from google.cloud import vision
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# import google.generativeai as genai

# # Load environment variables
# load_dotenv()

# # Ensure GOOGLE_APPLICATION_CREDENTIALS is set
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =r"C:\Users\Admin\Downloads\one-moon-423416-e2-176cbabb3187.json"

# # Set up Google Gemini API
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# def extract_text_from_image(image_path):
#     """
#     Extracts text from an image using Google Cloud Vision API.
#     """
#     client = vision.ImageAnnotatorClient()

#     with open(image_path, "rb") as image_file:
#         content = image_file.read()

#     image = vision.Image(content=content)
#     response = client.text_detection(image=image)
#     texts = response.text_annotations

#     if response.error.message:
#         raise Exception(f'{response.error.message}')

#     return texts[0].description if texts else ""

# def extract_info_from_text(text, instruction):
#     """
#     Uses Google Gemini model to extract specific information from the provided text based on user instructions.
#     """
#     model = genai.ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    
#     # Include user instructions in the prompt
#     prompt = f"Based on the following instruction, extract relevant information from the text:\n\nInstruction: {instruction}\n\nText:\n{text}\n\nExtracted Information:"
    
#     response = model.generate(prompt=prompt)
#     return response.text

# def main():
#     image_path = input("Enter the path to the image file: ").strip()

#     # Extract text from the image
#     extracted_text = extract_text_from_image(image_path)
#     print(f"Extracted text:\n{extracted_text}")

#     # Get extraction instructions from the user
#     instruction = input("Enter your instruction for what you want to extract from the text: ").strip()
    
#     # Extract information using the Gemini model based on user instruction
#     extracted_info = extract_info_from_text(extracted_text, instruction)
#     print(f"Extracted information:\n{extracted_info}")

# if __name__ == "__main__":
#     main()
