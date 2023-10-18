import pandas as pd
from openai.embeddings_utils import get_embedding
import openai
import constants
import fitz  # PyMuPDF

openai.api_key = "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"

file = '../solarData.pdf'

def extract_text_from_pdf(pdf_file):
    text = ""
    doc = fitz.open(pdf_file)
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def generate_embeddings_and_save(input_file="data.pdf", output_file="newData.csv"):
    # Extract text from the PDF file
    pdf_text = extract_text_from_pdf(input_file)

    # Split the text into paragraphs
    paragraphs = pdf_text.split("\n\n")  # Assuming paragraphs are separated by two line breaks

    # Initialize lists to store paragraphs and their embeddings
    paragraph_texts = []
    paragraph_embeddings = []

    # Generate embeddings for each paragraph
    for paragraph in paragraphs:
        paragraph = paragraph.strip()  # Remove extra whitespace at the beginning and end
        if paragraph:
            embedding = get_embedding(paragraph, engine="text-embedding-ada-002")
            paragraph_texts.append(paragraph)
            paragraph_embeddings.append(embedding)

    # Create a DataFrame with paragraphs and their embeddings
    data = pd.DataFrame({"text": paragraph_texts, "Embedding": paragraph_embeddings})

    # Save the data to a CSV file with the new name
    data.to_csv(output_file, index=False)

# Call the function to process the data.pdf file and save the results to testData.csv
generate_embeddings_and_save(file, "newData.csv")
