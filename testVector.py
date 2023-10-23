import pandas as pd
import PyPDF2
from openai.embeddings_utils import get_embedding
import openai
import constants

openai.api_key = constants.APIKEY

def extract_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def generate_embeddings_and_save(input_pdf, output_file="newData.csv"):
    text = extract_text_from_pdf(input_pdf)

    # Split the text into paragraphs
    paragraphs = text.split("\n\n")

    paragraph_texts = []
    paragraph_embeddings = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if paragraph:
            embedding = get_embedding(paragraph, engine="text-embedding-ada-002")
            paragraph_texts.append(paragraph)
            paragraph_embeddings.append(embedding)

    data = pd.DataFrame({"text": paragraph_texts, "Embedding": paragraph_embeddings})
    data.to_csv(output_file, index=False)

# Llama a la función para procesar el archivo PDF y guardar los resultados en newData.csv
generate_embeddings_and_save("../space1.pdf", "newData.csv")
