import pandas as pd
from openai.embeddings_utils import get_embedding
import openai
import constants

openai.api_key = constants.APIKEY

def generate_embeddings_and_save(input_file="data.txt", output_file="testData.csv"):  # Change the output file name
    # Read the content of the .txt file and split it into paragraphs
    with open(input_file, "r", encoding="utf-8") as file:
        paragraphs = file.read().split("\n\n")  # Assuming paragraphs are separated by two line breaks

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

# Call the function to process the data.txt file and save the results to testData.csv
generate_embeddings_and_save("data.txt", "testData.csv")
