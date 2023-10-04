import pandas as pd
from openai.embeddings_utils import get_embedding
import openai
import constants

openai.api_key = constants.APIKEY
# Constants if required (if they're not already set elsewhere)
info_filename = './info.txt'

# Función para vectorizar text
def embed(text):
    return get_embedding(text, engine='text-embedding-ada-002')

# Cargar y dividir el texto del archivo info.txt en párrafos
def create_lunaData_file():
    print('lunaData starting.')
    with open(info_filename, 'r', encoding='utf-8') as file:
        text = file.read()
        print('archivo abierto.')

    # Dividir el texto en párrafos (ajusta el tamaño de los párrafos según tus necesidades)
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    print('dividir parrafos.')

    # Vectorizar cada párrafo y almacenar en un DataFrame
    paragraph_embeddings = [embed(para) for para in paragraphs]  # Vectorizar cada párrafo
    paragraph_df = pd.DataFrame({'text': paragraphs, 'Embedding': paragraph_embeddings})
    print('vectorizado.')
    print(paragraph_df)

    # Guardar los vectores de párrafos en un archivo CSV (lunaData.csv)
    file_path = './lunaData.csv'
    paragraph_df.to_csv(file_path, index=False)
    print('lunaData Created at:', file_path)
    print('lunaData Created.')

#if we run this file directly, it will generate the LunaData
if __name__ == '__main__':
    create_lunaData_file()
