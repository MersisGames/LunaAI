import uuid
from flask import Flask, request, jsonify
import json

from testbot import generate_response_LTM, generate_short_response

app = Flask(__name__)

# Archivo JSON para almacenar respuestas
response_data = {}

short_term_memory_file = str(uuid.uuid4()) + "_STM.txt"  # Declarar la variable global

# Función para generar una respuesta desde el bot
def generate_bot_response(question):
    global short_term_memory_file
    short_response = generate_short_response(question)
    long_response = generate_response_LTM(question, short_term_memory_file)
    return long_response, short_response

@app.route('/ask', methods=['POST'])
def ask_question(long_response, short_response):
    
    # Actualizar el archivo JSON
    response_data['short_response'] = short_response
    response_data['long_response'] = long_response

    with open('responses.json', 'w') as json_file:
        json.dump(response_data, json_file)

    return jsonify({"message": "Respuestas almacenadas con éxito"})



@app.route('/ask_question/<question>', methods=['POST'])
def ask_question_from_url(question):
    # Envía la pregunta al bot y obtén la respuesta
    response = generate_bot_response(question)
    
    return jsonify({"response": response})


@app.route('/get_short_response', methods=['GET'])
def get_short_response():
    if 'short_response' in response_data:
        return jsonify({"short_response": response_data['short_response']})
    else:
        return jsonify({"error": "No hay respuesta corta almacenada"})

@app.route('/get_long_response', methods=['GET'])
def get_long_response():
    if 'long_response' in response_data:
        return jsonify({"long_response": response_data['long_response']})
    else:
        return jsonify({"error": "No hay respuesta larga almacenada"})


if __name__ == '__main__':
    try:
        with open('responses.json', 'r') as json_file:
            response_data = json.load(json_file)
    except FileNotFoundError:
        response_data = {}
    
    app.run(host='0.0.0.0', port=5000)
