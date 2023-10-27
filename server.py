import uuid
from flask import Flask, request, jsonify
import json
import os
from testbot import generate_response_LTM, generate_short_response

app = Flask(__name__)

# Archivo JSON para almacenar respuestas
response_data = {}

SHORT_TERM_MEMORY_FILE = str(uuid.uuid4()) + "_STM.txt"  # Declarar la variable global


@app.route('/ask_question/<question>', methods=['POST'])
def generate_bot_response(question):
    global SHORT_TERM_MEMORY_FILE
    short_response = generate_short_response(question)
    if os.path.exists('shortResponse.json'):
        os.remove('shortResponse.json')
    with open('shortResponse.json', 'w') as json_file:
        json.dump({"short_response": short_response}, json_file)
    
    long_response = generate_response_LTM(question, SHORT_TERM_MEMORY_FILE)
    if os.path.exists('longResponse.json'):
        os.remove('longResponse.json')
    with open('longResponse.json', 'w') as json_file:
        json.dump({"long_response": long_response}, json_file)


    return jsonify({"message": "Respuestas almacenadas con éxito"})

@app.route('/get_short_response', methods=['GET'])
def get_short_response():
    try:
        if os.path.exists('shortResponse.json'):
            with open('shortResponse.json', 'r') as json_file:
                response_data = json.load(json_file)
                return jsonify(response_data)
        else:
            return jsonify({"error": "No hay respuesta corta disponible"})
    except FileNotFoundError:
        return jsonify({"error": "No hay respuesta corta disponible"})

@app.route('/get_long_response', methods=['GET'])
def get_long_response():
    try:
        if os.path.exists('longResponse.json'):
            with open('longResponse.json', 'r') as json_file:
                response_data = json.load(json_file)
                return jsonify(response_data)
        else:
            return jsonify({"error": "No hay respuesta larga disponible"})
    except FileNotFoundError:
        return jsonify({"error": "No hay respuesta larga disponible"})


if __name__ == '__main__':
    try:
        with open('responses.json', 'r') as json_file:
            response_data = json.load(json_file)
    except FileNotFoundError:
        response_data = {}
    
    app.run(host='0.0.0.0', port=5000)
