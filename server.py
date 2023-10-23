from flask import Flask, request, jsonify
import json

app = Flask(__name__)

# Archivo JSON para almacenar respuestas
response_data = {}

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.json['question']
    response = request.json['response']

    response_data[question] = response

    with open('responses.json', 'w') as json_file:
        json.dump(response_data, json_file)

    return jsonify({"message": "Respuesta almacenada con éxito"})

@app.route('/get/<question>', methods=['GET'])
def get_response(question):
    if question in response_data:
        return jsonify({"response": response_data[question]})
    else:
        return jsonify({"error": "La pregunta no tiene respuesta"})

if __name__ == '__main__':
    try:
        with open('responses.json', 'r') as json_file:
            response_data = json.load(json_file)
    except FileNotFoundError:
        response_data = {}
    
    app.run(host='0.0.0.0', port=5000)
