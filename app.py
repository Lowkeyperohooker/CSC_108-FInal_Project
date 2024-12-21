from flask import Flask, request, jsonify
from my_script import main_logic  # Import the script's logic

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to my Python app!"

@app.route("/process", methods=["POST"])
def process():
    data = request.json  # Get JSON input
    input_data = data.get("input")
    if input_data is None:
        return jsonify({"error": "Missing 'input' in request"}), 400
    result = main_logic(input_data)  # Use the script's logic
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
