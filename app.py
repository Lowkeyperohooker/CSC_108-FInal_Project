from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def run_script():
    if request.method == "POST":
        # Assume the script takes input from the user
        input_data = request.json.get("input", "")
        # Call your script's functionality here
        result = f"Your input was: {input_data}"
        return jsonify({"result": result})
    return "Welcome to the Python script web service!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
