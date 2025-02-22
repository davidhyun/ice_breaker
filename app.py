from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from ice_breaker import ice_break_with

load_dotenv()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    name = request.form["name"]
    summary, profile_pic_url = ice_break_with(name=name)
    result = jsonify({"information": summary.to_dict(), "profile_pic_url": profile_pic_url})
    
    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)