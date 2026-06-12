from flask import Flask, request, jsonify
from flask_cors import CORS
from training import TrainingAbstraction
from rag import TrendRAG

app = Flask(__name__)
CORS(app, origins=["https://web-proj-frontend.vercel.app"]) 

# Initialize RAG
rag = TrendRAG()

@app.route("/", methods=["GET"])
def home():
    return "Backend is running. Use POST /forecast or POST /rag."

@app.route("/forecast", methods=["POST"])
def forecast():
    data = request.json
    domain = data.get("domain", "general")
    keywords = data.get("keywords", [])
    horizon = data.get("prediction_horizon", 30)

    trainer = TrainingAbstraction(domain, keywords, horizon)
    results = trainer.run_training_pipeline()
    return jsonify(results)

@app.route("/rag", methods=["POST"])
def rag_query():
    try:
        data = request.json
        keyword = data.get("keyword")
        if not keyword:
            return jsonify({"error": "Keyword is required"}), 400

        rag.ingest_from_semantic_scholar(keyword, limit=10)
        results = rag.query(keyword, top_k=5)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
