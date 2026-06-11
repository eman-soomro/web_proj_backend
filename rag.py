import os
import weaviate
from weaviate.auth import AuthApiKey
import requests


class TrendRAG:
    def __init__(self):
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),
            auth_credentials=AuthApiKey(os.getenv("WEAVIATE_API_KEY"))
        )
        print("Connected?", self.client.is_connected())

        # Delay model loading until needed
        self.model = None

        # Ensure schema exists
        if "Paper" not in [c.name for c in self.client.collections.list_all()]:
            self.client.collections.create(
                name="Paper",
                properties=[
                    {"name": "title", "dataType": "text"},
                    {"name": "abstract", "dataType": "text"},
                    {"name": "year", "dataType": "int"},
                    {"name": "authors", "dataType": "text"},
                    {"name": "citations", "dataType": "int"}
                ]
            )

    def get_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        return self.model

    def ingest_from_semantic_scholar(self, keyword, limit=10):
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={keyword}&limit={limit}"
        response = requests.get(url)
        if response.status_code == 200:
            papers = response.json().get("data", [])
            for paper in papers:
                self.client.collections.get("Paper").data.insert({
                    "title": paper.get("title"),
                    "abstract": paper.get("abstract", ""),
                    "year": paper.get("year", 0),
                    "authors": ", ".join([a.get("name") for a in paper.get("authors", [])]),
                    "citations": paper.get("citationCount", 0)
                })

    def query(self, keyword, top_k=5):
        model = self.get_model()
        embedding = model.encode(keyword).tolist()
        results = self.client.collections.get("Paper").query.near_vector(
            vector=embedding,
            limit=top_k
        )
        return results.objects

