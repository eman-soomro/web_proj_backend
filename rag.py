import os
import requests

class TrendRAG:
    def __init__(self):
        self.client = None
        self.model = None

    def get_client(self):
        if self.client is None:
            import weaviate
            from weaviate.auth import AuthApiKey
            from weaviate.collections.config import Property, DataType

            cluster_url = os.getenv("WEAVIATE_CLUSTER_URL")
            api_key = os.getenv("WEAVIATE_API_KEY")
            if not cluster_url or not api_key:
                raise Exception("Weaviate credentials missing")

            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=cluster_url,
                auth_credentials=AuthApiKey(api_key)
            )
            if not self.client.is_connected():
                raise Exception("Failed to connect to Weaviate")

            # Ensure schema exists
            if "Paper" not in [c.name for c in self.client.collections.list_all()]:
                self.client.collections.create(
                    name="Paper",
                    properties=[
                        Property(name="title", data_type=DataType.TEXT),
                        Property(name="abstract", data_type=DataType.TEXT),
                        Property(name="year", data_type=DataType.INT),
                        Property(name="authors", data_type=DataType.TEXT),
                        Property(name="citations", data_type=DataType.INT),
                    ]
                )
        return self.client

    def get_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        return self.model

    def ingest_from_semantic_scholar(self, keyword, limit=10):
        client = self.get_client()
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={keyword}&limit={limit}"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Semantic Scholar error {response.status_code}: {response.text}")

        papers = response.json().get("data", [])
        for paper in papers:
            try:
                client.collections.get("Paper").data.insert({
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", ""),
                    "year": paper.get("year", 0),
                    "authors": ", ".join([a.get("name", "") for a in paper.get("authors", [])]),
                    "citations": paper.get("citationCount", 0)
                })
            except Exception as e:
                # Log but don’t crash
                print("Insert failed:", e)

    def query(self, keyword, top_k=5):
        client = self.get_client()
        model = self.get_model()
        embedding = model.encode(keyword).tolist()
        try:
            results = client.collections.get("Paper").query.near_vector(
                vector=embedding,
                limit=top_k
            )
            return results.objects
        except Exception as e:
            raise Exception(f"Query failed: {e}")
