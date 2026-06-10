import os
import weaviate
from weaviate.auth import AuthApiKey
from sentence_transformers import SentenceTransformer
import requests

class TrendRAG:
    def __init__(self):
        # Connect to your cloud sandbox using env vars
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),
            auth_credentials=AuthApiKey(os.getenv("WEAVIATE_API_KEY"))
        )
        print("Connected?", self.client.is_connected())

        # Embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

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

