import os
from typing import List, Optional
from llama_index.core import VectorStoreIndex
from . import data_loader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from dotenv import load_dotenv
 
load_dotenv()
 
class Index:
    def __init__(self, directory: str = os.environ.get("DATA_STORAGE_DIRECTORY"),
                 storage_directory: str = os.environ.get("INDEX_STORAGE_DIRECTORY")) -> None:
        self.directory = directory
        self.storage_directory = storage_directory
        self.index = None
        print(f"Data storage directory: {self.directory}")
        print(f"Index storage directory: {self.storage_directory}")
 
    def create_index(self, documents: List[str]) -> VectorStoreIndex:
        """Creates an index from the provided documents."""
        self.embedding = OllamaEmbedding(model_name="mxbai-embed-large",)
        Settings.embed_model = self.embedding
        index = VectorStoreIndex.from_documents(documents, Settings=Settings)
        return index
    def persist_index(self, index: VectorStoreIndex) -> None:
        """Persists the provided index to the storage directory."""
        if not os.path.exists(self.storage_directory):
            os.makedirs(self.storage_directory)
        index.storage_context.persist(persist_dir=self.storage_directory)
    def load_index(self) -> Optional[VectorStoreIndex]:
        """Loads the index from storage, or creates a new one if necessary."""
        documents = data_loader.DataLoader().load_documents(self.directory)
        self.index = self.create_index(documents)
        self.persist_index(self.index)
        return self.index