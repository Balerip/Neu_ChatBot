import os
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv
from llama_index.core import Document
load_dotenv()
class DataLoader:
    def __init__(self, directory=os.environ.get("DATA_STORAGE_DIRECTORY")):
       self.directory = directory

    def load_documents(self, directory=None):
        if directory:
            self.directory = directory
            
        if not os.path.exists(self.directory):
            raise Exception("Directory does not exist")
        documents = SimpleDirectoryReader(self.directory).load_data()
        return documents



 

