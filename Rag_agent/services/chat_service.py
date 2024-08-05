import os
from fastapi import FastAPI,HTTPException
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding  # Assuming this is your embedding model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core import VectorStoreIndex, Settings
from . import index
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

    

class Agent:
    def __init__(self, directory: str, storage_directory: str):
        self.directory = directory
        self.storage_directory = storage_directory
        print(f"Agent class, Data storage directory: {self.directory}")
        print(f"Agent class, Index storage directory: {self.storage_directory}")

        self.index = index.Index(directory, storage_directory).load_index()
        if self.index is None:
            raise ValueError("Failed to load or create the index.")
        
        print("Creating embedding model...")
        embedding_model = OllamaEmbedding(model_name="mxbai-embed-large")

        print("Creating LLM model...")
        llm_model = Ollama(model="llama3", request_timeout=60.0)

    # Initialize Settings with the models
        Settings.embed_model=embedding_model
        Settings.llm=llm_model
        # Initialize query engine with similarity_top_k=3
       
        print("Creating query engine...")
        self.query_engine = self.index.as_query_engine(
            embedding_model=  Settings.embed_model# Use the embedding model
        )
        
       
        
        # # Define metadata for the tool
        self.metadata = ToolMetadata(
            name="QueryEngineTool",
            description=("Handles queries related to Computer Science courses, Data Science Courses, Information Systems courses, Project Management and CPS Analytics couurses")
        )
        
        # Initialize the query engine tool with metadata
        self.query_engine_tool = QueryEngineTool(
            query_engine=self.query_engine,
            metadata=self.metadata
        )
     
        # Define ReAct Agent with the query engine tool
        # self.chat_agent=self.index.as_chat_engine(chat_mode="react", llm=Settings.llm, verbose=True)
        
        self.query_agent= ReActAgent.from_tools([self.query_engine_tool], llm= Settings.llm, verbose=True)
        print("Agent created successfully.")
        
    # def query(self, query: str) -> str:
    #     if self.index is None:
    #         raise HTTPException(status_code=500, detail="Index is not loaded.")
        
    #     # Use the ReAct agent for querying
    #     response = self.agent.chat(query)
    #     return response

    def get_query_react_agent(self) -> ReActAgent:
        return self.query_agent
    # def get_chat_react_agent(self) -> ReActAgent:
    #     return self.chat_agent
    
async def get_query_response(message: str) -> str:
    """
    Mocks the processing of a chat message and returns a response.

    Args:
        message (str): The chat message to be processed.

    Returns:
        str: A mock response to the chat message.
    """
    try:
        directory = os.environ.get("DATA_STORAGE_DIRECTORY")
        storage_directory = os.environ.get("INDEX_STORAGE_DIRECTORY")

        print(f"query resp - DataLoader directory: {directory}")
        print(f"query resp - Index storage directory: {storage_directory}")
         
        # Configure and query the primary agent
        print(f"Attempting to create agent")
        agent = Agent(directory, storage_directory)

        react_agent = agent.get_query_react_agent()
        query_response = react_agent.chat(message)

        # Check if query_response has the attribute 'response'
        if hasattr(query_response, 'response'):
            agent_response = query_response.response
            print(f"Agent response: {agent_response}")
        else:
            print(f"Query response does not have 'response' attribute: {query_response}")
            agent_response = "Response attribute missing"
     
        return f"This is a mock response to your message: {agent_response}"
    except Exception as e:
        print(f"Exception in ml.Rag_agent.chat_service.get_query_reponse: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
   
# def get_chat_response(message: str) -> str:
#     """
#     Mocks the processing of a chat message and returns a response.

#     Args:
#         message (str): The chat message to be processed.

#     Returns:
#         str: A mock response to the chat message.
#     """
#     try:
#         directory = os.environ.get("DATA_STORAGE_DIRECTORY")
#         storage_directory = os.environ.get("INDEX_STORAGE_DIRECTORY")
         
#             # Configure and query the primary agent
#         agent = Agent(directory, storage_directory)
#         chat_react_agent = agent.get_chat_react_agent()
#         chat_response = chat_react_agent.chat(message)
#         chat_agent_response=chat_response.response
     
#         return f"This is a mock response to your message: {chat_agent_response}"
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
# Include if needed later
# """
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="localhost", port=8003)
# """


# Run the test
# question_1 = "Tell me about the prerequisites for the Web Development tools and methods course in Information Systems program?"
# await get_query_response(question_1)
# question_2="What are the prerequisites and topics covered in the Intermediate Programming with Data?"
# get_query_response(question_2)