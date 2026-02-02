import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG

# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)

# Create embedder
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# Create retriever
retriever = VectorRetriever(
    driver,
    index_name="moviePlots",
    embedder=embedder,
    return_properties=["title", "plot"],
)

# Create the LLM
llm = OpenAILLM(
    model_name="gpt-4o",
    model_params={
        "temperature": 0.4,
        #"max_tokens": 1000
    }
)

# Create GraphRAG pipeline
rag =  GraphRAG(
    retriever=retriever,
    llm=llm,
)

# Search
query_text = "Find me movies about toys coming alive" 

response = rag.search(
    query_text=query_text,
    retriever_config={"top_k":5},
    return_context=True
)

print(response.answer)
print("context:", response.retriever_result.items)

# CLose the database connection
driver.close()