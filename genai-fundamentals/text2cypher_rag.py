import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import Text2CypherRetriever

# Conexão física com o servidor Neo4j (Driver).
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# Criamos um LLM específico para gerar código. 
# Usamos 'temperature: 0' para que ele seja 100% lógico. 
# Se a temperatura fosse alta, ele poderia "inventar" comandos que não existem.
t2c_llm = OpenAILLM(
    model_name="gpt-4o", 
    model_params={"temperature": 0}
)

# =================================================================
# CONCEITO: O SCHEMA (A DEFINIÇÃO DO MUNDO)
# =================================================================

# Este texto explica ao LLM as "regras do jogo". 
# Ele define quais rótulos (Labels) existem, como Person e Movie, 
# e quais nomes de campos (Properties) ele pode usar. 
# Se você mudar 'title' para 'titulo' aqui, o LLM passará a escrever queries com 'titulo'.
neo4j_schema = """
Node properties:
Person {name: STRING, born: INTEGER}
Movie {tagline: STRING, title: STRING, released: INTEGER}
Genre {name: STRING}
User {name: STRING}

Relationship properties:
ACTED_IN {role: STRING}
RATED {rating: INTEGER}

The relationships:
(:Person)-[:ACTED_IN]->(:Movie)
(:Person)-[:DIRECTED]->(:Movie)
(:User)-[:RATED]->(:Movie)
(:Movie)-[:IN_GENRE]->(:Genre)
"""

# =================================================================
# CONCEITO: FEW-SHOT EXAMPLES (APRENDIZADO POR EXEMPLO)
# =================================================================

# Ensinamos ao modelo o padrão de resposta esperado. 
# Quando o usuário pergunta "Ratings", o modelo já viu aqui que deve 
# usar o relacionamento ':RATED'.
examples = [
    "USER INPUT: 'Get user ratings for a movie?' QUERY: MATCH (u:User)-[r:RATED]->(m:Movie) WHERE m.title = 'Movie Title' RETURN r.rating"
]

# Construímos o Retriever especializado em tradução de texto para Cypher.
# Ele recebe o driver para rodar a query e o t2c_llm para escrevê-la.
retriever = Text2CypherRetriever(
    driver=driver,
    llm=t2c_llm,
    neo4j_schema=neo4j_schema,
    examples=examples,
)

# O Pipeline final. Note que usamos o mesmo objeto 'GraphRAG'.
# A magia é que a interface é a mesma, mas o 'motor' (retriever) mudou.
llm = OpenAILLM(model_name="gpt-4o")
rag = GraphRAG(retriever=retriever, llm=llm)

# Pergunta baseada em um fato exato que está no banco.
query_text = "What year was the movie Babe released?"

# Execução do Pipeline:
# 1. LLM recebe a pergunta + Schema + Exemplos.
# 2. LLM gera o Cypher: MATCH (m:Movie {title: 'Babe'}) RETURN m.released.
# 3. O Driver executa esse Cypher no Neo4j.
# 4. O resultado volta e o LLM final monta a frase de resposta.
response = rag.search(
    query_text=query_text,
    return_context=True
)

print(response.answer)
print("CYPHER :", response.retriever_result.metadata["cypher"]) # Mostra a query gerada pela IA
print("CONTEXT:", response.retriever_result.items) # Mostra o dado vindo direto do banco

driver.close()