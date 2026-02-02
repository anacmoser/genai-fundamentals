# =================================================================
# FASE 1: CONFIGURAÇÃO DO AMBIENTE E INFRAESTRUTURA
# =================================================================

import os
from dotenv import load_dotenv

# load_dotenv(): Esta função lê o arquivo '.env' e carrega as chaves na memória 
# do processo atual. É uma prática de segurança: nunca escreva senhas no código.
load_dotenv()

# Importação do Driver oficial do Neo4j. 
# O "Driver" é o software que gerencia o protocolo binário (Bolt) de comunicação.
from neo4j import GraphDatabase

# Importação de componentes da biblioteca 'neo4j-graphrag'.
# Esta biblioteca foi instalada via pip e é uma camada de abstração (SDK) 
# criada pela Neo4j para você não ter que escrever centenas de linhas de código manual.
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import VectorCypherRetriever

# =================================================================
# FASE 2: CONEXÃO COM O BANCO (O DRIVER)
# =================================================================

# Aqui instanciamos o objeto 'driver'. 
# Pense nele como um "Cabo de Rede Virtual" que fica permanentemente 
# conectado ao seu Sandbox. Ele gerencia o "pool" de conexões (se uma cair, ele usa outra).
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), # O endereço (ex: bolt://44.200.x.x:7687)
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")  # A senha do Sandbox
    )
)

# =================================================================
# FASE 3: O EMBEDDER (TRADUTOR MATEMÁTICO)
# =================================================================

# O LLM não entende texto, entende números. O 'OpenAIEmbeddings' é o serviço 
# que envia o texto da sua pergunta para a OpenAI e recebe de volta uma lista 
# de 1536 números decimais (Vetor). 
# Este vetor representa o "significado semântico" da frase.
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# =================================================================
# FASE 4: A QUERY DE RECUPERAÇÃO (O MAPA DO GRAFO)
# =================================================================

# Por que escrevemos Cypher aqui? 
# O Vector Search só te entrega o nó "Filme". Se você quer saber os atores, 
# o Vetor não sabe navegar. Esta query diz ao sistema: 
# "Depois que você achar o filme pelo vetor, siga as setas (relações) para pegar o resto".
retrieval_query = """
MATCH (node) <-[r:RATED]-()                 // Acha os relacionamentos de avaliação
RETURN
    node.title AS title,                    // Extrai a propriedade título
    node.plot AS plot,                      // Extrai a sinopse
    score AS similarityScore,               // O quão parecido o filme é com a pergunta
    collect { MATCH (node)-[:IN_GENRE]->(g) RETURN g.name } as genres, // Navega até Gêneros
    collect { MATCH (node)<-[:ACTED_IN]->(a) RETURN a.name } as actors, // Navega até Atores
    avg(r.rating) as userRating             // Calcula a média das notas no banco
ORDER BY userRating DESC                    // Garante que os melhores venham primeiro
"""

# =================================================================
# FASE 5: CONSTRUÇÃO DO RETRIEVER (O BUSCADOR)
# =================================================================

# O Retriever é um objeto que "sabe onde procurar".
# Ele combina: Conexão (driver) + Onde procurar (index) + Como expandir (query) + Tradutor (embedder).
retriever = VectorCypherRetriever(
    driver,
    index_name="moviePlots", # Nome do índice criado no Neo4j (essencial!)
    retrieval_query=retrieval_query,
    embedder=embedder,
)

# =================================================================
# FASE 6: O LLM E O PIPELINE RAG (A LINHA DE MONTAGEM)
# =================================================================

# Instanciamos o modelo gpt-4o. Ele será o encarregado de ler o resultado 
# do banco e escrever a resposta.
llm = OpenAILLM(model_name="gpt-4o")

# 'GraphRAG' é a classe que orquestra tudo. Ela é o "Cérebro do Pipeline".
# Quando você cria este objeto, você está definindo o fluxo: 
# Entrada -> Busca no Grafo -> Contexto -> LLM -> Resposta.
rag = GraphRAG(retriever=retriever, llm=llm)

# =================================================================
# FASE 7: EXECUÇÃO E RESULTADO
# =================================================================

query_text = "Find the highest rated action movie about travelling to other planets"

# .search(): Este é o método principal. Ele dispara internamente:
# 1. Transformação da query em vetor.
# 2. Busca no Neo4j.
# 3. Execução da retrieval_query.
# 4. Envio do contexto para o LLM.
response = rag.search(
    query_text=query_text, 
    retriever_config={"top_k": 5}, # Pede os 5 melhores resultados
    return_context=True            # Pede para o sistema mostrar o que ele achou no banco
)

print(response.answer) # Resposta formatada para o usuário
print("CONTEXT:", response.retriever_result.items) # Dados brutos retornados pelo Neo4j

driver.close() # Encerra o túnel de conexão com o servidor