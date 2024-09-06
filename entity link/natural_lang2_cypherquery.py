import en_core_web_sm
from groq import Groq
from neo4j import GraphDatabase


# NEO4J_USER = ""
# NEO4J_PASSWORD=""
# NEO4J_URI=""

# Load SpaCy model
nlp = en_core_web_sm.load()

# Initialize Groq client
groq_client = Groq(
    api_key=GROQ_API_KEY
)

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

def generate_cypher_query(question):
    doc = nlp(question)

    # Extract relevant parts of the question
    keywords = [token.lemma_.lower() for token in doc if token.pos_ in ('NOUN', 'PROPN', 'ADJ', 'VERB') and len(token.lemma_) > 2]

    # Remove duplicates while preserving order
    keywords = list(dict.fromkeys(keywords))

    # Generate Cypher query
    node_conditions = " OR ".join([f"toLower(n.text) CONTAINS '{keyword}'" for keyword in keywords])
    relationship_conditions = " OR ".join([f"toLower(r.text) CONTAINS '{keyword}'" for keyword in keywords])

    cypher_query = f"""
    MATCH (n)
    WHERE n.text IS NOT NULL AND ({node_conditions})
    RETURN DISTINCT "node" as entity, n.text AS text
    LIMIT 1000
    UNION ALL
    MATCH ()-[r]-()
    WHERE r.text IS NOT NULL AND ({relationship_conditions})
    RETURN DISTINCT "relationship" as entity, r.text AS text
    LIMIT 200
    """

    return cypher_query

def get_llm_response(question, context):
    prompt = f"""
    Question: {question}

    Context from database:
    {context}
"
    Please provide a concise and informative answer to the question based on the given context.

    Answer:
    """
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=xxx,
        temperature=xx,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Do not alter any answer. \
                You will just process the answer based on the question and give me the answer. \
                If you don't found any answer don't give any answer by yourself."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

def ask_question(question):
    conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    cypher_query = generate_cypher_query(question)
    results = conn.query(cypher_query)

    conn.close()

    if results:
        context = "\n".join([f"{result['entity']}: {result['text']}" for result in results])
        llm_answer = get_llm_response(question, context)
        return llm_answer
    else:
        return "I'm sorry, I couldn't find any relevant information in the database to answer your question."

# Example usage
question = input("What would you like to know about the data in the database? ")
answer = ask_question(question)
print(answer)