import requests
import chromadb
import json
from model_apis import get_model_response
from config import DEFAULT_MODEL, MODELS

def ollama_query(prompt: str) -> str:
    print("Sending query to Ollama...")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:1b",
                "prompt": prompt,
                "stream": False 
            },
        )
        
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            print(f"Error: Received status code {response.status_code}")
            return f"Error: {response.text}"
            
    except requests.exceptions.RequestException as e:
        print(f"Error querying Ollama: {str(e)}")
        return f"Connection error: {str(e)}"
    except json.JSONDecodeError as e:
        print(f"Error parsing Ollama response: {str(e)}")
        return "Error: Unable to parse Ollama's response"

def nomic_embed(text: str) -> list[float]:
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    return response.json()["embedding"]

def query_with_chroma_and_ollama(user_query: str, model_provider: str = DEFAULT_MODEL) -> str:
    print(f"\n=== PDF Q&A System (Using {model_provider}) ===")
    try:
        print("1. Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path="./db")
        collection = client.get_collection(name="my_collection")
        
        print("2. Generating embedding for your query...")
        query_vector = nomic_embed(user_query)
        
        print("3. Searching relevant documents...")
        results = collection.query(query_embeddings=query_vector, n_results=2, include=["documents"])
        
        # Debug: Print if we found any documents
        if not results["documents"] or not results["documents"][0]:
            return "No relevant documents found in the database. Please check if documents were properly indexed."
        
        print("4. Preparing context from search results...")
        context = ""
        for doc_set in results["documents"]:
            for doc in doc_set:
                context += doc + "\n"
        
        print(f"Context length: {len(context)} characters")
        
        if len(context.strip()) == 0:
            return "Error: No context was found to answer the question."
        
        print(f"5. Generating answer using {model_provider.upper()}...")
        combined_prompt = (
            "Your name is AIVO and AIVO means advanced intelligent virtual orator."
            "You are a knowledgeable professor for 3rd-year students at KTU University."
            "Provide thorough and understandable answers to their queries."
            "Use the provided context to answer the questions,"
            "and if you cannot find the answer in the context, find the answer and provide it."
            "Dont hallucinate anything \n\n"
            f"Context:\n{context}\n\n"
            f"Question: {user_query}\n\n"
            "Answer: "
        )
        
        
        print(f"Using model provider: {model_provider}")
        print(f"Total prompt length: {len(combined_prompt)} characters")
        
        response = get_model_response(combined_prompt, model_provider)
        
        if not response or len(response.strip()) == 0:
            return "Error: The model provided an empty response. Please try again or choose a different model."
            
        print("\nAnswer:")
        print("---------------")
        return response
        
    except Exception as e:
        print(f"Debug - Full error: {str(e)}")  # Add full error message
        return f"An error occurred: {str(e)}"

# Add debug option to main
if __name__ == "__main__":
    print("Available models:", ", ".join(MODELS.keys()))
    model_choice = input("Choose a model provider (or press Enter for default): ").lower() or DEFAULT_MODEL
    debug_mode = input("Enable debug mode? (y/n): ").lower() == 'y'
    
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        answer = query_with_chroma_and_ollama(query, model_choice)
        if debug_mode:
            print("\nDebug Information:")
            print(f"Query: {query}")
            print(f"Model: {model_choice}")
            print(f"Answer length: {len(answer)}")
        print(answer)