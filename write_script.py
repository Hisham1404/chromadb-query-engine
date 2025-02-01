import requests
import os
import json
import PyPDF2
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_processed_files():
    try:
        with open('processed_files.json', 'r') as f:
            return set(json.load(f))
    except FileNotFoundError:
        return set()

def save_processed_files(processed):
    with open('processed_files.json', 'w') as f:
        json.dump(list(processed), f)

# Function to convert PDF to text
def pdf_to_text(file_path):
    print(f"Processing PDF: {file_path}")
    pdf_file = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range( len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    pdf_file.close()
    return text

def nomic_embed(text: str) -> list[float]:
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": text
            }
        )
        return response.json()["embedding"]
    except Exception as e:
        print(f"Error in embedding: {str(e)}")
        raise

def process_new_pdfs():
    print("Starting incremental PDF processing...")
    
    # Load list of already processed files
    processed_files = load_processed_files()
    
    # Get list of all PDFs in input directory
    current_pdfs = {f for f in os.listdir('./input') if f.endswith('.pdf')}
    
    # Find new PDFs that haven't been processed
    new_pdfs = current_pdfs - processed_files
    
    if not new_pdfs:
        print("No new PDFs to process!")
        return
    
    print(f"Found {len(new_pdfs)} new PDFs to process")
    
    # Initialize ChromaDB client and get existing collection
    client = chromadb.PersistentClient(path="./db")
    try:
        collection = client.get_collection(name="my_collection")
        print("Connected to existing collection")
    except:
        collection = client.create_collection(name="my_collection")
        print("Created new collection")

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    # Process each new PDF
    for filename in new_pdfs:
        print(f"\nProcessing new file: {filename}")
        try:
            # Convert PDF to text
            text = pdf_to_text(os.path.join('./input', filename))
            
            # Split text into chunks
            chunks = text_splitter.split_text(text)
            print(f"Generated {len(chunks)} chunks from {filename}")
            
            # Prepare batch data
            documents_list = []
            embeddings_list = []
            ids_list = []
            
            print("Creating embeddings...")
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}", end='\r')
                vector = nomic_embed(chunk)
                
                documents_list.append(chunk)
                embeddings_list.append(vector)
                ids_list.append(f"{filename}_{i}")
            
            # Add new data to collection
            collection.add(
                embeddings=embeddings_list,
                documents=documents_list,
                ids=ids_list
            )
            
            # Mark file as processed
            processed_files.add(filename)
            save_processed_files(processed_files)
            
            print(f"\nSuccessfully added {len(chunks)} chunks from {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

    print("\nIncremental processing complete!")
    print(f"Total PDFs processed: {len(new_pdfs)}")

if __name__ == "__main__":
    process_new_pdfs()
