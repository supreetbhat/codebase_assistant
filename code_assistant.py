import os
import argparse
import time  # <-- 1. IMPORTED TIME
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load the .env file (which contains your GOOGLE_API_KEY)
load_dotenv()
print("Loaded environment variables...")

def create_vector_db(repo_path: str):
    """
    Loads Python files, splits them, creates embeddings, 
    and stores them in a FAISS vector database.
    """
    
    # 1. Load all Python files from the repo path
    print(f"Loading code from: {repo_path}")
    loader = DirectoryLoader(
        repo_path,
        glob="**/*.py",  # Look for all .py files recursively
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        recursive=True,
        show_progress=True
    )
    docs = loader.load()
    
    if not docs:
        print(f"No Python files found in {repo_path}. Please check the path.")
        return None

    print(f"Loaded {len(docs)} Python documents.")

    # 2. Split the code into intelligent chunks
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language="python", chunk_size=1000, chunk_overlap=100
    )
    splits = python_splitter.split_documents(docs)
    print(f"Split documents into {len(splits)} code chunks.")

    # 3. Create embeddings using Google's model
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except Exception as e:
        print(f"Error initializing embeddings. Is your GOOGLE_API_KEY set? Error: {e}")
        return None

    # 4. Create and return the FAISS vector store **in batches**
    print("Creating vector store in batches... This may take a moment.")
    
    vectorstore = None
    batch_size = 100  # Process 100 docs at a time (Google's default limit)
    
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{len(splits) // batch_size + 1}...")
        
        if vectorstore is None:
            # Create the store with the first batch
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            # Add subsequent batches to the existing store
            vectorstore.add_documents(batch)
        
        # Wait for a second to avoid hitting per-minute rate limits
        print("Waiting 1 second to respect API rate limits...")
        time.sleep(1)

    print("Vector store created successfully.")
    
    # We return a "retriever" which is an object that can search the database
    return vectorstore.as_retriever(search_kwargs={"k": 5}) # k=5 means "get top 5 results"


def main(repo_path: str):
    """
    Main function to set up the RAG chain and start the chat loop.
    """
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY not found. Please create a .env file and add it.")
        return

    # --- Part 1: Indexing ---
    # The 'repo_path' from the argument will be used.
    retriever = create_vector_db(repo_path)
    if retriever is None:
        return

    # --- Part 2: Querying (The RAG Chain) ---

    # 1. Define the Prompt Template
    template = """
You are an expert FastAPI developer assistant.
Your job is to answer the user's question based *only* on the
following code snippets from their repository:

<context>
{context}
</context>

Question: {question}

If the context doesn't contain the answer, just say:
"I could not find the answer in the provided code snippets."
"""
    prompt = ChatPromptTemplate.from_template(template)

    # 2. Initialize the Gemini LLM
    # 4. CORRECTED THE MODEL NAME
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-09-2025", temperature=0.1)

    # 3. Helper function to format the retrieved docs
    def format_docs(docs):
        """Converts the retrieved code chunks into a single string."""
        return "\n\n---\n\n".join(
            f"File: {doc.metadata.get('source', 'N/A')}\n\n{doc.page_content}"
            for doc in docs
        )

    # 4. Build the RAG Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n✅ Codebase Assistant is ready!")
    print("Ask questions about your FastAPI repository. Type 'exit' to quit.")

    # 5. Start the conversation loop
    while True:
        try:
            question = input("\n> ")
            if question.lower() == 'exit':
                break
            
            print("\nThinking...")
            
            # Get the answer from our RAG chain
            answer = rag_chain.invoke(question)
            
            print("\nAnswer:")
            print(answer)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

    print("\nGoodbye!")


# This part lets us run the script from the command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastAPI Codebase Assistant")
    # It requires one argument: the path to the repo
    parser.add_argument("repo_path", type=str, help="The file path to your FastAPI repository.")
    args = parser.parse_args()
    
    if not os.path.isdir(args.repo_path):
        print(f"Error: Path '{args.repo_path}' is not a valid directory.")
    else:
        main(args.repo_path)