import os
import argparse
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
    # This requires your GOOGLE_API_KEY to be set
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except Exception as e:
        print(f"Error initializing embeddings. Is your GOOGLE_API_KEY set? Error: {e}")
        return None

    # 4. Create and return the FAISS vector store
    print("Creating vector store... This may take a moment.")
    vectorstore = FAISS.from_documents(splits, embeddings)
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
    repo_path = "/Users/supreetbhat/Github/codebase_assistant/fastapi_realworld"
    retriever = create_vector_db(repo_path)
    if retriever is None:
        return

    # --- Part 2: Querying (The RAG Chain) ---

    # 1. Define the Prompt Template
    # This tells the LLM how to behave.
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
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    # 3. Helper function to format the retrieved docs
    def format_docs(docs):
        """Converts the retrieved code chunks into a single string."""
        return "\n\n---\n\n".join(
            f"File: {doc.metadata.get('source', 'N/A')}\n\n{doc.page_content}"
            for doc in docs
        )

    # 4. Build the RAG Chain
    # This is the magic of LangChain. We "pipe" the steps together.
    rag_chain = (
        # This part runs in parallel:
        # 1. The user's `question` is passed through.
        # 2. The `question` is also sent to the `retriever` to find code,
        #    which is then formatted by `format_docs`.
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        # The output (context and question) is fed into our prompt
        | prompt
        # The formatted prompt is fed into the LLM
        | llm
        # The LLM's output is parsed into a simple string
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
