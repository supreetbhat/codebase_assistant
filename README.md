Codebase Assistant (RAG Demo with Gemini)

This project is a Python-based AI assistant that uses Retrieval-Augmented Generation (RAG) to answer questions about any Python codebase.

You can point it at a local repository (e..g., a FastAPI project), and it will "learn" the code. You can then ask it natural language questions like "How is user authentication handled?" or "Show me the Pydantic model for an Article."

This assistant is built using:

Google Gemini: For the powerful Large Language Model (LLM) and embedding generation.

LangChain: As the framework to "chain" all the RAG components together.

FAISS: As a fast, local, in-memory vector database.

How It Works (The RAG Pipeline)

The assistant follows a classic RAG pattern:

Load: Scans your target repository and loads all .py files.

Split: Intelligently splits the code into smaller, context-aware chunks.

Embed: Uses the Google embedding-001 model to convert each code chunk into a vector (a numerical representation).

Store: Stores all these vectors in a local FAISS vector database.

Retrieve: When you ask a question, it embeds your question and uses FAISS to find the top 5 most similar code chunks from the database.

Augment: It "augments" (adds) these 5 code chunks to your original question in a prompt.

Generate: It sends the complete prompt (question + code snippets) to the Gemini model, which generates an answer based only on the provided context.

Setup and Installation

1. Prerequisites

Python 3.9 or higher

Git

2. Clone the Repository

git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY


3. Set Up a Virtual Environment

It's highly recommended to use a virtual environment.

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate


4. Install Dependencies

Install all required libraries from requirements.txt:

pip install -r requirements.txt


5. Set Up Your API Key

You will need a Google Gemini API key.

Get your key from Google AI Studio.

This project uses a .env file to securely load your key. Create a .env file by copying the example:

cp .env.example .env


Open the .env file and paste your API key:

GOOGLE_API_KEY="YOUR_API_KEY_HERE"


You must also enable billing on your Google Cloud project to use the embedding API.

How to Run

Run the code_assistant.py script from your terminal, passing the file path to the codebase you want it to analyze as an argument.

python code_assistant.py /path/to/your/codebase


Example:

python code_assistant.py ../my-fastapi-project


The script will first index the codebase (this may take a minute). Once you see the "✅ Codebase Assistant is ready!" message, you can start asking questions.

Type exit to quit.

Example Questions to Ask

"What does the /api/users/login endpoint do?"

"How do I create a new article?"

"Show me the Pydantic model for a User."

"How is database dependency injection handled?"

"What fields are required to create a new user?"

"Explain the logic for following another user."
