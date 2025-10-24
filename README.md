# 🧠 Codebase Assistant (RAG Demo with Gemini)

An intelligent **AI assistant** built in Python that uses **Retrieval-Augmented Generation (RAG)** to understand and answer natural language questions about any Python codebase.

You can point it at a local repository (for example, a **FastAPI project**) — it will analyze and "learn" the code, allowing you to ask questions like:

> “How is user authentication handled?”  
> “Show me the Pydantic model for an Article.”  

---

## 🚀 Tech Stack

- **Google Gemini** → For powerful Large Language Model (LLM) and embedding generation.  
- **LangChain** → Framework to chain RAG components together.  
- **FAISS** → Fast, local, in-memory vector database for similarity search.

---

## ⚙️ How It Works (RAG Pipeline)

The assistant follows a classic **Retrieval-Augmented Generation** pipeline:

1. **Load** → Scans your repository and loads all `.py` files.  
2. **Split** → Intelligently splits code into smaller, context-aware chunks.  
3. **Embed** → Converts each chunk into a vector using **Google’s `embedding-001`** model.  
4. **Store** → Saves all vectors into a **local FAISS** vector database.  
5. **Retrieve** → When you ask a question, it finds the top **5 most relevant** code chunks.  
6. **Augment** → Adds these retrieved chunks to your original question as context.  
7. **Generate** → Sends the augmented prompt to **Gemini**, which produces an accurate answer based on the provided context.

---

## 🧩 Example Workflow

1. You ask:  
   > “How is database dependency injection handled?”

2. The assistant retrieves the top 5 related code snippets from your repository.

3. It combines your question + retrieved code snippets into a single prompt.

4. **Gemini** generates a precise, context-aware answer — **grounded only in your codebase**.

---

##  Setup and Installation

###  Prerequisites

Make sure you have:

- **Python 3.9+**
- **Git**
- A **Google Gemini API key**

---

###  Clone the Repository

```
git clone https://github.com/supreetbhat/codebase_assistant.git
cd codebase_assistant
```
 
## ⚙️ Setup and Usage

### 🧱 Set Up a Virtual Environment
It’s strongly recommended to use a virtual environment.

**For macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```
**For Windows:** 
```
python -m venv venv
.\venv\Scripts\activate
```
## Install Dependencies
Install all required libraries from `requirements.txt`
```
pip install -r requirements.txt
```
## Set Up Your API Key
You’ll need a Google Gemini API key.
Get it from Google AI Studio.
Then create a `.env` file:
```
cp .env.example .env
```
Open `.env` and add your key:
```
GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```
💡 Note: You must enable billing on your Google Cloud project to use the embedding API.

## 🚀 How to Run
Run the assistant and pass your target codebase path as an argument:
```
python code_assistant.py /path/to/your/codebase
```
Example:
```
python code_assistant.py ../my-fastapi-project
```
The script will first index the codebase (this may take a minute).
Once you see:
```
✅ Codebase Assistant is ready!
```
You can start asking natural language questions.
Type `exit` to quit.

## 💬 Example Questions to Ask
“What does the `/api/users/login` endpoint do?”

“How do I create a new article?”

“Show me the Pydantic model for a User.”

“How is database dependency injection handled?”

“What fields are required to create a new user?”

“Explain the logic for following another user.”
