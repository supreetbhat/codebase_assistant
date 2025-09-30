from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()  # loads variables from .env
api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key  # <-- pass API key explicitly
)

response = llm.invoke("Explain Retrieval-Augmented Generation in 2 lines.")
print(response.content)