import os
 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
 
load_dotenv()

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")

def key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY in your environment or .env file.")
    return key

def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=CHAT_MODEL,
        api_key=key(),
        base_url=GEMINI_BASE_URL,
        temperature=temperature,
    )

def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=key(),
        base_url=GEMINI_BASE_URL,
        check_embedding_ctx_length=False,  
        chunk_size=100,                   
    )



