from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
from typing import Optional
import logging
import os

from .database import ConversationDatabase
from .rag import BibleRAG
from .models import (
    ChatRequest,
    ChatResponse,
    ConversationListResponse,
    Conversation,
    Message
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings from environment variables"""
    # Ollama
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    LLM_MODEL: str = "llama3.2"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 500
    QUERY_REWRITE_TEMPERATURE: float = 0.3

    # RAG
    CONTEXT_WINDOW_SIZE: int = 2
    TOP_K_RESULTS: int = 5

    # Query Rewriting
    QUERY_REWRITE_ENABLED: bool = True
    QUERY_CONTEXT_MESSAGES: int = 5

    # ChromaDB
    CHROMA_HOST: str = "chromadb"
    CHROMA_PORT: int = 8001
    CHROMA_PERSIST_DIR: str = "/data/chroma"
    CHROMA_COLLECTION: str = "kjv_bible"

    # SQLite
    SQLITE_DB_PATH: str = "/data/conversations.db"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: str = "*"

    # Logging
    LOG_LEVEL: str = "INFO"

    # Bible Data
    BIBLE_JSON_PATH: str = "/assets/kjv.json"

    class Config:
        env_file = ".env"


# Initialize settings
settings = Settings()

# Configure logging level
logging.getLogger().setLevel(settings.LOG_LEVEL)

# Initialize FastAPI app
app = FastAPI(
    title="Verbum Ex Machina",
    description="RAG system for the King James Bible",
    version="1.0.0"
)

# Configure CORS
origins = settings.CORS_ORIGINS.split(",") if settings.CORS_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
db: Optional[ConversationDatabase] = None
rag: Optional[BibleRAG] = None


@app.on_event("startup")
async def startup_event():
    """Initialize database and RAG system on startup"""
    global db, rag

    logger.info("Starting Verbum Ex Machina...")

    # Initialize database
    db = ConversationDatabase(settings.SQLITE_DB_PATH)
    await db.init_db()

    # Initialize RAG system
    rag = BibleRAG(
        ollama_base_url=settings.OLLAMA_BASE_URL,
        llm_model=settings.LLM_MODEL,
        embedding_model=settings.EMBEDDING_MODEL,
        chroma_host=settings.CHROMA_HOST,
        chroma_port=settings.CHROMA_PORT,
        chroma_collection=settings.CHROMA_COLLECTION,
        context_window_size=settings.CONTEXT_WINDOW_SIZE,
        top_k_results=settings.TOP_K_RESULTS,
        llm_temperature=settings.LLM_TEMPERATURE,
        llm_max_tokens=settings.LLM_MAX_TOKENS,
        query_rewrite_temperature=settings.QUERY_REWRITE_TEMPERATURE,
        query_context_messages=settings.QUERY_CONTEXT_MESSAGES,
        query_rewrite_enabled=settings.QUERY_REWRITE_ENABLED,
    )

    # Check if collection exists, if not initialize it
    collection_exists = rag.get_or_create_collection()

    if not collection_exists:
        logger.info("ChromaDB collection not found. Initializing...")
        if not os.path.exists(settings.BIBLE_JSON_PATH):
            logger.error(f"Bible JSON file not found at {settings.BIBLE_JSON_PATH}")
            raise FileNotFoundError(f"Bible JSON file not found at {settings.BIBLE_JSON_PATH}")

        verses = rag.load_bible(settings.BIBLE_JSON_PATH)
        verses_with_context = rag.create_verse_contexts(verses)
        rag.initialize_collection(verses_with_context)
        logger.info("ChromaDB collection initialized successfully")
    else:
        logger.info("Using existing ChromaDB collection")

    logger.info("Verbum Ex Machina started successfully!")


@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    return FileResponse("app/static/index.html")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message and get a response"""
    try:
        # Get or create conversation
        conversation_id = request.conversation_id
        if not conversation_id:
            conversation_id = await db.create_conversation()
            logger.info(f"Created new conversation: {conversation_id}")

        # Get recent messages for context
        recent_messages = await db.get_recent_messages(
            conversation_id,
            limit=settings.QUERY_CONTEXT_MESSAGES
        )

        # Add user message to database
        await db.add_message(
            conversation_id=conversation_id,
            role="user",
            content=request.message
        )

        # Analyze query
        query_analysis = rag.analyze_query(request.message, recent_messages)
        logger.info(f"Query analysis: {query_analysis}")

        # Retrieve verses if needed
        retrieved_verses = None
        if query_analysis.needs_retrieval and query_analysis.rewritten_query:
            retrieved_verses = rag.retrieve_verses(query_analysis.rewritten_query)

        # Generate answer
        answer = rag.generate_answer(
            query=request.message,
            retrieved_verses=retrieved_verses,
            recent_messages=recent_messages
        )

        # Save assistant message
        assistant_message = await db.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
            retrieved_verses=retrieved_verses
        )

        return ChatResponse(
            conversation_id=conversation_id,
            message=assistant_message,
            retrieved_verses=retrieved_verses
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations", response_model=ConversationListResponse)
async def list_conversations():
    """List all conversations"""
    try:
        conversations = await db.list_conversations()
        return ConversationListResponse(conversations=conversations)
    except Exception as e:
        logger.error(f"Error listing conversations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """Get a specific conversation"""
    try:
        conversation = await db.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    try:
        await db.delete_conversation(conversation_id)
        return {"status": "success", "message": "Conversation deleted"}
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/api/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, title: str = Query(...)):
    """Update conversation title"""
    try:
        await db.update_conversation_title(conversation_id, title)
        return {"status": "success", "message": "Conversation updated"}
    except Exception as e:
        logger.error(f"Error updating conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database": "connected" if db else "not initialized",
        "rag": "ready" if rag and rag.collection else "not initialized"
    }


# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
