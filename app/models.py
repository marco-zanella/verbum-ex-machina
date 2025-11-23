from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class BibleVerse(BaseModel):
    """Single verse from the Bible"""
    source: str
    book: str
    chapter: str
    verse: str
    content: str


class VerseWithContext(BaseModel):
    """Verse with surrounding context for embedding"""
    book: str
    chapter: str
    verse: str
    content: str  # The actual verse content
    context: str  # The verse with surrounding verses


class RetrievedVerse(BaseModel):
    """Retrieved verse from vector DB with metadata"""
    book: str
    chapter: str
    verse: str
    content: str
    context: str
    score: float  # Similarity score


class QueryAnalysis(BaseModel):
    """Result of query analysis"""
    needs_retrieval: bool
    rewritten_query: Optional[str] = None
    reasoning: Optional[str] = None


class Message(BaseModel):
    """Single message in a conversation"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    retrieved_verses: Optional[List[RetrievedVerse]] = None


class Conversation(BaseModel):
    """Full conversation thread"""
    conversation_id: str
    title: Optional[str] = None
    messages: List[Message] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    """Request to send a message"""
    conversation_id: Optional[str] = None  # If None, create new conversation
    message: str


class ChatResponse(BaseModel):
    """Response from the chat"""
    conversation_id: str
    message: Message
    retrieved_verses: Optional[List[RetrievedVerse]] = None


class ConversationListItem(BaseModel):
    """Summary of a conversation for listing"""
    conversation_id: str
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    message_count: int


class ConversationListResponse(BaseModel):
    """List of conversations"""
    conversations: List[ConversationListItem]
