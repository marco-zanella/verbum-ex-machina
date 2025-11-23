import aiosqlite
import json
import uuid
from datetime import datetime
from typing import List, Optional
from .models import Conversation, Message, ConversationListItem, RetrievedVerse
import logging

logger = logging.getLogger(__name__)


class ConversationDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path

    async def init_db(self):
        """Initialize database tables"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    retrieved_verses TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                )
            """)

            await db.commit()
            logger.info("Database initialized successfully")

    async def create_conversation(self, title: Optional[str] = None) -> str:
        """Create a new conversation and return its ID"""
        conversation_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO conversations (conversation_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conversation_id, title, now, now)
            )
            await db.commit()

        logger.info(f"Created conversation {conversation_id}")
        return conversation_id

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        retrieved_verses: Optional[List[RetrievedVerse]] = None
    ) -> Message:
        """Add a message to a conversation"""
        now = datetime.utcnow()

        # Serialize retrieved verses if present
        verses_json = None
        if retrieved_verses:
            verses_json = json.dumps([v.model_dump() for v in retrieved_verses])

        async with aiosqlite.connect(self.db_path) as db:
            # Insert message
            await db.execute(
                "INSERT INTO messages (conversation_id, role, content, timestamp, retrieved_verses) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, role, content, now.isoformat(), verses_json)
            )

            # Update conversation timestamp
            await db.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (now.isoformat(), conversation_id)
            )

            await db.commit()

        return Message(
            role=role,
            content=content,
            timestamp=now,
            retrieved_verses=retrieved_verses
        )

    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a full conversation by ID"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # Get conversation metadata
            async with db.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None

                conv_data = dict(row)

            # Get messages
            messages = []
            async with db.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC",
                (conversation_id,)
            ) as cursor:
                async for row in cursor:
                    msg_data = dict(row)

                    # Deserialize retrieved verses
                    retrieved_verses = None
                    if msg_data['retrieved_verses']:
                        verses_data = json.loads(msg_data['retrieved_verses'])
                        retrieved_verses = [RetrievedVerse(**v) for v in verses_data]

                    messages.append(Message(
                        role=msg_data['role'],
                        content=msg_data['content'],
                        timestamp=datetime.fromisoformat(msg_data['timestamp']),
                        retrieved_verses=retrieved_verses
                    ))

        return Conversation(
            conversation_id=conv_data['conversation_id'],
            title=conv_data['title'],
            messages=messages,
            created_at=datetime.fromisoformat(conv_data['created_at']),
            updated_at=datetime.fromisoformat(conv_data['updated_at'])
        )

    async def get_recent_messages(
        self,
        conversation_id: str,
        limit: int = 10
    ) -> List[Message]:
        """Get the N most recent messages from a conversation"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            messages = []
            async with db.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp DESC LIMIT ?",
                (conversation_id, limit)
            ) as cursor:
                async for row in cursor:
                    msg_data = dict(row)

                    # Deserialize retrieved verses
                    retrieved_verses = None
                    if msg_data['retrieved_verses']:
                        verses_data = json.loads(msg_data['retrieved_verses'])
                        retrieved_verses = [RetrievedVerse(**v) for v in verses_data]

                    messages.append(Message(
                        role=msg_data['role'],
                        content=msg_data['content'],
                        timestamp=datetime.fromisoformat(msg_data['timestamp']),
                        retrieved_verses=retrieved_verses
                    ))

            # Reverse to get chronological order
            return list(reversed(messages))

    async def list_conversations(self, limit: int = 50) -> List[ConversationListItem]:
        """List all conversations, ordered by most recently updated"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            conversations = []
            async with db.execute(
                """
                SELECT
                    c.*,
                    COUNT(m.id) as message_count
                FROM conversations c
                LEFT JOIN messages m ON c.conversation_id = m.conversation_id
                GROUP BY c.conversation_id
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (limit,)
            ) as cursor:
                async for row in cursor:
                    data = dict(row)
                    conversations.append(ConversationListItem(
                        conversation_id=data['conversation_id'],
                        title=data['title'],
                        created_at=datetime.fromisoformat(data['created_at']),
                        updated_at=datetime.fromisoformat(data['updated_at']),
                        message_count=data['message_count']
                    ))

            return conversations

    async def update_conversation_title(self, conversation_id: str, title: str):
        """Update a conversation's title"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE conversations SET title = ? WHERE conversation_id = ?",
                (title, conversation_id)
            )
            await db.commit()

    async def delete_conversation(self, conversation_id: str):
        """Delete a conversation and all its messages"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,)
            )
            await db.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (conversation_id,)
            )
            await db.commit()

        logger.info(f"Deleted conversation {conversation_id}")
