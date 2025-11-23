import json
import logging
from typing import List, Dict, Optional, Tuple
import ollama
import chromadb
from chromadb.config import Settings
from .models import BibleVerse, VerseWithContext, RetrievedVerse, QueryAnalysis, Message

logger = logging.getLogger(__name__)


class BibleRAG:
    def __init__(
        self,
        ollama_base_url: str,
        llm_model: str,
        embedding_model: str,
        chroma_host: str,
        chroma_port: int,
        chroma_collection: str,
        context_window_size: int,
        top_k_results: int,
        llm_temperature: float,
        llm_max_tokens: int,
        query_rewrite_temperature: float,
        query_context_messages: int,
        query_rewrite_enabled: bool,
    ):
        self.ollama_client = ollama.Client(host=ollama_base_url)
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.context_window_size = context_window_size
        self.top_k_results = top_k_results
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.query_rewrite_temperature = query_rewrite_temperature
        self.query_context_messages = query_context_messages
        self.query_rewrite_enabled = query_rewrite_enabled

        # Initialize ChromaDB client
        self.chroma_client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = chroma_collection
        self.collection = None

    def load_bible(self, bible_json_path: str) -> List[BibleVerse]:
        """Load Bible verses from JSON file"""
        logger.info(f"Loading Bible from {bible_json_path}")
        with open(bible_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        verses = [BibleVerse(**verse) for verse in data]
        logger.info(f"Loaded {len(verses)} verses")
        return verses

    def create_verse_contexts(self, verses: List[BibleVerse]) -> List[VerseWithContext]:
        """Create context windows for each verse"""
        logger.info("Creating context windows for verses")

        # Group verses by book and chapter
        grouped: Dict[Tuple[str, str], List[BibleVerse]] = {}
        for verse in verses:
            key = (verse.book, verse.chapter)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(verse)

        # Sort each group by verse number
        for key in grouped:
            grouped[key].sort(key=lambda v: int(v.verse))

        # Create contexts
        verses_with_context = []
        for (book, chapter), chapter_verses in grouped.items():
            for i, verse in enumerate(chapter_verses):
                # Get surrounding verses within the window
                start_idx = max(0, i - self.context_window_size)
                end_idx = min(len(chapter_verses), i + self.context_window_size + 1)

                context_verses = chapter_verses[start_idx:end_idx]
                context = " ".join([v.content for v in context_verses])

                verses_with_context.append(VerseWithContext(
                    book=verse.book,
                    chapter=verse.chapter,
                    verse=verse.verse,
                    content=verse.content,
                    context=context
                ))

        logger.info(f"Created {len(verses_with_context)} verses with context")
        return verses_with_context

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using Ollama"""
        try:
            response = self.ollama_client.embeddings(
                model=self.embedding_model,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def initialize_collection(self, verses_with_context: List[VerseWithContext]):
        """Initialize ChromaDB collection with Bible verses"""
        logger.info(f"Initializing ChromaDB collection: {self.collection_name}")

        # Delete existing collection if it exists
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted existing collection: {self.collection_name}")
        except Exception:
            pass

        # Create new collection
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"description": "King James Bible verses with context"}
        )

        # Prepare data for insertion
        logger.info("Generating embeddings for verses...")
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for i, verse in enumerate(verses_with_context):
            # Generate unique ID
            verse_id = f"{verse.book}_{verse.chapter}_{verse.verse}"
            ids.append(verse_id)

            # Embed the context (not just the verse)
            embedding = self.embed_text(verse.context)
            embeddings.append(embedding)

            # Store the context as document
            documents.append(verse.context)

            # Store metadata
            metadatas.append({
                "book": verse.book,
                "chapter": verse.chapter,
                "verse": verse.verse,
                "content": verse.content,
                "context": verse.context
            })

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(verses_with_context)} verses")

        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            logger.info(f"Added batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1}")

        logger.info("ChromaDB collection initialized successfully")

    def get_or_create_collection(self):
        """Get existing collection or indicate it needs initialization"""
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            count = self.collection.count()
            if count > 0:
                logger.info(f"Retrieved existing collection: {self.collection_name} with {count} verses")
                return True
            else:
                logger.warning(f"Collection {self.collection_name} exists but is empty, needs re-indexing")
                return False
        except Exception:
            logger.warning(f"Collection {self.collection_name} does not exist")
            return False

    def retrieve_verses(self, query: str) -> List[RetrievedVerse]:
        """Retrieve relevant verses for a query"""
        if not self.collection:
            raise RuntimeError("Collection not initialized. Call get_or_create_collection first.")

        logger.info(f"Retrieving verses for query: {query}")

        # Generate query embedding
        query_embedding = self.embed_text(query)

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k_results
        )

        # Parse results
        retrieved_verses = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]

                # Convert distance to similarity score (1 - normalized distance)
                score = 1 / (1 + distance)

                retrieved_verses.append(RetrievedVerse(
                    book=metadata['book'],
                    chapter=metadata['chapter'],
                    verse=metadata['verse'],
                    content=metadata['content'],
                    context=metadata['context'],
                    score=score
                ))

        logger.info(f"Retrieved {len(retrieved_verses)} verses")
        return retrieved_verses

    def analyze_query(self, query: str, recent_messages: List[Message]) -> QueryAnalysis:
        """Analyze if query needs retrieval and rewrite if necessary"""
        if not self.query_rewrite_enabled:
            return QueryAnalysis(
                needs_retrieval=True,
                rewritten_query=query,
                reasoning="Query rewriting disabled"
            )

        # Build conversation context
        context_messages = []
        for msg in recent_messages[-self.query_context_messages:]:
            context_messages.append(f"{msg.role}: {msg.content}")

        context_str = "\n".join(context_messages) if context_messages else "No previous context"

        # Create prompt for query analysis
        system_prompt = """You are a query analysis assistant for a Bible Q&A system.

Your task is to analyze the user's query and determine:
1. Whether the query requires searching the Bible (needs_retrieval: true/false)
2. If yes, rewrite the query to be a standalone, searchable question

Examples:
- "What is the Garden of Eden?" → needs_retrieval: true, rewritten: "What is the Garden of Eden?"
- "Can you explain that more?" → needs_retrieval: true, rewritten: "Explain the Garden of Eden in more detail"
- "Thank you!" → needs_retrieval: false
- "What else does it say?" → needs_retrieval: true, rewritten: "What else does the Bible say about [previous topic]?"

Respond in JSON format:
{
  "needs_retrieval": true/false,
  "rewritten_query": "the standalone query if need_retrieval is true" or null,
  "reasoning": "brief explanation"
}"""

        user_prompt = f"""Conversation context:
{context_str}

User's current query: {query}

Analyze this query."""

        try:
            response = self.ollama_client.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": self.query_rewrite_temperature,
                    "num_predict": 200
                },
                format="json"
            )

            # Parse JSON response
            result = json.loads(response['message']['content'])
            return QueryAnalysis(**result)

        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            # Fallback: assume retrieval needed
            return QueryAnalysis(
                needs_retrieval=True,
                rewritten_query=query,
                reasoning=f"Error in analysis: {str(e)}"
            )

    def generate_answer(
        self,
        query: str,
        retrieved_verses: Optional[List[RetrievedVerse]],
        recent_messages: List[Message]
    ) -> str:
        """Generate answer using LLM"""
        # Build conversation history
        conversation = []
        for msg in recent_messages:
            conversation.append({
                "role": msg.role,
                "content": msg.content
            })

        # Build system prompt
        if retrieved_verses:
            verses_context = "\n\n".join([
                f"{v.book.capitalize()} {v.chapter}:{v.verse} - {v.content}"
                for v in retrieved_verses
            ])

            system_prompt = f"""You are a knowledgeable Bible assistant helping users understand Scripture.

The following verses have been retrieved from the King James Bible as relevant to the user's question:

{verses_context}

Guidelines:
- Answer the user's question based on the retrieved verses
- Always cite specific verse references (e.g., Genesis 1:1, John 3:16)
- Be accurate and faithful to the text
- If the verses don't fully answer the question, acknowledge this
- Be concise but thorough
- Maintain a respectful, scholarly tone"""
        else:
            system_prompt = """You are a knowledgeable Bible assistant helping users understand Scripture.

No specific verses were retrieved for this query. Answer based on the conversation context."""

        # Add system prompt and user query
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation)
        messages.append({"role": "user", "content": query})

        try:
            response = self.ollama_client.chat(
                model=self.llm_model,
                messages=messages,
                options={
                    "temperature": self.llm_temperature,
                    "num_predict": self.llm_max_tokens
                }
            )

            answer = response['message']['content']
            logger.info(f"Generated answer: {answer[:100]}...")
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
