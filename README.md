# Verbum Ex Machina

A Retrieval-Augmented Generation (RAG) system for exploring the King James Bible using local AI models. Ask questions about Scripture and receive contextual answers with verse references.

## Features

- **Conversational Bible Search**: Ask questions in natural language and get relevant biblical answers
- **Query Rewriting**: Intelligent analysis of conversational context to reformulate queries for better retrieval
- **Semantic Search**: Vector-based search through all ~31,102 verses of the King James Bible
- **Context-Aware Embeddings**: Each verse is embedded with surrounding context for better semantic understanding
- **Multiple Conversations**: Save, rename, and delete conversation histories
- **Markdown Support**: Rich text formatting in responses with lists, bold, italic, and more
- **Local AI Models**: Runs entirely locally using Ollama (no API keys or cloud dependencies)
- **Modern UI**: Clean, responsive chat interface with conversation management

## Architecture

- **Frontend**: Vanilla JavaScript with Markdown rendering (marked.js)
- **Backend**: FastAPI (Python)
- **Vector Database**: ChromaDB for semantic search
- **LLM**: Ollama (default: llama3.2:3b)
- **Embeddings**: Ollama (default: nomic-embed-text)
- **Conversation Storage**: SQLite
- **Deployment**: Docker Compose

### How It Works

1. **Indexing**: Bible verses are loaded and embedded with ±N surrounding verses for context
2. **Query Analysis**: User queries are analyzed to determine if retrieval is needed and rewritten for clarity
3. **Semantic Search**: ChromaDB retrieves the most relevant verses based on vector similarity
4. **Answer Generation**: LLM generates contextual answers using retrieved verses and conversation history

## Prerequisites

- Docker and Docker Compose
- Ollama installed and running on the host machine
- At least 4GB RAM for LLM inference
- ~2GB disk space for embeddings and models

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd verbum-ex-machina
```

### 2. Configure Ollama

Ollama must be accessible from Docker containers. Edit the Ollama service configuration:

**For systemd (Linux)**:
```bash
sudo systemctl edit ollama
```

Add:
```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

Then restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

**Or run manually**:
```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

**Verify Ollama is accessible**:
```bash
# Should show: 0.0.0.0:11434 (not 127.0.0.1:11434)
ss -tlnp | grep 11434
```

### 3. Download Ollama Models

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### 4. Configure Environment

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` to customize settings (optional). Key settings:
- `LLM_MODEL=llama3.2:3b` (must include the tag!)
- `EMBEDDING_MODEL=nomic-embed-text`
- `OLLAMA_BASE_URL=http://host.docker.internal:11434`

### 5. Start the Application

```bash
docker compose up -d
```

The first startup will take 10-30 minutes to index all Bible verses.

### 6. Monitor Indexing Progress

```bash
docker logs -f verbum-app
```

Wait for the message: "ChromaDB collection initialized successfully"

### 7. Access the Application

Open your browser to: http://localhost:8000

## Configuration

All configuration is managed via the `.env` file:

### LLM Settings
- `LLM_MODEL`: Model for answer generation (default: `llama3.2:3b`) - **Must include tag!**
- `EMBEDDING_MODEL`: Model for embeddings (default: `nomic-embed-text`)
- `LLM_TEMPERATURE`: Response randomness 0-1 (default: `0.7`)
- `LLM_MAX_TOKENS`: Maximum response length (default: `500`)
- `QUERY_REWRITE_TEMPERATURE`: Temperature for query analysis (default: `0.3`)

### RAG Settings
- `CONTEXT_WINDOW_SIZE`: Number of verses before/after for context (default: `2`)
- `TOP_K_RESULTS`: Number of verses to retrieve per query (default: `5`)
- `QUERY_REWRITE_ENABLED`: Enable intelligent query rewriting (default: `true`)
- `QUERY_CONTEXT_MESSAGES`: Number of previous messages for context (default: `5`)

### Service Settings
- `OLLAMA_BASE_URL`: Ollama API endpoint (default: `http://host.docker.internal:11434`)
- `CHROMA_HOST`: ChromaDB hostname (default: `chromadb`)
- `CHROMA_PORT`: ChromaDB port (default: `8001`)
- `API_HOST`: FastAPI bind address (default: `0.0.0.0`)
- `API_PORT`: FastAPI port (default: `8000`)

### Data Paths
- `BIBLE_JSON_PATH`: Path to Bible JSON file (default: `/assets/kjv.json`)
- `SQLITE_DB_PATH`: Conversation database path (default: `/data/conversations.db`)
- `CHROMA_PERSIST_DIR`: ChromaDB storage path (default: `/data/chroma`)

## Usage

### Asking Questions

Simply type your question in the chat interface:
- "What does the Bible say about love?"
- "Tell me about the Garden of Eden"
- "What are the Ten Commandments?"

### Managing Conversations

- **New Chat**: Click the "+ New Chat" button
- **Rename**: Hover over a conversation and click the pencil icon (✏️)
- **Delete**: Hover over a conversation and click the trash icon (🗑️)
- **Switch Conversations**: Click any conversation in the sidebar to load it

### Query Rewriting

The system automatically analyzes your queries and rewrites them for better search results. For example:
- Follow-up: "Can you explain that more?" → "Explain [previous topic] in more detail"
- Contextual: "What else does it say?" → "What else does the Bible say about [previous topic]?"

## Development

### Project Structure

```
verbum-ex-machina/
├── app/
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic data models
│   ├── database.py          # SQLite conversation management
│   ├── rag.py               # RAG pipeline implementation
│   └── static/
│       ├── index.html       # Chat UI
│       ├── style.css        # Styles
│       └── app.js           # Frontend logic
├── assets/
│   └── kjv.json             # King James Bible data
├── docker-compose.yml       # Service orchestration
├── Dockerfile               # Application container
├── requirements.txt         # Python dependencies
├── .env                     # Configuration
└── README.md
```

### Development Mode

The `app` directory is mounted as a volume, so code changes are reflected immediately:

**Backend changes** (Python files):
```bash
# Edit files in app/
docker compose restart app
```

**Frontend changes** (HTML/CSS/JS):
```bash
# Edit files in app/static/
# Just refresh your browser - no restart needed!
```

### Logs

View application logs:
```bash
docker logs -f verbum-app
```

View ChromaDB logs:
```bash
docker logs -f verbum-chromadb
```

### Rebuilding

Only needed if you change `requirements.txt` or `Dockerfile`:
```bash
docker compose up -d --build app
```

## Troubleshooting

### Connection Refused to Ollama

**Problem**: `Error generating embedding: [Errno 111] Connection refused`

**Solution**: Ensure Ollama is listening on `0.0.0.0:11434`, not just `127.0.0.1:11434`:
```bash
# Check what Ollama is bound to
ss -tlnp | grep 11434

# Should show: 0.0.0.0:11434
# If it shows: 127.0.0.1:11434, reconfigure as shown in Installation step 2
```

### Empty Collection After Startup

**Problem**: ChromaDB collection exists but has 0 verses

**Solution**: Delete the collection and restart:
```bash
curl -X DELETE http://localhost:8001/api/v1/collections/kjv_bible
docker compose restart app
```

### Model Not Found

**Problem**: `model "llama3.2" not found, try pulling it first`

**Solution**: The model name must include the tag:
```bash
# Check installed models
ollama list

# Update .env with the exact model name including tag
LLM_MODEL=llama3.2:3b

# Restart the app
docker compose restart app
```

### NumPy 2.0 Compatibility Error

**Problem**: `AttributeError: np.float_ was removed in NumPy 2.0`

**Solution**: Already fixed in `requirements.txt` with `numpy<2.0`. If you still see this:
```bash
docker compose down
docker compose up -d --build
```

### Port Already in Use

**Problem**: Port 8000 or 8001 already in use

**Solution**:
- Change ports in `docker-compose.yml`
- Or stop conflicting services

### Slow Response Times

**Problem**: Queries take a long time to respond

**Solutions**:
- Use a smaller/faster model (e.g., `llama3.2:3b` instead of larger variants)
- Reduce `TOP_K_RESULTS` in `.env`
- Reduce `LLM_MAX_TOKENS` for shorter responses
- Ensure your system has adequate RAM (4GB+ recommended)

### Rename/Delete Buttons Not Appearing

**Problem**: Can't see conversation action buttons

**Solution**: Hover over conversations in the sidebar - buttons appear on hover

## API Endpoints

### Chat
- `POST /api/chat` - Send a message and get AI response
  ```json
  {
    "conversation_id": "optional-uuid",
    "message": "What does the Bible say about love?"
  }
  ```

### Conversations
- `GET /api/conversations` - List all conversations
- `GET /api/conversations/{id}` - Get specific conversation
- `PATCH /api/conversations/{id}?title=...` - Rename conversation
- `DELETE /api/conversations/{id}` - Delete conversation

### Health
- `GET /api/health` - Health check endpoint

## Technology Stack

- **FastAPI**: Modern Python web framework
- **ChromaDB 0.4.24**: Open-source vector database
- **Ollama**: Local LLM inference
- **SQLite**: Lightweight conversation storage
- **Docker**: Containerized deployment
- **Marked.js**: Markdown rendering in the UI

## Performance Tips

- **Faster responses**: Use smaller models like `llama3.2:3b` or `phi3`
- **Better accuracy**: Use larger models like `llama3.2:70b`
- **More context**: Increase `CONTEXT_WINDOW_SIZE` (default: 2)
- **More verses**: Increase `TOP_K_RESULTS` (default: 5)
- **GPU acceleration**: Configure Ollama to use GPU for faster inference

## Data Privacy

- All data is stored locally
- No external API calls
- Conversation history stored in SQLite
- Vector embeddings stored in ChromaDB
- Bible data is public domain (King James Version)

## Credits

- Bible text: King James Version (Public Domain)
- UI inspired by modern chat interfaces
- Built with open-source technologies

## License

This project is open source. The King James Bible text is in the public domain.

---

**Verbum Ex Machina** - "The Word from the Machine"
