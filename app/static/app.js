// State management
let currentConversationId = null;
let isLoading = false;

// DOM Elements
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const messagesContainer = document.getElementById('messagesContainer');
const conversationsList = document.getElementById('conversationsList');
const newChatBtn = document.getElementById('newChatBtn');
const chatTitle = document.getElementById('chatTitle');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadConversations();
    setupEventListeners();
});

// Event Listeners
function setupEventListeners() {
    // Send button click
    sendBtn.addEventListener('click', sendMessage);

    // Enter to send (Shift+Enter for new line)
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = messageInput.scrollHeight + 'px';
        updateSendButton();
    });

    // New chat button
    newChatBtn.addEventListener('click', startNewChat);
}

// Update send button state
function updateSendButton() {
    const hasText = messageInput.value.trim().length > 0;
    sendBtn.disabled = !hasText || isLoading;
}

// Start new chat
function startNewChat() {
    currentConversationId = null;
    messagesContainer.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">✨</div>
            <h2>Welcome to Verbum Ex Machina</h2>
            <p>Ask me questions about the King James Bible. I'll search through Scripture and provide answers with verse references.</p>
            <div class="example-questions">
                <p class="example-header">Try asking:</p>
                <button class="example-btn" onclick="askExample('What does the Bible say about love?')">
                    "What does the Bible say about love?"
                </button>
                <button class="example-btn" onclick="askExample('Tell me about the Garden of Eden')">
                    "Tell me about the Garden of Eden"
                </button>
                <button class="example-btn" onclick="askExample('What are the Ten Commandments?')">
                    "What are the Ten Commandments?"
                </button>
            </div>
        </div>
    `;
    chatTitle.textContent = 'Ask me anything about the Bible';
    messageInput.value = '';
    messageInput.focus();

    // Remove active class from all conversations
    document.querySelectorAll('.conversation-item').forEach(item => {
        item.classList.remove('active');
    });
}

// Ask example question
function askExample(question) {
    messageInput.value = question;
    updateSendButton();
    sendMessage();
}

// Send message
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isLoading) return;

    // Clear input
    messageInput.value = '';
    messageInput.style.height = 'auto';
    updateSendButton();

    // Remove welcome message if present
    const welcomeMsg = messagesContainer.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }

    // Add user message to UI
    addMessageToUI('user', message);

    // Add loading indicator
    const loadingId = addLoadingMessage();

    isLoading = true;
    updateSendButton();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                conversation_id: currentConversationId,
                message: message
            })
        });

        if (!response.ok) {
            throw new Error('Failed to send message');
        }

        const data = await response.json();

        // Update conversation ID if it's a new conversation
        if (!currentConversationId) {
            currentConversationId = data.conversation_id;
            await loadConversations();
        }

        // Remove loading indicator
        removeLoadingMessage(loadingId);

        // Add assistant response
        addMessageToUI('assistant', data.message.content, data.retrieved_verses);

    } catch (error) {
        console.error('Error sending message:', error);
        removeLoadingMessage(loadingId);
        addMessageToUI('assistant', 'Sorry, I encountered an error. Please try again.');
    } finally {
        isLoading = false;
        updateSendButton();
        messageInput.focus();
    }
}

// Add message to UI
function addMessageToUI(role, content, retrievedVerses = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const header = document.createElement('div');
    header.className = 'message-header';
    header.textContent = role === 'user' ? 'You' : 'Verbum Ex Machina';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    // Render Markdown for assistant messages, plain text for user messages
    if (role === 'assistant' && typeof marked !== 'undefined') {
        contentDiv.innerHTML = marked.parse(content);
    } else {
        // Fallback to plain text with paragraphs
        const paragraphs = content.split('\n').filter(p => p.trim());
        paragraphs.forEach(p => {
            const pElement = document.createElement('p');
            pElement.textContent = p;
            contentDiv.appendChild(pElement);
        });
    }

    messageDiv.appendChild(header);
    messageDiv.appendChild(contentDiv);

    // Add verse references if present
    if (retrievedVerses && retrievedVerses.length > 0) {
        const versesDiv = document.createElement('div');
        versesDiv.className = 'verse-references';

        const title = document.createElement('div');
        title.className = 'verse-references-title';
        title.textContent = '📖 Referenced Verses:';
        versesDiv.appendChild(title);

        retrievedVerses.forEach(verse => {
            const verseItem = document.createElement('div');
            verseItem.className = 'verse-item';

            const ref = document.createElement('span');
            ref.className = 'verse-ref';
            ref.textContent = `${capitalizeFirstLetter(verse.book)} ${verse.chapter}:${verse.verse}`;

            const text = document.createTextNode(` - ${verse.content}`);

            verseItem.appendChild(ref);
            verseItem.appendChild(text);
            versesDiv.appendChild(verseItem);
        });

        messageDiv.appendChild(versesDiv);
    }

    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Add loading message
function addLoadingMessage() {
    const loadingId = `loading-${Date.now()}`;
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant loading';
    messageDiv.id = loadingId;

    const header = document.createElement('div');
    header.className = 'message-header';
    header.textContent = 'Verbum Ex Machina';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = `
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;

    messageDiv.appendChild(header);
    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    return loadingId;
}

// Remove loading message
function removeLoadingMessage(loadingId) {
    const loadingMsg = document.getElementById(loadingId);
    if (loadingMsg) {
        loadingMsg.remove();
    }
}

// Load conversations list
async function loadConversations() {
    try {
        const response = await fetch('/api/conversations');
        if (!response.ok) {
            throw new Error('Failed to load conversations');
        }

        const data = await response.json();
        displayConversations(data.conversations);
    } catch (error) {
        console.error('Error loading conversations:', error);
    }
}

// Display conversations in sidebar
function displayConversations(conversations) {
    conversationsList.innerHTML = '';

    if (conversations.length === 0) {
        conversationsList.innerHTML = `
            <div style="padding: 20px; text-align: center; color: rgba(255, 255, 255, 0.5); font-size: 14px;">
                No conversations yet
            </div>
        `;
        return;
    }

    conversations.forEach(conv => {
        const item = document.createElement('div');
        item.className = 'conversation-item';
        if (conv.conversation_id === currentConversationId) {
            item.classList.add('active');
        }

        const content = document.createElement('div');
        content.className = 'conversation-content';

        const title = document.createElement('div');
        title.className = 'conversation-title';
        title.textContent = conv.title || `Conversation ${conv.message_count} messages`;

        const meta = document.createElement('div');
        meta.className = 'conversation-meta';
        meta.textContent = formatDate(conv.updated_at);

        content.appendChild(title);
        content.appendChild(meta);

        const actions = document.createElement('div');
        actions.className = 'conversation-actions';

        const renameBtn = document.createElement('button');
        renameBtn.className = 'action-btn';
        renameBtn.innerHTML = '✏️';
        renameBtn.title = 'Rename';
        renameBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            renameConversation(conv.conversation_id, conv.title);
        });

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'action-btn delete-btn';
        deleteBtn.innerHTML = '🗑️';
        deleteBtn.title = 'Delete';
        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            deleteConversation(conv.conversation_id);
        });

        actions.appendChild(renameBtn);
        actions.appendChild(deleteBtn);

        item.appendChild(content);
        item.appendChild(actions);

        content.addEventListener('click', () => loadConversation(conv.conversation_id));

        conversationsList.appendChild(item);
    });
}

// Load a specific conversation
async function loadConversation(conversationId) {
    try {
        const response = await fetch(`/api/conversations/${conversationId}`);
        if (!response.ok) {
            throw new Error('Failed to load conversation');
        }

        const conversation = await response.json();
        currentConversationId = conversationId;

        // Clear messages container
        messagesContainer.innerHTML = '';

        // Display all messages
        conversation.messages.forEach(msg => {
            addMessageToUI(msg.role, msg.content, msg.retrieved_verses);
        });

        // Update chat title
        chatTitle.textContent = conversation.title || 'Bible Chat';

        // Update active conversation in sidebar
        document.querySelectorAll('.conversation-item').forEach(item => {
            item.classList.remove('active');
        });
        event.target.closest('.conversation-item')?.classList.add('active');

    } catch (error) {
        console.error('Error loading conversation:', error);
    }
}

// Utility: Format date
function formatDate(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays === 0) {
        return 'Today';
    } else if (diffDays === 1) {
        return 'Yesterday';
    } else if (diffDays < 7) {
        return `${diffDays} days ago`;
    } else {
        return date.toLocaleDateString();
    }
}

// Utility: Capitalize first letter
function capitalizeFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
}

// Rename conversation
async function renameConversation(conversationId, currentTitle) {
    const newTitle = prompt('Enter new conversation title:', currentTitle || '');
    if (!newTitle || newTitle.trim() === '') {
        return;
    }

    try {
        const response = await fetch(`/api/conversations/${conversationId}?title=${encodeURIComponent(newTitle.trim())}`, {
            method: 'PATCH',
        });

        if (!response.ok) {
            throw new Error('Failed to rename conversation');
        }

        await loadConversations();

        // Update chat title if this is the current conversation
        if (conversationId === currentConversationId) {
            chatTitle.textContent = newTitle.trim();
        }
    } catch (error) {
        console.error('Error renaming conversation:', error);
        alert('Failed to rename conversation. Please try again.');
    }
}

// Delete conversation
async function deleteConversation(conversationId) {
    if (!confirm('Are you sure you want to delete this conversation? This action cannot be undone.')) {
        return;
    }

    try {
        const response = await fetch(`/api/conversations/${conversationId}`, {
            method: 'DELETE',
        });

        if (!response.ok) {
            throw new Error('Failed to delete conversation');
        }

        // If deleting current conversation, start a new one
        if (conversationId === currentConversationId) {
            startNewChat();
        }

        await loadConversations();
    } catch (error) {
        console.error('Error deleting conversation:', error);
        alert('Failed to delete conversation. Please try again.');
    }
}
