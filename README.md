# AI Python Agent (IGRIS)

A powerful personal AI assistant with full control over a Windows PC. Connects to a backend server and uses Groq LLM for natural language processing.

## Features  

- **Natural Language Commands** - Open apps, run system commands, manage files
- **Conversation Memory** - Learns from past interactions over time
- **User Profile** - Tracks preferences and frequently used commands
- **Screen Reading** - Vision AI analyzes what's on screen
- **Google Integration** - Gmail and Google Drive access
- **Camera Support** - Photo capture, video recording, live streaming
- **YouTube Integration** - Search and play videos directly
- **File Management** - Read, write, edit files with AI assistance

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and configure:
   ```
   BACKEND_URL=http://localhost:3001
   AGENT_KEY=your-secret-key
   GROQ_API_KEY=your-groq-api-key
   ```
4. Run the agent:
   ```
   python agent.py
   ```

Or use the batch file:
```
start-agent.bat
```

## Supported Commands

### Apps
- Open Chrome, Firefox, Edge, Brave, VS Code, Notepad, Calculator, etc.
- Microsoft Office: Word, Excel, PowerPoint, Outlook
- Communication: Teams, Slack, Discord, WhatsApp, Telegram, Zoom, Gmail

### System
- Lock PC, shutdown, restart, sleep
- Volume control (mute, up, down)
- System info, IP address, WiFi info, disk space
- List processes, running apps

### Files
- Open folders: Downloads, Documents, Desktop
- Browse folders, read/write files
- Search files by name

### Media
- Play YouTube videos
- Open Spotify, Netflix, YouTube

### Network
- Search Google
- Flush DNS, ping
- Open any URL

## Architecture

- `agent.py` - Main agent with all handlers
- `.env` - Configuration (create from `.env.example`)
- `.agent_data/` - Conversation history and user profile (auto-created)

## Requirements

- Python 3.8+
- Windows 10/11
- Groq API key (free at groq.com)
- Backend server running at configured URL
