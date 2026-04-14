import os
import time
import json
import base64
import subprocess
import requests
import re
import threading
import logging
from datetime import datetime
from urllib.parse import quote_plus
from dotenv import load_dotenv
from web_tester import WebTestAgent

# ──────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.agent_data')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, 'agent.log'), encoding='utf-8'),
    ]
)
log = logging.getLogger('jarvis')

load_dotenv()

BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:3001')
AGENT_KEY = os.getenv('AGENT_KEY', 'change-me-secret')
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
GROQ_API_KEY_BACKUP = os.getenv('GROQ_API_KEY1', '')  # backup key for rate limits
HEADERS = {'X-Agent-Key': AGENT_KEY, 'Content-Type': 'application/json'}
POLL_INTERVAL = 3

AGENT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.agent_data')
os.makedirs(AGENT_DATA_DIR, exist_ok=True)

pending_sensitive_action = None

CONFIRM_POSITIVE = re.compile(r'^(?:yes|y|sure|ok|okay|confirm|go ahead|please do it|send it|do it)\b', re.I)
CONFIRM_NEGATIVE = re.compile(r'^(?:no|n|cancel|stop|don\'t|do not|never|not now)\b', re.I)


def is_positive_confirmation(text):
    return bool(CONFIRM_POSITIVE.search(str(text).strip()))


def is_negative_confirmation(text):
    return bool(CONFIRM_NEGATIVE.search(str(text).strip()))


def is_sensitive_action(data):
    if not isinstance(data, dict):
        return False
    action = data.get('action', '')
    if action in ('send_email', 'draft_email'):
        return True
    if action == 'execute':
        cmd = str(data.get('command', '')).lower()
        return bool(re.search(r'\b(del|erase|remove|rmdir|rd)\b', cmd))
    return False


def build_confirmation_text(data):
    action = data.get('action', '')
    if action == 'send_email':
        to = data.get('to', 'the recipient')
        subject = data.get('subject', '')
        return f'I can send an email to {to}{" with subject " + subject if subject else ""}. Please confirm before I proceed.'
    if action == 'draft_email':
        to = data.get('to', 'the recipient')
        return f'I can draft an email to {to}. Please confirm before I proceed.'
    if action == 'execute':
        cmd = data.get('command', '')
        return f'I am ready to run this command: {cmd}. This may be destructive. Please confirm before I proceed.'
    return 'I found a sensitive action. Please confirm before I proceed.'

# ──────────────────────────────────────────────
# Conversation Memory — persists chat history
# ──────────────────────────────────────────────
class ConversationMemory:
    """Stores conversation history with smart summarization. Old messages get
    compressed into summaries so the agent retains long-term context without
    blowing up the token budget."""
    FILE = os.path.join(AGENT_DATA_DIR, 'conversations.json')
    SUMMARY_FILE = os.path.join(AGENT_DATA_DIR, 'conversation_summaries.json')
    MAX_MESSAGES = 50          # keep last N raw messages
    SUMMARY_THRESHOLD = 40     # summarize when we hit this many messages
    SUMMARY_BATCH = 20         # how many old messages to summarize at once
    MAX_SUMMARIES = 20         # keep last N summaries

    def __init__(self):
        self.messages = []
        self.summaries = []
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.FILE):
                with open(self.FILE, 'r', encoding='utf-8') as f:
                    self.messages = json.load(f)
                self.messages = self.messages[-self.MAX_MESSAGES:]
        except Exception:
            self.messages = []
        try:
            if os.path.exists(self.SUMMARY_FILE):
                with open(self.SUMMARY_FILE, 'r', encoding='utf-8') as f:
                    self.summaries = json.load(f)
                self.summaries = self.summaries[-self.MAX_SUMMARIES:]
        except Exception:
            self.summaries = []

    def _save(self):
        try:
            with open(self.FILE, 'w', encoding='utf-8') as f:
                json.dump(self.messages[-self.MAX_MESSAGES:], f, indent=2)
        except Exception:
            pass

    def _save_summaries(self):
        try:
            with open(self.SUMMARY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.summaries[-self.MAX_SUMMARIES:], f, indent=2)
        except Exception:
            pass

    def add(self, role, content):
        self.messages.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        self._save()
        # Auto-summarize old messages when buffer gets large
        if len(self.messages) >= self.SUMMARY_THRESHOLD:
            self._auto_summarize()

    def _auto_summarize(self):
        """Compress oldest messages into a summary to free up space."""
        old_messages = self.messages[:self.SUMMARY_BATCH]
        if not old_messages:
            return
        # Build a compact summary locally (no API call needed)
        lines = []
        for m in old_messages:
            prefix = 'User' if m['role'] == 'user' else 'Agent'
            text = m['content'][:100]
            lines.append(f'{prefix}: {text}')
        summary = {
            'period_start': old_messages[0].get('timestamp', ''),
            'period_end': old_messages[-1].get('timestamp', ''),
            'message_count': len(old_messages),
            'summary': '\n'.join(lines),
            'created': datetime.now().isoformat()
        }
        self.summaries.append(summary)
        self._save_summaries()
        # Remove summarized messages
        self.messages = self.messages[self.SUMMARY_BATCH:]
        self._save()

    def get_context(self, limit=15):
        """Return recent messages formatted for the LLM, including summary context."""
        msgs = []
        # Inject the most recent summary as context if available
        if self.summaries:
            latest_summary = self.summaries[-1]
            msgs.append({
                'role': 'system',
                'content': f'[Earlier conversation summary ({latest_summary["message_count"]} messages)]\n{latest_summary["summary"]}'
            })
        recent = self.messages[-limit:]
        for m in recent:
            msgs.append({'role': m['role'], 'content': m['content']})
        return msgs

    def get_summary_context(self):
        """Return a compact summary of recent interactions for system prompt."""
        parts = []
        # Include latest summary if exists
        if self.summaries:
            latest = self.summaries[-1]
            parts.append(f'[Earlier: {latest["message_count"]} messages summarized]')
            # Show key points from summary (first 3 lines)
            summary_lines = latest['summary'].split('\n')[:3]
            parts.extend(summary_lines)
            parts.append('...')
        # Recent raw messages
        recent = self.messages[-10:]
        if not recent and not parts:
            return 'No previous conversation.'
        for m in recent:
            prefix = 'User' if m['role'] == 'user' else 'Agent'
            text = m['content'][:150] + '...' if len(m['content']) > 150 else m['content']
            parts.append(f'{prefix}: {text}')
        return '\n'.join(parts)

    def search(self, query):
        """Search conversation history for relevant context."""
        query_lower = query.lower()
        results = []
        for m in reversed(self.messages):
            if query_lower in m['content'].lower():
                results.append(m)
                if len(results) >= 5:
                    break
        return results

    def clear(self):
        self.messages = []
        self.summaries = []
        self._save()
        self._save_summaries()


# ──────────────────────────────────────────────
# User Profile — learns preferences over time
# ──────────────────────────────────────────────
class UserProfile:
    """Tracks user patterns, preferences, and frequently used commands.
    Categorizes learned facts for smarter context injection."""
    FILE = os.path.join(AGENT_DATA_DIR, 'user_profile.json')

    TRACKED_KEYWORDS = [
        'screenshot', 'email', 'open chrome', 'lock', 'volume', 'search',
        'youtube', 'whatsapp', 'teams', 'files', 'system info', 'shutdown',
        'restart', 'drive', 'gmail', 'vscode', 'code', 'notepad', 'camera',
        'timer', 'wallpaper', 'weather', 'spotify', 'discord', 'zoom',
        'project', 'git', 'npm', 'python', 'install',
    ]

    def __init__(self):
        self.data = {
            'name': '',
            'preferences': {},
            'frequent_commands': {},
            'favorite_apps': [],
            'common_contacts': [],
            'facts': [],             # legacy flat list
            'categorized_facts': {   # organized by category
                'identity': [],      # name, age, location
                'work': [],          # job, company, role
                'tech': [],          # skills, tools, languages
                'preferences': [],   # likes, dislikes, habits
                'schedule': [],      # routines, meetings
                'contacts': [],      # people they mention
                'projects': [],      # current projects / repos
            },
            'total_interactions': 0,
            'first_seen': '',
            'last_seen': '',
            'session_count': 0,
        }
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.FILE):
                with open(self.FILE, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                    # Deep merge categorized_facts
                    if 'categorized_facts' in saved:
                        for cat, facts in saved['categorized_facts'].items():
                            if cat in self.data['categorized_facts']:
                                self.data['categorized_facts'][cat] = facts
                        del saved['categorized_facts']
                    self.data.update(saved)
        except Exception:
            pass

    def _save(self):
        try:
            with open(self.FILE, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass

    def track_command(self, user_input):
        """Track what commands/requests are used most."""
        self.data['total_interactions'] += 1
        self.data['last_seen'] = datetime.now().isoformat()
        if not self.data['first_seen']:
            self.data['first_seen'] = self.data['last_seen']
        lower = user_input.strip().lower()
        for keyword in self.TRACKED_KEYWORDS:
            if keyword in lower:
                self.data['frequent_commands'][keyword] = \
                    self.data['frequent_commands'].get(keyword, 0) + 1
        self._save()

    def learn_from_ai(self, learned_facts):
        """Store facts the AI learned, auto-categorizing them."""
        if not learned_facts or not isinstance(learned_facts, list):
            return
        for fact in learned_facts:
            if not fact or not isinstance(fact, str):
                continue
            # Skip duplicates (fuzzy — check if very similar fact exists)
            fact_lower = fact.lower().strip()
            if any(fact_lower == f.lower().strip() for f in self.data['facts']):
                continue
            # Add to flat list (backward compat)
            self.data['facts'].append(fact)
            # Auto-categorize
            category = self._categorize_fact(fact_lower)
            cat_list = self.data['categorized_facts'].get(category, [])
            if fact not in cat_list:
                cat_list.append(fact)
                # Keep each category manageable
                self.data['categorized_facts'][category] = cat_list[-15:]
        self.data['facts'] = self.data['facts'][-50:]
        self._save()

    def _categorize_fact(self, fact_lower):
        """Auto-categorize a learned fact based on keywords."""
        if any(w in fact_lower for w in ['name is', 'called', 'years old', 'lives in', 'from ']):
            return 'identity'
        if any(w in fact_lower for w in ['works', 'job', 'company', 'developer', 'engineer', 'manager', 'student']):
            return 'work'
        if any(w in fact_lower for w in ['uses', 'prefers', 'codes in', 'python', 'javascript', 'react', 'node']):
            return 'tech'
        if any(w in fact_lower for w in ['likes', 'prefers', 'favorite', 'hates', 'always', 'never', 'usually']):
            return 'preferences'
        if any(w in fact_lower for w in ['morning', 'evening', 'meeting', 'routine', 'schedule', 'every day']):
            return 'schedule'
        if any(w in fact_lower for w in ['project', 'repo', 'building', 'working on', 'app called']):
            return 'projects'
        if any(w in fact_lower for w in ['friend', 'colleague', 'boss', 'email to', 'message to']):
            return 'contacts'
        return 'preferences'  # default

    def set_name(self, name):
        if name and name != self.data['name']:
            self.data['name'] = name
            self._save()

    def start_session(self):
        """Call at agent startup to track sessions."""
        self.data['session_count'] = self.data.get('session_count', 0) + 1
        self.data['last_seen'] = datetime.now().isoformat()
        self._save()

    def get_profile_summary(self):
        """Return a rich, categorized profile for the system prompt."""
        parts = []
        if self.data['name']:
            parts.append(f"Name: {self.data['name']}")

        # Show categorized facts
        cat_facts = self.data.get('categorized_facts', {})
        for category in ['identity', 'work', 'tech', 'projects', 'preferences', 'schedule', 'contacts']:
            facts = cat_facts.get(category, [])
            if facts:
                # Show last 5 per category
                recent = facts[-5:]
                parts.append(f"{category.title()}: {'; '.join(recent)}")

        # Fallback to flat facts if no categorized facts yet
        if not any(cat_facts.get(c) for c in cat_facts):
            if self.data['facts']:
                parts.append(f"Known facts: {'; '.join(self.data['facts'][-10:])}")

        top_cmds = sorted(self.data['frequent_commands'].items(), key=lambda x: -x[1])[:7]
        if top_cmds:
            parts.append(f"Most used: {', '.join(f'{c[0]}({c[1]}x)' for c in top_cmds)}")
        parts.append(f"Sessions: {self.data.get('session_count', 0)} | Total interactions: {self.data['total_interactions']}")
        return '\n'.join(parts) if parts else 'New user, no history yet.'


# Initialize global instances
memory = ConversationMemory()
profile = UserProfile()


GROQ_MODELS = [
    'llama-3.3-70b-versatile',   # primary — best quality
    'llama-3.1-8b-instant',      # fallback — fast, lower quality
]


def _groq_request(messages, api_key, model, temperature=0.3, max_tokens=2048, timeout=20):
    """Low-level Groq API call. Returns (response_text, error_string)."""
    try:
        resp = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
            },
            json={
                'model': model,
                'messages': messages,
                'max_tokens': max_tokens,
                'temperature': temperature,
            },
            timeout=timeout,
        )
        if resp.status_code == 429:
            return None, 'rate_limited'
        if resp.status_code != 200:
            log.warning(f'Groq HTTP {resp.status_code}: {resp.text[:300]}')
            return None, f'http_{resp.status_code}'
        data = resp.json()
        result = data['choices'][0]['message']['content'].strip()
        # Strip markdown code fences if present
        if result.startswith('```'):
            result = result.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
        return result, None
    except requests.exceptions.Timeout:
        return None, 'timeout'
    except Exception as e:
        log.error(f'Groq error: {e}')
        return None, str(e)


def ask_groq(user_input, system_prompt=None, use_context=False, temperature=0.3, max_tokens=2048):
    """Use Groq LLaMA to process a request with smart fallback chain:
    1. Primary model + primary key
    2. Primary model + backup key (if rate limited)
    3. Fallback model + primary key
    4. Fallback model + backup key
    """
    if not GROQ_API_KEY:
        return None

    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    if use_context:
        messages.extend(memory.get_context(limit=15))
    else:
        messages.append({'role': 'user', 'content': user_input})

    # Build fallback chain: (model, api_key) pairs
    attempts = [(GROQ_MODELS[0], GROQ_API_KEY)]
    if GROQ_API_KEY_BACKUP:
        attempts.append((GROQ_MODELS[0], GROQ_API_KEY_BACKUP))
    attempts.append((GROQ_MODELS[1], GROQ_API_KEY))
    if GROQ_API_KEY_BACKUP:
        attempts.append((GROQ_MODELS[1], GROQ_API_KEY_BACKUP))

    for model, key in attempts:
        result, error = _groq_request(messages, key, model, temperature, max_tokens)
        if result is not None:
            return result
        if error == 'rate_limited':
            key_label = 'primary' if key == GROQ_API_KEY else 'backup'
            log.warning(f'Rate limited on {model} ({key_label} key), trying next...')
            continue
        if error == 'timeout':
            log.warning(f'Timeout on {model}, trying next...')
            continue
        # Other errors — still try next in chain
        log.warning(f'Error on {model}: {error}, trying next...')

    log.error('All Groq API attempts exhausted')
    return None


def ask_groq_command(user_input):
    """Translate natural language to Windows command via Groq."""
    cmd = ask_groq(user_input, system_prompt=(
        'You are a Windows command translator. Convert the user request '
        'into the exact Windows cmd or powershell command to execute.\n'
        'Rules:\n'
        '- Reply with ONLY the raw command, nothing else\n'
        '- No explanation, no markdown, no backticks\n'
        '- Use "start" to open apps/URLs on Windows\n'
        '- Use double backslashes for paths\n'
        '- If already a valid command, return as-is'
    ), temperature=0.1, max_tokens=256)
    return cmd


# Local command translator — no API needed
COMMAND_MAP = {
    # Browsers
    'open chrome': 'start chrome',
    'open browser': 'start chrome',
    'open firefox': 'start firefox',
    'open edge': 'start msedge',
    'open brave': 'start brave',
    # Productivity
    'open notepad': 'start notepad',
    'open calculator': 'start calc',
    'open calc': 'start calc',
    'open paint': 'start mspaint',
    'open file explorer': 'start explorer',
    'open explorer': 'start explorer',
    'open cmd': 'start cmd',
    'open terminal': 'start wt',
    'open powershell': 'start powershell',
    'open task manager': 'start taskmgr',
    'open settings': 'start ms-settings:',
    'open control panel': 'start control',
    'open vscode': 'start code',
    'open vs code': 'start code',
    # Microsoft Office
    'open word': 'start winword',
    'open excel': 'start excel',
    'open powerpoint': 'start powerpnt',
    'open outlook': 'start outlook',
    'open onenote': 'start onenote',
    # Communication
    'open teams': 'start chrome https://teams.microsoft.com',
    'open slack': 'start chrome https://app.slack.com',
    'open discord': 'start chrome https://discord.com/app',
    'open whatsapp': 'start chrome https://web.whatsapp.com',
    'open telegram': 'start chrome https://web.telegram.org',
    'open zoom': 'start chrome https://app.zoom.us',
    'open gmail': 'start https://mail.google.com',
    'open mail': 'start https://mail.google.com',
    # Media
    'open spotify': 'start spotify:',
    'open youtube': 'start https://youtube.com',
    'open netflix': 'start https://netflix.com',
    # Social
    'open twitter': 'start https://twitter.com',
    'open x': 'start https://x.com',
    'open instagram': 'start https://instagram.com',
    'open facebook': 'start https://facebook.com',
    'open linkedin': 'start https://linkedin.com',
    'open reddit': 'start https://reddit.com',
    'open github': 'start https://github.com',
    # Utilities
    'open downloads': 'start explorer shell:Downloads',
    'open documents': 'start explorer shell:Documents',
    'open desktop': 'start explorer shell:Desktop',
    'open recycle bin': 'start explorer shell:RecycleBinFolder',
    # System commands
    'show ip': 'ipconfig',
    'my ip': 'ipconfig',
    'public ip': 'powershell -c "(Invoke-WebRequest ifconfig.me).Content"',
    'show wifi': 'netsh wlan show interfaces',
    'wifi password': 'netsh wlan show profile name=* key=clear',
    'list processes': 'tasklist',
    'running apps': 'tasklist /FI "STATUS eq RUNNING"',
    'shutdown': 'shutdown /s /t 60',
    'restart': 'shutdown /r /t 60',
    'cancel shutdown': 'shutdown /a',
    'sleep': 'rundll32.exe powrprof.dll,SetSuspendState 0,1,0',
    'whoami': 'whoami',
    'battery': 'powercfg /batteryreport & echo Battery report saved',
    'disk space': 'wmic logicaldisk get size,freespace,caption',
    'system info': 'systeminfo',
    'date': 'echo %date% %time%',
    'hostname': 'hostname',
    'uptime': 'powershell -c "(Get-Date) - (gcim Win32_OperatingSystem).LastBootUpTime"',
    # Volume
    'mute': 'powershell -c "(New-Object -ComObject WScript.Shell).SendKeys([char]173)"',
    'volume up': 'powershell -c "(New-Object -ComObject WScript.Shell).SendKeys([char]175)"',
    'volume down': 'powershell -c "(New-Object -ComObject WScript.Shell).SendKeys([char]174)"',
    # Clipboard
    'clipboard': 'powershell -c "Get-Clipboard"',
    'clear clipboard': 'powershell -c "Set-Clipboard -Value $null"',
    # Network
    'flush dns': 'ipconfig /flushdns',
    'ping google': 'ping google.com -n 4',
}


def search_youtube_video_url(query):
    """Search YouTube and return the direct URL of the first video result."""
    try:
        search_url = f'https://www.youtube.com/results?search_query={quote_plus(query)}'
        resp = requests.get(search_url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }, timeout=10)
        # Extract first video ID from the page
        video_ids = re.findall(r'watch\?v=([a-zA-Z0-9_-]{11})', resp.text)
        if video_ids:
            return f'https://www.youtube.com/watch?v={video_ids[0]}'
    except Exception as e:
        log.warning(f'YouTube search error: {e}')
    # Fallback to search results page
    return f'https://www.youtube.com/results?search_query={quote_plus(query)}'


def handle_youtube_play(query):
    """Search YouTube and directly play the first matching video."""
    try:
        url = search_youtube_video_url(query)
        subprocess.Popen(f'start "" "{url}"', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if '/watch?v=' in url:
            return f'▶ Now playing: {query}\n{url}'
        else:
            return f'🔍 Opened YouTube search for: {query}\n{url}'
    except Exception as e:
        return f'Error playing YouTube: {e}'


def translate_command(user_input):
    """Translate natural language to Windows command using local pattern matching."""
    lower = user_input.strip().lower()

    # Exact match
    if lower in COMMAND_MAP:
        return COMMAND_MAP[lower]

    # --- YouTube PLAY patterns (before generic search) ---
    # "play X on youtube" / "play X" / "play song X"
    play_yt_match = re.match(r'play\s+(?:song\s+)?(.+?)(?:\s+on\s+youtube)?$', lower)
    if play_yt_match:
        query = play_yt_match.group(1).strip()
        url = search_youtube_video_url(query)
        return f'start "" "{url}"'

    # "youtube play X" / "yt play X"
    yt_play_match = re.match(r'(?:youtube|yt)\s+play\s+(.+)', lower)
    if yt_play_match:
        url = search_youtube_video_url(yt_play_match.group(1).strip())
        return f'start "" "{url}"'

    # --- URL patterns ---
    # "go to <url>" or "open <url>"
    url_match = re.match(r'(?:go to|open|visit|browse)\s+(https?://\S+)', lower)
    if url_match:
        return f'start {url_match.group(1)}'

    # "open <domain.com>" pattern
    domain_match = re.match(r'(?:go to|open|visit|browse)\s+(\S+\.\S+)', lower)
    if domain_match:
        domain = domain_match.group(1)
        if not domain.startswith('http'):
            domain = f'https://{domain}'
        return f'start {domain}'

    # --- Email patterns ---
    # "send email to X" or "email X"
    email_match = re.match(r'(?:send\s+)?(?:email|mail)\s+(?:to\s+)?(\S+@\S+)(?:\s+subject\s+(.+?))?(?:\s+body\s+(.+))?$', lower)
    if email_match:
        to = email_match.group(1)
        subject = quote_plus(email_match.group(2) or '')
        body = quote_plus(email_match.group(3) or '')
        return f'start mailto:{to}?subject={subject}&body={body}'

    # "compose email" / "new email"
    if lower in ('compose email', 'new email', 'write email'):
        return 'start mailto:'

    # --- Messaging patterns ---
    # "message X on teams" / "teams message X"
    teams_match = re.match(r'(?:message|chat|msg|dm)\s+(.+?)\s+(?:on\s+)?teams', lower) or \
                  re.match(r'teams\s+(?:message|chat|msg|dm)\s+(.+)', lower)
    if teams_match:
        person = quote_plus(teams_match.group(1))
        return f'start https://teams.microsoft.com/l/chat/0/0?users={person}'

    # "message X on slack" / "slack message X"
    slack_match = re.match(r'(?:message|chat|msg|dm)\s+(.+?)\s+(?:on\s+)?slack', lower) or \
                  re.match(r'slack\s+(?:message|chat|msg|dm)\s+(.+)', lower)
    if slack_match:
        return 'start chrome https://app.slack.com'

    # "whatsapp X" / "message X on whatsapp"
    wa_match = re.match(r'(?:whatsapp|message\s+.+?\s+on\s+whatsapp)\s*(\+?\d[\d\s-]*)?', lower)
    if wa_match and wa_match.group(1):
        phone = re.sub(r'[\s-]', '', wa_match.group(1))
        return f'start https://wa.me/{phone}'

    # --- Search patterns ---
    search_match = re.match(r'(?:search|google|look up|find)\s+(?:for\s+)?(.+)', lower)
    if search_match:
        query = quote_plus(search_match.group(1))
        return f'start https://www.google.com/search?q={query}'

    # "youtube X" — search YouTube (non-play, just browse)
    yt_match = re.match(r'(?:youtube|yt)\s+(?:search\s+)?(.+)', lower)
    if yt_match:
        query = quote_plus(yt_match.group(1))
        return f'start https://www.youtube.com/results?search_query={query}'

    # --- App patterns ---
    # "open X on youtube" — play directly
    open_yt_match = re.match(r'open\s+(.+?)\s+on\s+youtube', lower)
    if open_yt_match:
        url = search_youtube_video_url(open_yt_match.group(1).strip())
        return f'start "" "{url}"'

    # "open X" fallback for any app
    if lower.startswith('open '):
        app = lower[5:].strip()
        return f'start {app}'

    # "close X" / "kill X"
    close_match = re.match(r'(?:close|kill|stop|end)\s+(.+)', lower)
    if close_match:
        app = close_match.group(1).strip()
        return f'taskkill /IM {app}.exe /F'

    # "type X" — type text using keyboard
    type_match = re.match(r'type\s+(.+)', lower)
    if type_match:
        text = type_match.group(1)
        safe = text.replace('"', '\\"')
        return f'powershell -c "(New-Object -ComObject WScript.Shell).SendKeys(\'{safe}\')"'

    # "copy X to clipboard"
    copy_match = re.match(r'copy\s+(.+?)(?:\s+to\s+clipboard)?$', lower)
    if copy_match:
        text = copy_match.group(1)
        return f'powershell -c "Set-Clipboard -Value \'{text}\'"'

    # No translation needed — return as-is (raw command)
    return user_input


def send_heartbeat():
    try:
        requests.post(f'{BACKEND_URL}/heartbeat', headers=HEADERS, timeout=5)
    except Exception as e:
        log.debug(f'Heartbeat error: {e}')


def post_result(command_id, output, screenshot=None):
    try:
        body = {'commandId': command_id, 'output': output}
        if screenshot:
            body['screenshot'] = screenshot
        # Try to parse structured response with suggestions and screenshot
        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict) and 'text' in parsed:
                body['output'] = parsed['text']
                if parsed.get('suggestions'):
                    body['suggestions'] = parsed['suggestions']
                if parsed.get('screenshot') and not screenshot:
                    body['screenshot'] = parsed['screenshot']
        except (json.JSONDecodeError, TypeError):
            pass
        requests.post(f'{BACKEND_URL}/result', headers=HEADERS, json=body, timeout=30)
    except Exception as e:
        log.error(f'Result post error: {e}')


def handle_shell(payload):
    try:
        # Try AI translation first, fall back to local translator
        cmd = ask_groq_command(payload) or translate_command(payload)
        cmd_lower = cmd.strip().lower()

        # Detect GUI/launch commands that shouldn't block
        is_gui = (
            cmd_lower.startswith('start ') or
            cmd_lower.startswith('code ') or
            cmd_lower.startswith('explorer ') or
            cmd_lower.startswith('notepad ') or
            'Start-Process' in cmd
        )

        if is_gui:
            subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            output = '✓ Launched'
        else:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            output = (result.stdout + result.stderr).strip() or '✓ Done'

        if cmd != payload:
            output = f'[Translated: {cmd}]\n{output}'
        return output
    except subprocess.TimeoutExpired:
        return 'Command timed out after 30s'
    except Exception as e:
        return f'Error: {e}'


def capture_screenshot(monitor=0):
    """Capture screenshot using mss. monitor=0 means ALL monitors combined,
    monitor=1 is first display, monitor=2 is second, etc."""
    try:
        import mss
        from PIL import Image
        from io import BytesIO
        with mss.mss() as sct:
            # mss monitors: [0] = all combined, [1] = first, [2] = second, etc.
            if monitor < 0 or monitor >= len(sct.monitors):
                monitor = 0  # fallback to all
            sct_img = sct.grab(sct.monitors[monitor])
            img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
            return img
    except ImportError:
        # Fallback to pyautogui if mss not installed
        import pyautogui
        return pyautogui.screenshot()
    except Exception as e:
        log.error(f'Screenshot error: {e}')
        import pyautogui
        return pyautogui.screenshot()


def handle_screenshot(monitor=0):
    """Take screenshot. monitor=0 for all screens, 1 for first, 2 for second, etc."""
    try:
        from io import BytesIO
        img = capture_screenshot(monitor)
        buf = BytesIO()
        img.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        log.error(f'Screenshot error: {e}')
        return None


def handle_camera_photo():
    """Capture a single photo from the webcam and return as base64."""
    try:
        import cv2
        from io import BytesIO
        from PIL import Image
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return json.dumps({'error': 'Could not open camera. Make sure no other app is using it.'})
        # Warm up camera (first few frames are often dark)
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return json.dumps({'error': 'Failed to capture frame from camera'})
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return json.dumps({'screenshot': b64, 'text': '📸 Photo captured from webcam'})
    except ImportError:
        return json.dumps({'error': 'opencv-python not installed. Run: pip install opencv-python'})
    except Exception as e:
        return json.dumps({'error': f'Camera error: {e}'})


def handle_camera_stream(duration=10):
    """Capture frames from webcam for N seconds, return them as a list of base64 images.
    This gives a 'live' feel by sending multiple frames."""
    try:
        import cv2
        from io import BytesIO
        from PIL import Image
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return json.dumps({'error': 'Could not open camera'})
        duration = min(duration, 30)  # cap at 30 seconds
        fps = 2  # capture 2 frames per second to keep payload small
        total_frames = int(duration * fps)
        frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            # Resize to reduce size
            w, h = img.size
            if w > 640:
                ratio = 640 / w
                img = img.resize((640, int(h * ratio)))
            buf = BytesIO()
            img.save(buf, format='JPEG', quality=60)
            frames.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
            time.sleep(1 / fps)
        cap.release()
        return json.dumps({
            'text': f'🎥 Captured {len(frames)} frames over {duration}s from webcam',
            'frames': frames,
            'screenshot': frames[-1] if frames else None,
            'fps': fps,
            'duration': duration,
        })
    except ImportError:
        return json.dumps({'error': 'opencv-python not installed. Run: pip install opencv-python'})
    except Exception as e:
        return json.dumps({'error': f'Camera stream error: {e}'})


def handle_camera_video(duration=10):
    """Record a video from webcam, save as mp4, return as base64."""
    try:
        import cv2
        duration = min(duration, 60)  # cap at 60s
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return json.dumps({'error': 'Could not open camera'})
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_cam = cap.get(cv2.CAP_PROP_FPS) or 20
        video_path = os.path.join(AGENT_DATA_DIR, 'webcam_recording.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps_cam, (w, h))
        start = time.time()
        frame_count = 0
        while time.time() - start < duration:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_count += 1
        cap.release()
        out.release()
        # Read and encode
        size = os.path.getsize(video_path)
        if size > 50 * 1024 * 1024:
            return json.dumps({'error': 'Video too large (>50MB)'})
        with open(video_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        return json.dumps({
            'text': f'🎥 Recorded {duration}s video ({frame_count} frames, {size // 1024}KB)',
            'video': b64,
            'mime': 'video/mp4',
            'name': 'webcam_recording.mp4',
            'size': size,
        })
    except ImportError:
        return json.dumps({'error': 'opencv-python not installed. Run: pip install opencv-python'})
    except Exception as e:
        return json.dumps({'error': f'Video recording error: {e}'})


def read_screen_with_ai(prompt="Describe everything you see on the screen in detail. Extract all text, buttons, links, and UI elements.", monitor=0):
    """Take a screenshot and use Groq vision model to understand what's on screen."""
    if not GROQ_API_KEY:
        return None, "No Groq API key configured"
    try:
        from io import BytesIO
        img = capture_screenshot(monitor)
        # Resize to reduce token usage (max 1280px wide)
        w, h = img.size
        if w > 1280:
            ratio = 1280 / w
            img = img.resize((1280, int(h * ratio)))
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=80)
        b64_img = base64.b64encode(buf.getvalue()).decode('utf-8')

        resp = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {GROQ_API_KEY}',
                'Content-Type': 'application/json',
            },
            json={
                'model': 'llama-3.2-90b-vision-preview',
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': prompt},
                            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64_img}'}}
                        ]
                    }
                ],
                'max_tokens': 2000,
            },
            timeout=30,
        )
        data = resp.json()
        if 'error' in data:
            return b64_img, f"Vision API error: {data['error'].get('message', str(data['error']))}"
        result = data['choices'][0]['message']['content'].strip()
        return b64_img, result
    except Exception as e:
        log.error(f'Screen read error: {e}')
        return None, f'Error reading screen: {e}'


def handle_screen_read(prompt=None):
    """Read the current screen and answer a question about it."""
    if not prompt:
        prompt = "Describe everything visible on the screen. List all text, UI elements, buttons, and important information."
    screenshot_b64, analysis = read_screen_with_ai(prompt)
    return json.dumps({
        'text': analysis,
        'screenshot': screenshot_b64,
    })


def handle_screen_action(data):
    """Multi-step: execute a command, wait for screen to load, then read the screen.
    Used for things like 'open vercel and list my projects'."""
    command = data.get('command', '')
    wait_seconds = min(data.get('wait', 5), 15)  # cap at 15s
    read_prompt = data.get('read_prompt', 'List everything you see on this screen. Extract all text, project names, links, and important details.')

    outputs = []

    # Step 1: Execute the command (open app/website)
    if command:
        try:
            cmd_lower = command.strip().lower()
            subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            outputs.append(f'✓ Launched: {command}')
        except Exception as e:
            outputs.append(f'❌ Failed to launch: {e}')
            return '\n'.join(outputs)

    # Step 2: Wait for the page/app to load
    log.info(f'Screen action: waiting {wait_seconds}s for screen to load...')
    time.sleep(wait_seconds)

    # Step 3: Read the screen
    screenshot_b64, analysis = read_screen_with_ai(read_prompt)
    outputs.append(analysis)

    return json.dumps({
        'text': '\n'.join(outputs),
        'screenshot': screenshot_b64,
    })


def handle_list_files(path):
    try:
        entries = []
        for name in os.listdir(path):
            full = os.path.join(path, name)
            is_dir = os.path.isdir(full)
            entry = {'name': name, 'type': 'dir' if is_dir else 'file'}
            if not is_dir:
                ext = os.path.splitext(name)[1].lower()
                entry['ext'] = ext
                try:
                    entry['size'] = os.path.getsize(full)
                except:
                    pass
            entries.append(entry)
        return json.dumps(entries)
    except Exception as e:
        return f'Error: {e}'


def handle_preview_file(file_path):
    """Read an image/video file and return as base64 with its mime type."""
    try:
        ext = os.path.splitext(file_path)[1].lower()
        mime_map = {
            '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.gif': 'image/gif', '.webp': 'image/webp', '.bmp': 'image/bmp',
            '.svg': 'image/svg+xml', '.ico': 'image/x-icon',
            '.mp4': 'video/mp4', '.webm': 'video/webm', '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo', '.mkv': 'video/x-matroska',
        }
        mime = mime_map.get(ext)
        if not mime:
            return json.dumps({'error': f'Unsupported file type: {ext}'})
        size = os.path.getsize(file_path)
        if size > 20 * 1024 * 1024:
            return json.dumps({'error': 'File too large (>20MB)'})
        with open(file_path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
        return json.dumps({'mime': mime, 'data': data, 'name': os.path.basename(file_path), 'size': size})
    except Exception as e:
        return json.dumps({'error': str(e)})


def handle_download_file(file_path):
    """Read any file and return as base64 for download."""
    try:
        size = os.path.getsize(file_path)
        if size > 50 * 1024 * 1024:
            return json.dumps({'error': 'File too large (>50MB)'})
        with open(file_path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
        ext = os.path.splitext(file_path)[1].lower()
        return json.dumps({'data': data, 'name': os.path.basename(file_path), 'size': size, 'ext': ext})
    except Exception as e:
        return json.dumps({'error': str(e)})


def handle_lock():
    try:
        subprocess.run('rundll32.exe user32.dll,LockWorkStation', shell=True)
        return 'PC locked'
    except Exception as e:
        return f'Error: {e}'


def handle_google_drive(query=''):
    """List files from Google Drive via the backend."""
    try:
        url = f'{BACKEND_URL}/google/drive/files'
        if query:
            url += f'?q={query}'
        resp = requests.get(url, headers=HEADERS, timeout=15)
        data = resp.json()
        if 'error' in data:
            return f'Error: {data["error"]}'
        files = data.get('files', [])
        if not files:
            return 'No files found in Google Drive.'
        lines = ['Google Drive files (recent 30):']
        for f in files:
            size = f.get('size', '')
            size_str = f' ({int(size)//1024}KB)' if size else ''
            lines.append(f'• {f["name"]}{size_str} — {f.get("webViewLink", "")}')
        return '\n'.join(lines)
    except Exception as e:
        return f'Error listing Drive files: {e}'


def handle_gmail_inbox():
    """Fetch recent Gmail inbox messages via the backend."""
    try:
        resp = requests.get(f'{BACKEND_URL}/google/gmail/inbox', headers=HEADERS, timeout=15)
        data = resp.json()
        if 'error' in data:
            return f'Error: {data["error"]}'
        msgs = data.get('messages', [])
        if not msgs:
            return 'No messages in inbox.'
        lines = ['Recent inbox (last 10):']
        for m in msgs:
            lines.append(f'• From: {m.get("from","?")} | Subject: {m.get("subject","?")}')
            if m.get('snippet'):
                lines.append(f'  {m["snippet"][:100]}')
        return '\n'.join(lines)
    except Exception as e:
        return f'Error fetching inbox: {e}'


def handle_send_email(payload):
    """Send email via Gmail API through the backend."""
    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
        to = data.get('to', '')
        subject = data.get('subject', '')
        body = data.get('body', '')
        if not to:
            return 'Error: No recipient specified'
        resp = requests.post(f'{BACKEND_URL}/google/gmail/send', headers=HEADERS,
                             json={'to': to, 'subject': subject, 'body': body}, timeout=15)
        result = resp.json()
        if result.get('ok'):
            return f'✓ Email sent to {to}'
        return f'Error: {result.get("error", "Unknown error")}'
    except Exception as e:
        return f'Error sending email: {e}'


def handle_draft_email(payload):
    """Create email draft via Gmail API through the backend."""
    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
        to = data.get('to', '')
        subject = data.get('subject', '')
        body = data.get('body', '')
        resp = requests.post(f'{BACKEND_URL}/google/gmail/draft', headers=HEADERS,
                             json={'to': to, 'subject': subject, 'body': body}, timeout=15)
        result = resp.json()
        if result.get('ok'):
            return f'✓ Draft created for {to}'
        return f'Error: {result.get("error", "Unknown error")}'
    except Exception as e:
        return f'Error creating draft: {e}'


def handle_open_app(app_name):
    """Dynamically open any app using AI command translation, with smart fallback."""
    try:
        # First: Ask AI how to open this app on Windows
        ai_cmd = ask_groq_command(f'open {app_name}')
        if ai_cmd:
            subprocess.Popen(ai_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return f'Opened {app_name}'

        # Fallback: Try to find the app dynamically
        lower = app_name.lower().strip()

        # Try Windows "where" to find the executable
        try:
            result = subprocess.run(f'where {lower}', shell=True, capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                exe_path = result.stdout.strip().split('\n')[0]
                subprocess.Popen(f'start "" "{exe_path}"', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return f'Opened {app_name}'
        except Exception:
            pass

        # Try Start Menu search via PowerShell
        try:
            ps_cmd = f'powershell -c "(Get-StartApps | Where-Object {{$_.Name -like \'*{lower}*\'}} | Select-Object -First 1).AppID"'
            result = subprocess.run(ps_cmd, shell=True, capture_output=True, text=True, timeout=10)
            app_id = result.stdout.strip()
            if app_id:
                subprocess.Popen(f'start shell:AppsFolder\\{app_id}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return f'Opened {app_name}'
        except Exception:
            pass

        # Last resort: just try "start <name>"
        subprocess.Popen(f'start {app_name}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f'Opened {app_name}'
    except Exception as e:
        return f'Error: {e}'


def build_system_prompt(profile_summary, google_connected, google_email, chat_summary):
    """Build a modular, well-structured system prompt for the AI agent."""

    google_status = f'Connected ({google_email})' if google_connected else 'Not connected'

    return f'''You are JARVIS — an elite, personal AI assistant with FULL control over a Windows 11 PC.
You are not a chatbot. You are an intelligent agent that THINKS before acting, remembers context,
learns from every interaction, and proactively helps the user get things done.

═══ IDENTITY & PERSONALITY ═══
- You are confident, resourceful, and slightly witty — like a smart friend who happens to control a PC.
- Be concise. No walls of text. 1-2 sentence messages unless the user asks for detail.
- Use the user's name when you know it. Remember their preferences.
- If you made a mistake before, own it briefly and fix it — don't over-apologize.
- When the user is vague ("do that thing", "open it"), use conversation history to infer what they mean.
- Be proactive: if opening VS Code, suggest relevant recent projects. If checking email, mention unread count.

═══ USER PROFILE ═══
{profile_summary}
Google: {google_status}

═══ RECENT CONVERSATION ═══
{chat_summary}

═══ THINKING PROCESS ═══
Before responding, mentally:
1. What is the user actually trying to accomplish? (not just what they literally said)
2. Do I have enough context from conversation history? Check for "it", "that", "same", "again" references.
3. Which action type best fits? Prefer specific actions over generic "execute".
4. Is this sensitive/destructive? If so, confirm first.
5. What would be a useful follow-up suggestion?

═══ RESPONSE FORMAT ═══
RESPOND WITH ONLY A JSON OBJECT. No text before or after. No markdown wrapping.

{{
  "action": "<action_type>",
  ... action-specific fields ...,
  "message": "Short, natural response to show the user",
  "suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"],
  "learned": ["any new fact about the user — name, job, preference, etc."]
}}

═══ ACTIONS REFERENCE ═══

EXECUTE — Run any Windows command:
{{"action": "execute", "command": "<cmd>", "message": "...", "suggestions": [...], "learned": [...]}}

Key commands to know:
  Apps:       start chrome | start firefox | start msedge | code "path" | start notepad | start calc
  System:     shutdown /s /t 60 | shutdown /r /t 60 | shutdown /a | rundll32.exe user32.dll,LockWorkStation
  Files:      dir "path" | mkdir "path" | del "path" | copy "src" "dst" | move "src" "dst" | type "path"
  Info:       systeminfo | ipconfig | tasklist | hostname | wmic logicaldisk get size,freespace,caption
  Volume:     powershell -Command "(New-Object -ComObject WScript.Shell).SendKeys([char]173)" (173=mute, 175=up, 174=down)
  Clipboard:  powershell -Command "Get-Clipboard" | powershell -Command "Set-Clipboard -Value 'text'"
  Settings:   start ms-settings: | start ms-settings:bluetooth | start ms-settings:network-wifi
  Folders:    explorer shell:Downloads | explorer shell:Documents | explorer shell:Desktop
  URLs:       start chrome "https://example.com"
  Kill:       taskkill /IM app.exe /F
  Write file: powershell -Command "Set-Content -Path 'path' -Value 'content'"
  Date:       powershell -Command "Get-Date"

Rules: Use \\\\ in JSON paths. Quote paths with spaces. Full paths always (no cd). Use "start" for GUI apps.

ANSWER — For conversation, questions, knowledge:
{{"action": "answer", "response": "...", "suggestions": [...], "learned": [...]}}

MULTI-STEP — Chain sequential actions (MANDATORY for open+write, open+type tasks):
{{"action": "multi", "steps": [{{action1}}, {{action2}}, ...], "message": "...", "suggestions": [...], "learned": [...]}}
Each step is a full action object. Steps run sequentially with automatic delays between GUI launches.

*** MANDATORY MULTI EXAMPLE — "open notepad and write X": ***
{{"action": "multi", "steps": [
  {{"action": "execute", "command": "start notepad"}},
  {{"action": "type_text", "text": "the full text content here", "app": "Notepad"}}
], "message": "Opened Notepad and wrote the content!", "suggestions": [...], "learned": []}}

You MUST use "multi" with BOTH an "execute" step AND a "type_text" step when the user says:
- "open notepad and write/type X"
- "open notepad and write a song/poem/letter"
- "launch notepad and put X in it"
A single "execute" action CANNOT type text. You MUST include a separate "type_text" step with the ACTUAL content.

YOUTUBE PLAY — Play a song/video (searches and opens first result):
{{"action": "youtube_play", "query": "song name", "message": "...", "suggestions": [...], "learned": [...]}}
IMPORTANT: "play X", "play X on youtube" → ALWAYS use youtube_play, never execute with search URL.

EMAIL — Send or draft:
{{"action": "send_email", "to": "...", "subject": "...", "body": "...", "message": "...", "suggestions": [...], "learned": [...]}}
{{"action": "draft_email", "to": "...", "subject": "...", "body": "...", "message": "...", "suggestions": [...], "learned": [...]}}
Write complete emails with greeting and sign-off. NEVER use mailto or outlook — always use these actions.

GOOGLE — Drive and Gmail:
{{"action": "google_drive", "query": "...", "message": "...", "suggestions": [...], "learned": [...]}}
{{"action": "gmail_inbox", "message": "...", "suggestions": [...], "learned": [...]}}

SCREEN — Vision capabilities:
{{"action": "screen_read", "prompt": "what to look for", "message": "...", "suggestions": [...], "learned": [...]}}
{{"action": "screen_action", "command": "start chrome \\"url\\"", "wait": 6, "read_prompt": "what to extract", "message": "...", "suggestions": [...], "learned": [...]}}
User has MULTIPLE MONITORS: monitor=0 (all), monitor=1 (main), monitor=2 (second), etc.

CAMERA:
{{"action": "camera_photo", "message": "...", "suggestions": [...], "learned": [...]}}
{{"action": "camera_stream", "duration": 10, "message": "...", "suggestions": [...], "learned": [...]}}
{{"action": "camera_video", "duration": 10, "message": "...", "suggestions": [...], "learned": [...]}}

FILE OPERATIONS:
{{"action": "write_file", "path": "D:\\\\path\\\\file.txt", "content": "...", "message": "...", "suggestions": [...], "learned": [...]}}
{{"action": "append_file", "path": "D:\\\\path\\\\file.txt", "content": "...", "message": "...", "suggestions": [...], "learned": [...]}}
{{"action": "read_file", "path": "D:\\\\path\\\\file.txt", "message": "...", "suggestions": [...], "learned": [...]}}
{{"action": "edit_file", "path": "D:\\\\path\\\\file.txt", "find": "old", "replace": "new", "message": "...", "suggestions": [...], "learned": [...]}}
{{"action": "browse_folder", "path": "D:\\\\path", "message": "...", "suggestions": [...], "learned": [...]}}

CREATE PROJECT — Scaffold entire projects with files + commands:
{{"action": "create_project", "path": "D:\\\\Users\\\\user\\\\project", "steps": [
  {{"type": "file", "path": "index.html", "content": "<!DOCTYPE html>..."}},
  {{"type": "command", "command": "npm install"}}
], "message": "...", "suggestions": [...], "learned": [...]}}
Write COMPLETE, WORKING, PRODUCTION-READY code. Never placeholders.

UI & INPUT:
{{"action": "type_text", "text": "...", "app": "Notepad", "message": "...", "suggestions": [...], "learned": [...]}}
{{"action": "hotkey", "keys": "^s", "message": "...", "suggestions": [...], "learned": [...]}}
  Key codes: ^ = Ctrl, % = Alt, + = Shift. Ex: ^s = Ctrl+S, %{{F4}} = Alt+F4
{{"action": "clipboard_get", "message": "...", "suggestions": [...], "learned": [...]}}
{{"action": "clipboard_set", "text": "...", "message": "...", "suggestions": [...], "learned": [...]}}

SYSTEM:
{{"action": "notify", "title": "...", "text": "...", "message": "...", "suggestions": [...], "learned": [...]}}
{{"action": "set_wallpaper", "path": "D:\\\\path\\\\img.jpg", "message": "...", "suggestions": [...], "learned": [...]}}
{{"action": "window_manage", "operation": "minimize_all|restore_all|close_app|list_windows", "app": "app.exe", "message": "...", "suggestions": [...], "learned": [...]}}
{{"action": "set_timer", "seconds": 300, "label": "Break", "message": "...", "suggestions": [...], "learned": [...]}}
{{"action": "search_files", "query": "resume", "folder": "D:\\\\Users", "message": "...", "suggestions": [...], "learned": [...]}}
{{"action": "installed_apps", "message": "...", "suggestions": [...], "learned": [...]}}

WEB SEARCH — Search the web and return results:
{{"action": "web_search", "query": "search terms", "message": "...", "suggestions": [...], "learned": [...]}}
Use for: current events, weather, facts, prices, anything requiring up-to-date info.

WEB TEST — Autonomous website testing (signup, onboarding, bug finding):
{{"action": "web_test", "url": "https://example.com", "instructions": "optional specific test instructions", "message": "...", "suggestions": [...], "learned": [...]}}
Use when user says: "test this website", "find bugs on", "try signing up on", "test the onboarding", etc.
The agent will auto-create a temp email, sign up, go through onboarding, and report all bugs found.

SYSTEM HEALTH — Get a full system health report:
{{"action": "system_health", "message": "...", "suggestions": [...], "learned": [...]}}
Use for: "how's my PC", "system status", "check my computer", "diagnostics"

═══ CRITICAL RULES ═══
1. ALWAYS respond with valid JSON. Never plain text. Never markdown-wrapped.
2. "message" — short, natural, friendly. Use user's name if known.
3. "suggestions" — ALWAYS include 2-3 smart follow-ups. Complete sentences the user can tap.
4. "learned" — extract facts (name, job, preferences, schedule). Empty array [] if nothing new.
5. CONVERSATION CONTEXT — "do that again", "same thing", "that folder" → check history.
6. SENSITIVE ACTIONS — email sending, file deletion → these are handled automatically, just emit the action.
7. CODING — you ARE a full-stack developer. Write COMPLETE code. Use write_file, edit_file, create_project.
8. COMPLEX TASKS — use "multi" to chain steps. You are not limited to one action per turn.
9. You can run ANY Windows command, write ANY file, create ENTIRE projects. Use your full power.
10. When user asks to "open" something — figure out the right command. Don't be limited by examples.

═══ COMMON MISTAKES TO AVOID ═══
!! NEVER say you typed/wrote something in the "message" field without ACTUALLY including a type_text action.
!! "open notepad and write X" → MUST be "multi" with execute + type_text. NOT just execute alone.
!! "execute" action can ONLY run commands. It CANNOT type text into windows. Use "type_text" for that.
!! If the user asks you to write content somewhere, the ACTUAL TEXT must appear in a type_text or write_file action.
!! Do NOT fabricate results. If you only opened notepad, say "Opened Notepad" — don't claim you wrote lyrics.'''


def handle_ai(payload):
    """Smart AI handler with conversation context, memory, learning, and suggestions."""
    global pending_sensitive_action

    # Track user interaction
    profile.track_command(payload)
    memory.add('user', payload)

    google_email = ''
    google_connected = False
    try:
        resp = requests.get(f'{BACKEND_URL}/auth/google/status', timeout=5)
        gdata = resp.json()
        google_email = gdata.get('email', '')
        google_connected = gdata.get('connected', False)
    except:
        pass

    # If the user is confirming a pending sensitive action, execute or cancel it.
    if pending_sensitive_action is not None:
        if is_positive_confirmation(payload):
            action_data = pending_sensitive_action
            pending_sensitive_action = None
            output = _execute_ai_action(action_data, google_connected=google_connected)
            return json.dumps({
                'text': f'Confirmed. {output}',
                'suggestions': ['What else can I do?', 'Show me my inbox']
            })
        if is_negative_confirmation(payload):
            pending_sensitive_action = None
            return json.dumps({
                'text': 'Okay, I canceled that request. Let me know if you want to try something else.',
                'suggestions': ['What can you do?', 'Show me my inbox']
            })
        return json.dumps({
            'text': 'Please reply yes to confirm or no to cancel the pending action.',
            'suggestions': ['Yes, proceed', 'No, cancel']
        })

    profile_summary = profile.get_profile_summary()
    chat_summary = memory.get_summary_context()

    system_prompt = build_system_prompt(profile_summary, google_connected, google_email, chat_summary)

    # Use conversation-aware Groq call
    memory.add('user', payload)  # ensure current message is in context
    # Remove duplicate — we already added above
    memory.messages.pop()

    result = ask_groq(payload, system_prompt=system_prompt, use_context=True)
    if not result:
        return json.dumps({
            'text': "I'm having trouble connecting to my brain right now. Try again in a moment!",
            'suggestions': ['Check system status', 'Take a screenshot', 'What can you do?']
        })

    # Clean markdown
    result = result.strip()
    if result.startswith('```'):
        result = result.split('\n', 1)[-1].rsplit('```', 1)[0].strip()

    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        log.warning(f'LLM returned non-JSON: {result[:200]}')
        memory.add('assistant', result)
        return json.dumps({
            'text': result,
            'suggestions': ['Tell me more', 'Take a screenshot', 'What else can you do?']
        })

    # Log what the LLM decided to do — critical for debugging
    action = data.get('action', 'unknown')
    if action == 'multi':
        step_actions = [s.get('action', '?') for s in data.get('steps', [])]
        log.info(f'LLM action: multi → {step_actions}')
    else:
        log.info(f'LLM action: {action}')

    # Extract suggestions and learned facts before executing
    suggestions = data.get('suggestions', [])
    learned = data.get('learned', [])
    friendly_msg = data.get('message', '')

    # Learn from conversation
    if learned:
        profile.learn_from_ai(learned)
    # If AI detected user's name
    if data.get('user_name'):
        profile.set_name(data['user_name'])

    # If the AI wants to perform a sensitive action, ask the user to confirm first.
    if is_sensitive_action(data):
        pending_sensitive_action = data
        confirm_text = build_confirmation_text(data)
        return json.dumps({
            'text': confirm_text,
            'suggestions': ['Yes, proceed', 'No, cancel']
        })

    # Execute the action
    action_output = _execute_ai_action(data, google_connected)

    # Check if action_output is a JSON string with screenshot (from screen_read/screen_action)
    screenshot_b64 = None
    try:
        parsed_output = json.loads(action_output)
        if isinstance(parsed_output, dict) and 'screenshot' in parsed_output:
            screenshot_b64 = parsed_output.get('screenshot')
            action_output = parsed_output.get('text', action_output)
    except (json.JSONDecodeError, TypeError):
        pass

    # Build the final response with message + action output
    if friendly_msg:
        if action_output and action_output not in friendly_msg:
            final_text = f"{friendly_msg}\n\n{action_output}"
        else:
            final_text = friendly_msg
    else:
        final_text = action_output

    # Save agent's response to memory
    memory.add('assistant', final_text)

    # Return structured response with suggestions (and screenshot if available)
    result = {
        'text': final_text,
        'suggestions': suggestions[:3]
    }
    if screenshot_b64:
        result['screenshot'] = screenshot_b64
    return json.dumps(result)


def _execute_ai_action(data, google_connected):
    """Execute a single AI action or multi-step actions."""
    action = data.get('action', '')

    if action == 'multi':
        # Execute multiple steps sequentially
        outputs = []
        prev_was_gui = False
        for step in data.get('steps', []):
            # If previous step launched a GUI app, wait for it to be ready
            if prev_was_gui:
                if step.get('action') in ('type_text', 'hotkey'):
                    # Wait longer and verify window is ready before typing
                    time.sleep(5)
                else:
                    time.sleep(2)
            out = _execute_ai_action(step, google_connected)
            outputs.append(out)
            # Detect if this step launched a GUI app
            prev_was_gui = False
            if step.get('action') == 'execute':
                cmd_l = step.get('command', '').strip().lower()
                if cmd_l.startswith('start ') or 'notepad' in cmd_l or 'mspaint' in cmd_l or cmd_l.startswith('code '):
                    prev_was_gui = True
            # Small delay between steps
            time.sleep(1)
        return '\n'.join(outputs)

    elif action == 'send_email':
        return handle_send_email(data)

    elif action == 'draft_email':
        return handle_draft_email(data)

    elif action == 'google_drive':
        query = data.get('query', '') if isinstance(data, dict) else ''
        return handle_google_drive(query)

    elif action == 'gmail_inbox':
        return handle_gmail_inbox()

    elif action == 'youtube_play':
        query = data.get('query', '')
        if not query:
            return 'No song/video name provided'
        return handle_youtube_play(query)

    elif action == 'screen_read':
        prompt = data.get('prompt', None)
        return handle_screen_read(prompt)

    elif action == 'screen_action':
        return handle_screen_action(data)

    elif action == 'camera_photo':
        return handle_camera_photo()

    elif action == 'camera_stream':
        duration = data.get('duration', 10)
        return handle_camera_stream(duration)

    elif action == 'camera_video':
        duration = data.get('duration', 10)
        return handle_camera_video(duration)

    elif action == 'write_file':
        file_path = data.get('path', '')
        content = data.get('content', '')
        if not file_path:
            return 'No file path provided'
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True) if os.path.dirname(file_path) else None
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f'✓ Created/updated: {file_path} ({len(content)} chars)'
        except Exception as e:
            return f'❌ Error writing file: {e}'

    elif action == 'append_file':
        file_path = data.get('path', '')
        content = data.get('content', '')
        if not file_path:
            return 'No file path provided'
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(content)
            return f'✓ Appended to: {file_path}'
        except Exception as e:
            return f'❌ Error appending: {e}'

    elif action == 'read_file':
        file_path = data.get('path', '')
        if not file_path:
            return 'No file path provided'
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read(50000)  # Limit to 50KB
            return content if content else '(Empty file)'
        except Exception as e:
            return f'❌ Error reading file: {e}'

    elif action == 'create_project':
        steps = data.get('steps', [])
        project_path = data.get('path', '')
        if project_path:
            os.makedirs(project_path, exist_ok=True)
        outputs = []
        for step in steps:
            if step.get('type') == 'file':
                fpath = step.get('path', '')
                fcontent = step.get('content', '')
                try:
                    full_path = os.path.join(project_path, fpath) if project_path else fpath
                    os.makedirs(os.path.dirname(full_path), exist_ok=True) if os.path.dirname(full_path) else None
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(fcontent)
                    outputs.append(f'✓ {fpath}')
                except Exception as e:
                    outputs.append(f'❌ {fpath}: {e}')
            elif step.get('type') == 'command':
                try:
                    proc = subprocess.run(step.get('command', ''), shell=True, capture_output=True, text=True, timeout=120, cwd=project_path or None)
                    out = (proc.stdout + proc.stderr).strip()
                    outputs.append(f'✓ {step.get("command", "")}: {out[:200]}')
                except Exception as e:
                    outputs.append(f'❌ {step.get("command", "")}: {e}')
        return '\n'.join(outputs)

    elif action == 'edit_file':
        file_path = data.get('path', '')
        find_text = data.get('find', '')
        replace_text = data.get('replace', '')
        if not file_path:
            return 'No file path provided'
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if find_text and find_text in content:
                content = content.replace(find_text, replace_text, 1)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f'✓ Edited: {file_path}'
            else:
                return f'❌ Text not found in {file_path}'
        except Exception as e:
            return f'❌ Error editing: {e}'

    elif action == 'browse_folder':
        folder_path = data.get('path', '')
        if not folder_path:
            return 'No folder path provided'
        try:
            items = os.listdir(folder_path)
            result_lines = [f'📁 {folder_path} ({len(items)} items):']
            for item in sorted(items)[:50]:
                full = os.path.join(folder_path, item)
                if os.path.isdir(full):
                    result_lines.append(f'  📂 {item}/')
                else:
                    size = os.path.getsize(full)
                    size_str = f'{size//1024}KB' if size >= 1024 else f'{size}B'
                    result_lines.append(f'  📄 {item} ({size_str})')
            if len(items) > 50:
                result_lines.append(f'  ... and {len(items)-50} more')
            return '\n'.join(result_lines)
        except Exception as e:
            return f'❌ Error browsing: {e}'

    elif action == 'execute':
        cmd = data.get('command', '')
        if not cmd:
            return 'No command provided'
        try:
            # Detect GUI/launch commands that shouldn't block
            cmd_lower = cmd.strip().lower()
            is_gui = (
                cmd_lower.startswith('start ') or
                cmd_lower.startswith('code ') or
                cmd_lower.startswith('explorer ') or
                cmd_lower.startswith('notepad ') or
                cmd_lower.startswith('mspaint') or
                'Start-Process' in cmd or
                'start chrome' in cmd_lower or
                'start msedge' in cmd_lower or
                '.exe' in cmd_lower and not cmd_lower.startswith('taskkill')
            )

            if is_gui:
                # Non-blocking launch for GUI apps
                subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return f'✓ Launched: {cmd}'
            else:
                # Blocking execution for commands that return output
                proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
                output = (proc.stdout + proc.stderr).strip()
                return f'{output}' if output else f'✓ Done: {cmd}'
        except subprocess.TimeoutExpired:
            return f'⏱ Command still running (timed out after 60s): {cmd}'
        except Exception as e:
            return f'❌ Error executing [{cmd}]: {e}'

    elif action == 'clipboard_get':
        try:
            proc = subprocess.run('powershell -c "Get-Clipboard"', shell=True, capture_output=True, text=True, timeout=5)
            return proc.stdout.strip() or '(Clipboard is empty)'
        except Exception as e:
            return f'❌ Error: {e}'

    elif action == 'clipboard_set':
        text = data.get('text', '')
        try :
            # Use stdin to avoid shell injection
            proc = subprocess.run(['powershell', '-c', 'Set-Clipboard -Value $input'], input=text, capture_output=True, text=True, timeout=5)
            return f'✓ Copied to clipboard: {text[:100]}{"..." if len(text)>100 else ""}'
        except Exception as e:
            return f'❌ Error: {e}'

    elif action == 'notify':
        title = data.get('title', 'JARVIS')
        message = data.get('text', '')
        try:
            ps = f'''
            [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
            $template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
            $textNodes = $template.GetElementsByTagName("text")
            $textNodes.Item(0).AppendChild($template.CreateTextNode("{title}")) > $null
            $textNodes.Item(1).AppendChild($template.CreateTextNode("{message}")) > $null
            $toast = [Windows.UI.Notifications.ToastNotification]::new($template)
            [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("JARVIS").Show($toast)
            '''
            subprocess.run(['powershell', '-c', ps], capture_output=True, timeout=10)
            return f'✓ Notification sent: {title}'
        except Exception as e:
            return f'❌ Error: {e}'

    elif action == 'set_wallpaper':
        path = data.get('path', '')
        try:
            ps = f'''
            Add-Type -TypeDefinition @"
            using System.Runtime.InteropServices;
            public class Wallpaper {{
                [DllImport("user32.dll", CharSet=CharSet.Auto)]
                public static extern int SystemParametersInfo(int uAction, int uParam, string lpvParam, int fuWinIni);
            }}
"@
            [Wallpaper]::SystemParametersInfo(0x0014, 0, "{path}", 0x01 -bor 0x02)
            '''
            subprocess.run(['powershell', '-c', ps], capture_output=True, timeout=10)
            return f'✓ Wallpaper set to: {path}'
        except Exception as e:
            return f'❌ Error: {e}'

    elif action == 'window_manage':
        operation = data.get('operation', '')
        try:
            if operation == 'minimize_all':
                subprocess.run(['powershell', '-c', '(New-Object -ComObject Shell.Application).MinimizeAll()'], timeout=5)
                return '✓ All windows minimized'
            elif operation == 'restore_all':
                subprocess.run(['powershell', '-c', '(New-Object -ComObject Shell.Application).UndoMinimizeAll()'], timeout=5)
                return '✓ All windows restored'
            elif operation == 'close_app':
                app = data.get('app', '')
                subprocess.run(f'taskkill /IM {app} /F', shell=True, capture_output=True, timeout=5)
                return f'✓ Closed {app}'
            elif operation == 'list_windows':
                proc = subprocess.run(['powershell', '-c', 'Get-Process | Where-Object {$_.MainWindowTitle -ne ""} | Select-Object ProcessName, MainWindowTitle | Format-Table -AutoSize | Out-String'], capture_output=True, text=True, timeout=10)
                return proc.stdout.strip() or 'No windows found'
            else:
                return f'Unknown operation: {operation}'
        except Exception as e:
            return f'❌ Error: {e}'

    elif action == 'set_timer':
        seconds = data.get('seconds', 0)
        label = data.get('label', 'Timer')
        try:
            # Schedule a notification after N seconds using PowerShell background job
            ps = f'Start-Job -ScriptBlock {{ Start-Sleep -Seconds {seconds}; Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show("{label} - Time is up!", "JARVIS Timer") }}'
            subprocess.run(['powershell', '-c', ps], capture_output=True, timeout=5)
            mins = seconds // 60
            secs = seconds % 60
            time_str = f'{mins}m {secs}s' if mins else f'{secs}s'
            return f'✓ Timer set: {label} ({time_str})'
        except Exception as e:
            return f'❌ Error: {e}'

    elif action == 'type_text':
        text = data.get('text', '')
        target_app = data.get('app', '')  # optional: activate a specific window first
        try:
            # If a target app is specified, wait for it and bring it to foreground
            if target_app:
                # Retry up to 5 times (total ~6s) to find and activate the window
                activated = False
                for attempt in range(5):
                    activate_ps = (
                        f"Add-Type -AssemblyName Microsoft.VisualBasic; "
                        f"$procs = Get-Process | Where-Object {{$_.MainWindowTitle -like '*{target_app}*'}}; "
                        f"if ($procs) {{ "
                        f"  [Microsoft.VisualBasic.Interaction]::AppActivate($procs[0].Id); "
                        f"  Write-Output 'ACTIVATED'; "
                        f"  Start-Sleep -Milliseconds 500 "
                        f"}} else {{ Write-Output 'NOT_FOUND' }}"
                    )
                    proc = subprocess.run(['powershell', '-c', activate_ps],
                                          capture_output=True, text=True, timeout=5)
                    if 'ACTIVATED' in proc.stdout:
                        activated = True
                        log.info(f'Window "{target_app}" activated on attempt {attempt + 1}')
                        break
                    log.info(f'Window "{target_app}" not found, attempt {attempt + 1}/5...')
                    time.sleep(1.5)

                if not activated:
                    log.warning(f'Could not find window "{target_app}" after 5 attempts')
                    return f'❌ Could not find {target_app} window — it may not be open yet'

            # Always use clipboard + paste — it's the most reliable method
            # Works regardless of text length, handles special chars, and is instant
            escaped_text = text.replace("'", "''")
            subprocess.run(['powershell', '-c', f"Set-Clipboard -Value '{escaped_text}'"],
                           capture_output=True, timeout=5)
            time.sleep(0.3)
            subprocess.run(['powershell', '-c',
                            "Add-Type -AssemblyName System.Windows.Forms; "
                            "[System.Windows.Forms.SendKeys]::SendWait('^v')"],
                           capture_output=True, timeout=5)
            display = text[:80] + '...' if len(text) > 80 else text
            return f'✓ Typed: {display}'
        except Exception as e:
            return f'❌ Error: {e}'

    elif action == 'hotkey':
        keys = data.get('keys', '')
        try:
            subprocess.run(['powershell', '-c', f"Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('{keys}')"], capture_output=True, timeout=5)
            return f'✓ Pressed: {keys}'
        except Exception as e:
            return f'❌ Error: {e}'

    elif action == 'search_files':
        query = data.get('query', '')
        folder = data.get('folder', 'C:\\Users')
        try:
            proc = subprocess.run(f'where /r "{folder}" *{query}*', shell=True, capture_output=True, text=True, timeout=30)
            results = proc.stdout.strip().split('\n')[:20]
            if results and results[0]:
                return f'Found {len(results)} files:\n' + '\n'.join(f'• {r}' for r in results)
            return f'No files matching "{query}" found in {folder}'
        except subprocess.TimeoutExpired:
            return '⏱ Search timed out (try a more specific folder)'
        except Exception as e:
            return f'❌ Error: {e}'

    elif action == 'installed_apps':
        try:
            proc = subprocess.run(['powershell', '-c', 'Get-ItemProperty HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\* | Where-Object {$_.DisplayName} | Select-Object DisplayName, DisplayVersion | Sort-Object DisplayName | Format-Table -AutoSize | Out-String'], capture_output=True, text=True, timeout=15)
            return proc.stdout.strip() or 'No apps found'
        except Exception as e:
            return f'❌ Error: {e}'

    elif action == 'web_search':
        query = data.get('query', '')
        if not query:
            return 'No search query provided'
        return handle_web_search(query)

    elif action == 'web_test':
        url = data.get('url', '')
        instructions = data.get('instructions', '')
        if not url:
            return 'No URL provided for testing'
        return handle_web_test(url, instructions)

    elif action == 'system_health':
        return handle_system_health()

    elif action == 'reminder':
        seconds = data.get('seconds', 0)
        label = data.get('label', 'Reminder')
        if not seconds:
            return 'No time specified for reminder'
        return handle_schedule_reminder(seconds, label)

    elif action == 'answer':
        return data.get('response', 'No response')

    else:
        return data.get('response', str(data))


def handle_web_test(url, instructions=''):
    """Run autonomous web testing on a URL using Playwright + AI."""
    try:
        tester = WebTestAgent(GROQ_API_KEY, GROQ_API_KEY_BACKUP)
        report = tester.run_test(url, instructions or None)

        result = {
            'text': report.get('summary', 'Test completed but no summary generated.'),
            'suggestions': [
                f'Test {url} again',
                'Test another website',
                'Show me the bug details',
            ],
        }
        if report.get('screenshot'):
            result['screenshot'] = report['screenshot']

        return json.dumps(result)
    except ImportError:
        return json.dumps({
            'text': '❌ Playwright is not installed. Run: pip install playwright && python -m playwright install chromium',
            'suggestions': ['Install playwright', 'What else can you do?'],
        })
    except Exception as e:
        return json.dumps({
            'text': f'❌ Web test failed: {e}',
            'suggestions': ['Try again', 'Test a different site'],
        })


def handle_web_search(query):
    """Search the web using DuckDuckGo instant answer API (no API key needed)."""
    try:
        # DuckDuckGo instant answer API
        resp = requests.get(
            'https://api.duckduckgo.com/',
            params={'q': query, 'format': 'json', 'no_redirect': 1, 'no_html': 1},
            timeout=10,
            headers={'User-Agent': 'JARVIS-Agent/4.0'}
        )
        data = resp.json()
        results = []

        # Abstract (Wikipedia-style answer)
        if data.get('Abstract'):
            results.append(f"📖 {data['Abstract']}")
            if data.get('AbstractURL'):
                results.append(f"Source: {data['AbstractURL']}")

        # Instant answer
        if data.get('Answer'):
            results.append(f"💡 {data['Answer']}")

        # Related topics
        related = data.get('RelatedTopics', [])[:5]
        for topic in related:
            if isinstance(topic, dict) and topic.get('Text'):
                results.append(f"• {topic['Text'][:200]}")

        if results:
            return '\n'.join(results)

        # Fallback: open in browser and tell user
        subprocess.Popen(f'start chrome "https://www.google.com/search?q={quote_plus(query)}"',
                         shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f'🔍 Opened Google search for: {query}\n(No instant answer available — check your browser)'

    except Exception as e:
        # Fallback to opening browser
        subprocess.Popen(f'start chrome "https://www.google.com/search?q={quote_plus(query)}"',
                         shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f'🔍 Opened Google search for: {query}'


def handle_system_health():
    """Collect comprehensive system health information."""
    health = {}
    try:
        # CPU usage
        proc = subprocess.run(
            ['powershell', '-c',
             'Get-CimInstance Win32_Processor | Select-Object -ExpandProperty LoadPercentage'],
            capture_output=True, text=True, timeout=10)
        health['cpu'] = f"{proc.stdout.strip()}%" if proc.stdout.strip() else 'N/A'
    except Exception:
        health['cpu'] = 'N/A'

    try:
        # Memory
        proc = subprocess.run(
            ['powershell', '-c',
             '$os = Get-CimInstance Win32_OperatingSystem; '
             '$used = [math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1MB, 1); '
             '$total = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1); '
             '$pct = [math]::Round(($used / $total) * 100, 0); '
             'Write-Output "$used/$total GB ($pct%)"'],
            capture_output=True, text=True, timeout=10)
        health['memory'] = proc.stdout.strip() or 'N/A'
    except Exception:
        health['memory'] = 'N/A'

    try:
        # Disk
        proc = subprocess.run(
            ['powershell', '-c',
             'Get-PSDrive -PSProvider FileSystem | Where-Object {$_.Used -gt 0} | '
             'ForEach-Object { $pct = [math]::Round(($_.Used / ($_.Used + $_.Free)) * 100, 0); '
             '"$($_.Name): $([math]::Round($_.Free / 1GB, 1))GB free ($pct% used)" }'],
            capture_output=True, text=True, timeout=10)
        health['disk'] = proc.stdout.strip() or 'N/A'
    except Exception:
        health['disk'] = 'N/A'

    try:
        # Uptime
        proc = subprocess.run(
            ['powershell', '-c',
             '$boot = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime; '
             '$up = (Get-Date) - $boot; '
             '"$($up.Days)d $($up.Hours)h $($up.Minutes)m"'],
            capture_output=True, text=True, timeout=10)
        health['uptime'] = proc.stdout.strip() or 'N/A'
    except Exception:
        health['uptime'] = 'N/A'

    try:
        # Battery (laptops)
        proc = subprocess.run(
            ['powershell', '-c',
             '$b = Get-CimInstance Win32_Battery; '
             'if ($b) { "$($b.EstimatedChargeRemaining)% ($($b.BatteryStatus))" } '
             'else { "No battery (desktop)" }'],
            capture_output=True, text=True, timeout=10)
        health['battery'] = proc.stdout.strip() or 'N/A'
    except Exception:
        health['battery'] = 'N/A'

    try:
        # Top processes by memory
        proc = subprocess.run(
            ['powershell', '-c',
             'Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 5 '
             'ProcessName, @{N="MemMB";E={[math]::Round($_.WorkingSet64/1MB,0)}} | '
             'ForEach-Object { "$($_.ProcessName): $($_.MemMB)MB" }'],
            capture_output=True, text=True, timeout=10)
        health['top_processes'] = proc.stdout.strip() or 'N/A'
    except Exception:
        health['top_processes'] = 'N/A'

    lines = [
        '🖥️ System Health Report',
        f'  CPU:      {health["cpu"]}',
        f'  Memory:   {health["memory"]}',
        f'  Disk:     {health["disk"]}',
        f'  Uptime:   {health["uptime"]}',
        f'  Battery:  {health["battery"]}',
        f'  Top apps: {health["top_processes"]}',
    ]
    return '\n'.join(lines)


def handle_schedule_reminder(seconds, label):
    """Schedule a reminder that shows a Windows notification after N seconds."""
    try:
        # Use a background PowerShell job
        ps = (
            f'Start-Job -ScriptBlock {{ '
            f'Start-Sleep -Seconds {seconds}; '
            f'[System.Reflection.Assembly]::LoadWithPartialName("System.Windows.Forms") | Out-Null; '
            f'[System.Windows.Forms.MessageBox]::Show("{label}", "JARVIS Reminder", '
            f'"OK", "Information") }}'
        )
        subprocess.run(['powershell', '-c', ps], capture_output=True, timeout=5)
        mins = seconds // 60
        secs = seconds % 60
        if mins > 0:
            time_str = f'{mins}m {secs}s' if secs else f'{mins}m'
        else:
            time_str = f'{secs}s'
        return f'✓ Reminder set: "{label}" in {time_str}'
    except Exception as e:
        return f'❌ Error setting reminder: {e}'


def execute_command(command):
    cmd_type = command.get('type')
    payload = command.get('payload', '')
    cmd_id = command.get('id')

    if cmd_type == 'shell':
        output = handle_shell(payload)
        post_result(cmd_id, output)
    elif cmd_type == 'ai':
        output = handle_ai(payload)
        post_result(cmd_id, output)
    elif cmd_type == 'screenshot':
        # Support monitor selection: payload can be monitor number or empty
        monitor = 0
        if payload:
            try:
                monitor = int(payload)
            except (ValueError, TypeError):
                pass
        screenshot = handle_screenshot(monitor)
        post_result(cmd_id, 'Screenshot taken', screenshot)
    elif cmd_type == 'camera_photo':
        output = handle_camera_photo()
        post_result(cmd_id, output)
    elif cmd_type == 'camera_stream':
        duration = 10
        if payload:
            try:
                duration = int(payload)
            except (ValueError, TypeError):
                pass
        output = handle_camera_stream(duration)
        post_result(cmd_id, output)
    elif cmd_type == 'camera_video':
        duration = 10
        if payload:
            try:
                duration = int(payload)
            except (ValueError, TypeError):
                pass
        output = handle_camera_video(duration)
        post_result(cmd_id, output)
    elif cmd_type == 'list_files':
        output = handle_list_files(payload)
        post_result(cmd_id, output)
    elif cmd_type == 'lock':
        output = handle_lock()
        post_result(cmd_id, output)
    elif cmd_type == 'open_app':
        output = handle_open_app(payload)
        post_result(cmd_id, output)
    elif cmd_type == 'preview_file':
        output = handle_preview_file(payload)
        post_result(cmd_id, output)
    elif cmd_type == 'download_file':
        output = handle_download_file(payload)
        post_result(cmd_id, output)
    elif cmd_type == 'send_email':
        output = handle_send_email(payload)
        post_result(cmd_id, output)
    elif cmd_type == 'draft_email':
        output = handle_draft_email(payload)
        post_result(cmd_id, output)
    elif cmd_type == 'google_drive':
        query = payload if isinstance(payload, str) else (payload or {}).get('query', '')
        output = handle_google_drive(query)
        post_result(cmd_id, output)
    elif cmd_type == 'gmail_inbox':
        output = handle_gmail_inbox()
        post_result(cmd_id, output)
    elif cmd_type == 'youtube_play':
        output = handle_youtube_play(payload)
        post_result(cmd_id, output)
    elif cmd_type == 'screen_read':
        output = handle_screen_read(payload if payload else None)
        post_result(cmd_id, output)
    elif cmd_type == 'screen_action':
        data = json.loads(payload) if isinstance(payload, str) else (payload or {})
        output = handle_screen_action(data)
        post_result(cmd_id, output)
    elif cmd_type == 'web_test':
        data = json.loads(payload) if isinstance(payload, str) else (payload or {})
        url = data.get('url', payload) if isinstance(data, dict) else payload
        instructions = data.get('instructions', '') if isinstance(data, dict) else ''
        output = handle_web_test(url, instructions)
        post_result(cmd_id, output)
    else:
        post_result(cmd_id, f'Unknown command type: {cmd_type}')


def main():
    profile.start_session()
    log.info('═══════════════════════════════════════')
    log.info('  JARVIS Agent v4.0 (Enhanced)')
    log.info(f'  Backend: {BACKEND_URL}')
    log.info(f'  Groq AI: {"✓ Connected" if GROQ_API_KEY else "✗ No API key"}')
    log.info(f'  Backup key: {"✓" if GROQ_API_KEY_BACKUP else "✗"}')
    log.info(f'  Memory: {len(memory.messages)} messages + {len(memory.summaries)} summaries')
    log.info(f'  Profile: {profile.data["total_interactions"]} interactions, {profile.data.get("session_count", 0)} sessions')
    log.info('═══════════════════════════════════════')

    # Background heartbeat thread — keeps agent online during long commands
    def heartbeat_loop():
        while True:
            try:
                send_heartbeat()
            except Exception:
                pass
            time.sleep(5)

    hb_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    hb_thread.start()
    log.info('Background heartbeat thread started')

    consecutive_errors = 0
    while True:
        try:
            resp = requests.get(f'{BACKEND_URL}/command/pending', headers=HEADERS, timeout=5)
            command = resp.json()
            consecutive_errors = 0  # reset on success
            if command:
                cmd_type = command.get('type', '?')
                payload = command.get('payload', '')
                log.info(f'CMD [{cmd_type}]: {str(payload)[:80]}{"..." if len(str(payload)) > 80 else ""}')
                try:
                    execute_command(command)
                    log.info(f'CMD [{cmd_type}] ✓ completed')
                except Exception as e:
                    log.error(f'CMD [{cmd_type}] ✗ failed: {e}')
                    post_result(command.get('id'), f'Agent error: {e}')
        except requests.exceptions.ConnectionError:
            consecutive_errors += 1
            if consecutive_errors <= 3:
                log.warning('Backend offline, retrying...')
            elif consecutive_errors == 10:
                log.error('Backend offline for extended period — still retrying')
            # Back off on repeated failures
            if consecutive_errors > 5:
                time.sleep(min(consecutive_errors * 2, 30))
        except Exception as e:
            consecutive_errors += 1
            log.error(f'Poll error: {e}')

        time.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    main()