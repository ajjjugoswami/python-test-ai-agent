import os
import time
import json
import base64
import subprocess
import requests
import re
import threading
from datetime import datetime
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:3001')
AGENT_KEY = os.getenv('AGENT_KEY', 'change-me-secret')
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
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
    """Stores conversation history with a rolling window. Persists to disk."""
    FILE = os.path.join(AGENT_DATA_DIR, 'conversations.json')
    MAX_MESSAGES = 50  # keep last N messages for context

    def __init__(self):
        self.messages = []
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.FILE):
                with open(self.FILE, 'r', encoding='utf-8') as f:
                    self.messages = json.load(f)
                # trim old messages
                self.messages = self.messages[-self.MAX_MESSAGES:]
        except Exception:
            self.messages = []

    def _save(self):
        try:
            with open(self.FILE, 'w', encoding='utf-8') as f:
                json.dump(self.messages[-self.MAX_MESSAGES:], f, indent=2)
        except Exception:
            pass

    def add(self, role, content):
        self.messages.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        self._save()

    def get_context(self, limit=20):
        """Return recent messages formatted for the LLM."""
        recent = self.messages[-limit:]
        return [{'role': m['role'], 'content': m['content']} for m in recent]

    def get_summary_context(self):
        """Return a compact summary of recent interactions for system prompt."""
        recent = self.messages[-10:]
        if not recent:
            return "No previous conversation."
        lines = []
        for m in recent:
            prefix = "User" if m['role'] == 'user' else "Agent"
            # Truncate long messages
            text = m['content'][:150] + '...' if len(m['content']) > 150 else m['content']
            lines.append(f"{prefix}: {text}")
        return '\n'.join(lines)

    def clear(self):
        self.messages = []
        self._save()


# ──────────────────────────────────────────────
# User Profile — learns preferences over time
# ──────────────────────────────────────────────
class UserProfile:
    """Tracks user patterns, preferences, and frequently used commands."""
    FILE = os.path.join(AGENT_DATA_DIR, 'user_profile.json')

    def __init__(self):
        self.data = {
            'name': '',
            'preferences': {},
            'frequent_commands': {},
            'favorite_apps': [],
            'common_contacts': [],
            'facts': [],  # things the agent learned about the user
            'total_interactions': 0,
        }
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.FILE):
                with open(self.FILE, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
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
        # Extract key action words
        lower = user_input.strip().lower()
        for keyword in ['screenshot', 'email', 'open chrome', 'lock', 'volume',
                        'search', 'youtube', 'whatsapp', 'teams', 'files',
                        'system info', 'shutdown', 'restart', 'drive', 'gmail']:
            if keyword in lower:
                self.data['frequent_commands'][keyword] = \
                    self.data['frequent_commands'].get(keyword, 0) + 1
        self._save()

    def learn_from_ai(self, learned_facts):
        """Store facts the AI learned about the user from conversation."""
        if learned_facts and isinstance(learned_facts, list):
            for fact in learned_facts:
                if fact and fact not in self.data['facts']:
                    self.data['facts'].append(fact)
            # Keep only last 50 facts
            self.data['facts'] = self.data['facts'][-50:]
            self._save()

    def set_name(self, name):
        if name and name != self.data['name']:
            self.data['name'] = name
            self._save()

    def get_profile_summary(self):
        """Return a compact profile for the system prompt."""
        parts = []
        if self.data['name']:
            parts.append(f"User's name: {self.data['name']}")
        if self.data['facts']:
            parts.append(f"Known facts: {'; '.join(self.data['facts'][-10:])}")
        top_cmds = sorted(self.data['frequent_commands'].items(), key=lambda x: -x[1])[:5]
        if top_cmds:
            parts.append(f"Most used: {', '.join(c[0] for c in top_cmds)}")
        parts.append(f"Total interactions: {self.data['total_interactions']}")
        return '\n'.join(parts) if parts else 'New user, no history yet.'


# Initialize global instances
memory = ConversationMemory()
profile = UserProfile()


def ask_groq(user_input, system_prompt=None, use_context=False):
    """Use Groq LLaMA to process a request. Optionally includes conversation context."""
    if not GROQ_API_KEY:
        return None
    try:
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        if use_context:
            # Inject recent conversation history
            messages.extend(memory.get_context(limit=15))
        else:
            messages.append({'role': 'user', 'content': user_input})
        resp = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {GROQ_API_KEY}',
                'Content-Type': 'application/json',
            },
            json={
                'model': 'llama-3.3-70b-versatile',
                'messages': messages,
                'max_tokens': 2000,
            },
            timeout=15,
        )
        if resp.status_code == 429:
            print(f'[groq] Rate limited on primary model, trying fallback...')
            resp = requests.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {GROQ_API_KEY}',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': 'llama-3.1-8b-instant',
                    'messages': messages,
                    'max_tokens': 2000,
                },
                timeout=15,
            )
        if resp.status_code != 200:
            print(f'[groq] HTTP {resp.status_code}: {resp.text[:300]}')
            return None
        data = resp.json()
        result = data['choices'][0]['message']['content'].strip()
        if result.startswith('```'):
            result = result.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
        return result
    except Exception as e:
        print(f'[groq] error: {e}')
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
        '- If already a valid command, return as-is'
    ))
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
        print(f'[youtube] search error: {e}')
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
        print(f'[heartbeat] error: {e}')


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
        print(f'[result] error: {e}')


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
        print(f'[screenshot] error: {e}')
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
        print(f'[screenshot] error: {e}')
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
        print(f'[screen-read] error: {e}')
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
    print(f'[screen-action] Waiting {wait_seconds}s for screen to load...')
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

    system_prompt = f'''You are JARVIS — a powerful, personal AI assistant with FULL control over a Windows 11 PC.
You have memory of past conversations and you learn from the user over time.

═══ USER PROFILE ═══
{profile_summary}
Google: {"Connected (" + google_email + ")" if google_connected else "Not connected"}

═══ RECENT CONVERSATION ═══
{chat_summary}

═══ RESPONSE FORMAT ═══
RESPOND WITH ONLY A JSON OBJECT. No text before or after.

{{
  "action": "<action_type>",
  ... action-specific fields ...,
  "message": "A friendly, conversational response to show the user",
  "suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"],
  "learned": ["any new fact learned about the user from this conversation"]
}}

═══ ACTION TYPES ═══

1. EXECUTE — Run a command:
{{"action": "execute", "command": "...", "message": "...", "suggestions": [...], "learned": [...]}}

CRITICAL COMMAND KNOWLEDGE (use exactly these):
- Open folder in VS Code: code "D:\\path\\to\\folder"
- Open file in VS Code: code "D:\\path\\to\\file.txt"
- Open VS Code: code .
- Open folder in Explorer: explorer "D:\\path\\to\\folder"
- Open Chrome: start chrome
- Open URL: start chrome "https://example.com"
- Open any app: start appname
- Open specific file: start "" "D:\\path\\to\\file.docx"
- Close app: taskkill /IM chrome.exe /F
- Lock PC: rundll32.exe user32.dll,LockWorkStation
- Shutdown: shutdown /s /t 60
- Restart: shutdown /r /t 60
- Cancel shutdown: shutdown /a
- Sleep: rundll32.exe powrprof.dll,SetSuspendState 0,1,0
- Volume mute: powershell -Command "(New-Object -ComObject WScript.Shell).SendKeys([char]173)"
- Volume up: powershell -Command "(New-Object -ComObject WScript.Shell).SendKeys([char]175)"
- Volume down: powershell -Command "(New-Object -ComObject WScript.Shell).SendKeys([char]174)"
- List files: dir "D:\\path"
- Create folder: mkdir "D:\\path\\newfolder"
- Delete file: del "D:\\path\\file.txt"
- Copy file: copy "D:\\src" "D:\\dst"
- Move file: move "D:\\src" "D:\\dst"
- Read file content: type "D:\\path\\file.txt"
- Write to file: powershell -Command "Set-Content -Path 'D:\\path\\file.txt' -Value 'content'"
- System info: systeminfo
- IP address: ipconfig
- WiFi info: netsh wlan show interfaces
- Disk space: wmic logicaldisk get size,freespace,caption
- Running processes: tasklist
- Clipboard: powershell -Command "Get-Clipboard"
- Set clipboard: powershell -Command "Set-Clipboard -Value 'text'"
- WhatsApp message: start chrome "https://web.whatsapp.com/send?phone=NUMBER&text=MESSAGE"
- Search Google: start chrome "https://www.google.com/search?q=QUERY"
- YouTube search: start chrome "https://www.youtube.com/results?search_query=QUERY"
- YouTube PLAY a song/video: USE ACTION "youtube_play" with the song/video name — this finds and plays the first result directly!
  IMPORTANT: When user says "play X", "play X on youtube", "open X song" — ALWAYS use youtube_play action, NOT execute with a search URL.
- Type text into active window: powershell -Command "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('text')"
- Open Task Manager: start taskmgr
- Open Settings: start ms-settings:
- Open Downloads: explorer shell:Downloads
- Open Documents: explorer shell:Documents
- Open Desktop: explorer shell:Desktop
- Open Bluetooth settings: start ms-settings:bluetooth
- Open WiFi settings: start ms-settings:network-wifi
- Date/time: powershell -Command "Get-Date"
- Uptime: powershell -Command "(Get-Date) - (Get-CimInstance Win32_OperatingSystem).LastBootUpTime"
- Installed apps: powershell -Command "Get-ItemProperty HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\* | Select DisplayName, DisplayVersion | Format-Table -AutoSize"
- Kill by window title: powershell -Command "Get-Process | Where-Object {{$_.MainWindowTitle -like '*keyword*'}} | Stop-Process -Force"

IMPORTANT:
- Use DOUBLE BACKSLASHES in JSON paths: D:\\\\Users\\\\...
- Always quote paths with spaces
- For "open X in VS Code" ALWAYS use: code "full\\path"
- For GUI apps use the direct executable name, NOT 'start' unless needed
- NEVER use 'cd' then run — put the full path in the command itself
- For any sensitive or destructive actions like sending email or deleting files, do not execute them immediately. Ask the user to confirm before proceeding.

2. SEND EMAIL:
{{"action": "send_email", "to": "...", "subject": "...", "body": "...", "message": "...", "suggestions": [...], "learned": [...]}}
- Compose complete, well-written emails with greeting and sign-off

3. DRAFT EMAIL:
{{"action": "draft_email", "to": "...", "subject": "...", "body": "...", "message": "...", "suggestions": [...], "learned": [...]}}

4. GOOGLE DRIVE:
{{"action": "google_drive", "query": "...", "message": "...", "suggestions": [...], "learned": [...]}}

5. GMAIL INBOX:
{{"action": "gmail_inbox", "message": "...", "suggestions": [...], "learned": [...]}}

6. ANSWER/CHAT — For questions, conversation, help:
{{"action": "answer", "response": "...", "suggestions": [...], "learned": [...]}}

7. MULTI-STEP — Complex tasks needing sequential steps:
{{"action": "multi", "steps": [...], "message": "...", "suggestions": [...], "learned": [...]}}
Each step is its own action object. Steps run sequentially. Use this for complex tasks like "create a project and open it in VS Code".

8. YOUTUBE PLAY — Play a song/video directly on YouTube:
{{"action": "youtube_play", "query": "song or video name", "message": "...", "suggestions": [...], "learned": [...]}}

9. SCREEN READ — Read/analyze what's currently on screen:
{{"action": "screen_read", "prompt": "what to look for on screen", "message": "...", "suggestions": [...], "learned": [...]}}

10. SCREEN ACTION — Open something, wait for it to load, then read the screen:
{{"action": "screen_action", "command": "start chrome \\"https://vercel.com/dashboard\\"", "wait": 6, "read_prompt": "List all project names and their status visible on this dashboard", "message": "...", "suggestions": [...], "learned": [...]}}

11. CAMERA PHOTO — Take a photo from webcam:
{{"action": "camera_photo", "message": "...", "suggestions": [...], "learned": [...]}}

12. CAMERA STREAM — Capture live frames from webcam for N seconds:
{{"action": "camera_stream", "duration": 10, "message": "...", "suggestions": [...], "learned": [...]}}

13. CAMERA VIDEO — Record a video from webcam:
{{"action": "camera_video", "duration": 10, "message": "...", "suggestions": [...], "learned": [...]}}

14. WRITE FILE — Create or overwrite a file with content:
{{"action": "write_file", "path": "D:\\\\path\\\\file.txt", "content": "file content here", "message": "...", "suggestions": [...], "learned": [...]}}
Use for: creating scripts, code files, config files, notes, HTML pages, etc.

15. APPEND FILE — Add content to end of existing file:
{{"action": "append_file", "path": "D:\\\\path\\\\file.txt", "content": "content to add", "message": "...", "suggestions": [...], "learned": [...]}}

16. READ FILE — Read and return file contents:
{{"action": "read_file", "path": "D:\\\\path\\\\file.txt", "message": "...", "suggestions": [...], "learned": [...]}}
Use when user says "read this file", "show me the code in...", "what's in this file"

17. EDIT FILE — Find and replace text in a file:
{{"action": "edit_file", "path": "D:\\\\path\\\\file.txt", "find": "old text", "replace": "new text", "message": "...", "suggestions": [...], "learned": [...]}}
Use for: fixing bugs, updating code, changing config values.

18. BROWSE FOLDER — List contents of a folder with sizes:
{{"action": "browse_folder", "path": "D:\\\\path\\\\folder", "message": "...", "suggestions": [...], "learned": [...]}}
Use when user says "show me what's in this folder", "list my projects", "what files are on desktop"

19. CREATE PROJECT — Scaffold an entire project with multiple files + commands:
{{"action": "create_project", "path": "D:\\\\Users\\\\username\\\\project-name", "steps": [
  {{"type": "file", "path": "index.html", "content": "<!DOCTYPE html>..."}},
  {{"type": "file", "path": "style.css", "content": "body {{ ... }}"}},
  {{"type": "file", "path": "package.json", "content": "{{...}}"}},
  {{"type": "command", "command": "npm install"}},
  {{"type": "command", "command": "git init"}}
], "message": "...", "suggestions": [...], "learned": [...]}}
YOU ARE A FULL-STACK DEVELOPER. Write COMPLETE, WORKING code. Not placeholders.

20. CLIPBOARD GET — Read current clipboard content:
{{"action": "clipboard_get", "message": "...", "suggestions": [...], "learned": [...]}}

21. CLIPBOARD SET — Copy text to clipboard:
{{"action": "clipboard_set", "text": "content to copy", "message": "...", "suggestions": [...], "learned": [...]}}

22. DESKTOP NOTIFICATION — Show a Windows toast notification:
{{"action": "notify", "title": "Reminder", "text": "Meeting in 5 mins", "message": "...", "suggestions": [...], "learned": [...]}}

23. SET WALLPAPER — Change desktop wallpaper:
{{"action": "set_wallpaper", "path": "D:\\\\path\\\\image.jpg", "message": "...", "suggestions": [...], "learned": [...]}}

24. WINDOW MANAGEMENT — Control windows on screen:
{{"action": "window_manage", "operation": "minimize_all|restore_all|close_app|list_windows", "app": "chrome.exe", "message": "...", "suggestions": [...], "learned": [...]}}
Use for: "minimize all windows", "close chrome", "what windows are open", "show me running apps"

25. SET TIMER — Set a countdown timer with notification:
{{"action": "set_timer", "seconds": 300, "label": "Break time", "message": "...", "suggestions": [...], "learned": [...]}}
Use for: "set a timer for 5 minutes", "remind me in 30 seconds", "pomodoro timer"

26. TYPE TEXT — Type text into the currently focused window:
{{"action": "type_text", "text": "hello world", "app": "Notepad", "message": "...", "suggestions": [...], "learned": [...]}}
Use for: "type this for me", "fill in the field with..."
The "app" field is OPTIONAL — if provided, it will activate that window first before typing.
IMPORTANT: When user says "open notepad and write X", use multi action:
{{"action": "multi", "steps": [{{"action": "execute", "command": "start notepad"}}, {{"action": "type_text", "text": "X", "app": "Notepad"}}], "message": "...", "suggestions": [...], "learned": [...]}}

27. HOTKEY — Press keyboard shortcuts:
{{"action": "hotkey", "keys": "^s", "message": "...", "suggestions": [...], "learned": [...]}}
Key codes: ^ = Ctrl, % = Alt, + = Shift. Examples: ^s = Ctrl+S, ^c = Ctrl+C, %{{F4}} = Alt+F4, ^+{{ESC}} = Ctrl+Shift+Esc

28. SEARCH FILES — Search for files by name:
{{"action": "search_files", "query": "resume", "folder": "D:\\\\Users", "message": "...", "suggestions": [...], "learned": [...]}}

29. INSTALLED APPS — List all installed applications:
{{"action": "installed_apps", "message": "...", "suggestions": [...], "learned": [...]}}

NOTE ON SCREENSHOTS: User has MULTIPLE MONITORS. When taking screenshots or reading screen:
- monitor=0 captures ALL monitors combined (default)
- monitor=1 captures first/main display
- monitor=2 captures second display, etc.
If user says "screenshot of second monitor" or "read my other screen", set monitor accordingly.

═══ CRITICAL RULES ═══
- ALWAYS respond with valid JSON. Never plain text.
- "message" — natural, friendly, short (1-2 sentences). Use user's name if known.
- "suggestions" — ALWAYS include 2-3 smart, relevant follow-ups the user might want next. Complete sentences they can tap to execute.
- "learned" — extract personal facts (name, job, preferences). Empty array if nothing new.
- USE CONVERSATION HISTORY — if user says "do that again", "same thing", "that folder", refer to previous messages.
- For emails, ALWAYS use send_email/draft_email action. NEVER use mailto or outlook.
- Be proactive — suggest useful next steps.
- If unsure, try the most likely approach. Don't over-ask questions.
- YOU HAVE FULL PC ACCESS. You can create files, write code, edit code, install packages, run any command, manage git repos, open any app, browse any folder. USE IT.
- For coding tasks: write COMPLETE, PRODUCTION-READY code. Never say "I can't code" — you CAN. Use write_file, edit_file, create_project.
- For complex tasks, use "multi" action to chain multiple steps together.
- You are not limited to the examples above. You can run ANY Windows command via "execute", write ANY file via "write_file", and create ENTIRE projects via "create_project".
- When the user asks to "open" something — figure out the right command dynamically. Don't rely on hardcoded app lists.'''

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
        memory.add('assistant', result)
        return json.dumps({
            'text': result,
            'suggestions': ['Tell me more', 'Take a screenshot', 'What else can you do?']
        })

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
            # If previous step launched a GUI app, wait longer for it to load
            if prev_was_gui and step.get('action') in ('type_text', 'hotkey'):
                time.sleep(3)
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
        try:
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
            # If a target app is specified, try to bring it to foreground first
            if target_app:
                activate_ps = f"""Add-Type -AssemblyName Microsoft.VisualBasic; $procs = Get-Process | Where-Object {{$_.MainWindowTitle -like '*{target_app}*'}}; if ($procs) {{ [Microsoft.VisualBasic.Interaction]::AppActivate($procs[0].Id); Start-Sleep -Milliseconds 500 }}"""
                subprocess.run(['powershell', '-c', activate_ps], capture_output=True, timeout=5)
            
            # For long text (>100 chars), use clipboard + paste (instant)
            if len(text) > 100:
                escaped_text = text.replace("'", "''")
                subprocess.run(['powershell', '-c', f"Set-Clipboard -Value '{escaped_text}'"], capture_output=True, timeout=5)
                subprocess.run(['powershell', '-c', "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('^v')"], capture_output=True, timeout=5)
                return f'✓ Pasted: {text[:80]}...'
            else:
                # For short text, use SendKeys (more reliable for special keys)
                escaped = text.replace('{', '{{').replace('}', '}}').replace('+', '{+}').replace('^', '{^}').replace('%', '{%}').replace('~', '{~}')
                subprocess.run(['powershell', '-c', f"Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('{escaped}')"], capture_output=True, timeout=5)
                return f'✓ Typed: {text[:80]}'
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

    elif action == 'answer':
        return data.get('response', 'No response')

    else:
        return data.get('response', str(data))


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
    else:
        post_result(cmd_id, f'Unknown command type: {cmd_type}')


def main():
    print(f'═══════════════════════════════════════')
    print(f'  JARVIS Agent v3.0 (Vision Enabled)')
    print(f'  Backend: {BACKEND_URL}')
    print(f'  Groq AI: {"✓ Connected" if GROQ_API_KEY else "✗ No API key"}')
    print(f'  Memory: {len(memory.messages)} messages loaded')
    print(f'  Profile: {profile.data["total_interactions"]} total interactions')
    print(f'═══════════════════════════════════════')

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
    print('[heartbeat] Background heartbeat thread started')

    while True:
        try:
            resp = requests.get(f'{BACKEND_URL}/command/pending', headers=HEADERS, timeout=5)
            command = resp.json()
            if command:
                cmd_type = command.get('type', '?')
                payload = command.get('payload', '')
                print(f'[cmd] {cmd_type}: {payload[:80]}{"..." if len(str(payload)) > 80 else ""}')
                try:
                    execute_command(command)
                    print(f'[cmd] ✓ {cmd_type} completed')
                except Exception as e:
                    print(f'[cmd] ✗ {cmd_type} failed: {e}')
                    post_result(command.get('id'), f'Agent error: {e}')
        except requests.exceptions.ConnectionError:
            print(f'[poll] Backend offline, retrying...')
        except Exception as e:
            print(f'[poll] error: {e}')

        time.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    main()
