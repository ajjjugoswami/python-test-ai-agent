"""
Web Testing Agent — AI-powered autonomous website tester.
Uses Playwright for browser automation and Groq AI for decision-making.
Can auto-create accounts with temp mail, navigate onboarding flows,
and produce detailed bug reports.
"""

import os
import json
import time
import base64
import random
import string
import logging
import requests
from datetime import datetime
from io import BytesIO

log = logging.getLogger('jarvis.webtester')

# ──────────────────────────────────────────────
# Temp Mail via mail.tm API (free, no signup)
# ──────────────────────────────────────────────

class TempMail:
    """Disposable email using mail.tm API."""
    BASE = 'https://api.mail.tm'

    def __init__(self):
        self.email = None
        self.password = None
        self.token = None
        self.account_id = None

    def create(self):
        """Create a new temp email account."""
        try:
            # Get available domains
            resp = requests.get(f'{self.BASE}/domains', timeout=10)
            domains = resp.json().get('hydra:member', [])
            if not domains:
                # Fallback: generate a fake email for sites that don't verify
                self.email = self._generate_fallback_email()
                self.password = self._random_password()
                log.info(f'Using fallback email (no mail.tm domains): {self.email}')
                return self.email

            domain = domains[0]['domain']
            username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
            self.email = f'{username}@{domain}'
            self.password = self._random_password()

            # Create account
            resp = requests.post(f'{self.BASE}/accounts', json={
                'address': self.email,
                'password': self.password,
            }, timeout=10)

            if resp.status_code in (200, 201):
                data = resp.json()
                self.account_id = data.get('id')
                # Get auth token
                self._authenticate()
                log.info(f'Temp email created: {self.email}')
                return self.email
            else:
                log.warning(f'mail.tm account creation failed: {resp.status_code} {resp.text[:200]}')
                self.email = self._generate_fallback_email()
                return self.email

        except Exception as e:
            log.warning(f'Temp mail error: {e}')
            self.email = self._generate_fallback_email()
            self.password = self._random_password()
            return self.email

    def _authenticate(self):
        try:
            resp = requests.post(f'{self.BASE}/token', json={
                'address': self.email,
                'password': self.password,
            }, timeout=10)
            if resp.status_code == 200:
                self.token = resp.json().get('token')
        except Exception as e:
            log.warning(f'mail.tm auth error: {e}')

    def wait_for_email(self, subject_contains=None, timeout=60):
        """Poll inbox for new emails. Returns the first matching email body."""
        if not self.token:
            return None

        start = time.time()
        headers = {'Authorization': f'Bearer {self.token}'}

        while time.time() - start < timeout:
            try:
                resp = requests.get(f'{self.BASE}/messages', headers=headers, timeout=10)
                if resp.status_code == 200:
                    messages = resp.json().get('hydra:member', [])
                    for msg in messages:
                        if subject_contains and subject_contains.lower() not in msg.get('subject', '').lower():
                            continue
                        # Fetch full message
                        msg_resp = requests.get(f'{self.BASE}/messages/{msg["id"]}', headers=headers, timeout=10)
                        if msg_resp.status_code == 200:
                            full = msg_resp.json()
                            return {
                                'subject': full.get('subject', ''),
                                'from': full.get('from', {}).get('address', ''),
                                'text': full.get('text', ''),
                                'html': full.get('html', [''])[0] if full.get('html') else '',
                            }
            except Exception as e:
                log.debug(f'Mail check error: {e}')

            time.sleep(5)

        return None

    def _generate_fallback_email(self):
        username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
        return f'{username}@testuser.local'

    def _random_password(self):
        chars = string.ascii_letters + string.digits + '!@#$'
        return ''.join(random.choices(chars, k=14))


# ──────────────────────────────────────────────
# Web Testing Engine
# ──────────────────────────────────────────────

class WebTestAgent:
    """Autonomous web testing agent using Playwright + AI."""

    def __init__(self, groq_api_key, groq_backup_key=None):
        self.groq_api_key = groq_api_key
        self.groq_backup_key = groq_backup_key
        self.temp_mail = TempMail()
        self.test_log = []
        self.bugs = []
        self.page = None
        self.browser = None
        self.context = None
        self.step_count = 0
        self.max_steps = 50
        self._consecutive_failures = 0
        self._last_actions = []  # Track recent actions to detect loops
        self._url_visit_count = {}  # Track how many steps spent on each URL
        self._filled_fields = {}  # Track which fields have been filled already

    def _ask_ai(self, prompt, system_prompt, max_tokens=2048):
        """Call Groq AI for decision making."""
        keys = [self.groq_api_key]
        if self.groq_backup_key:
            keys.append(self.groq_backup_key)

        for key in keys:
            try:
                resp = requests.post(
                    'https://api.groq.com/openai/v1/chat/completions',
                    headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
                    json={
                        'model': 'llama-3.3-70b-versatile',
                        'messages': [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': prompt},
                        ],
                        'max_tokens': max_tokens,
                        'temperature': 0.2,
                    },
                    timeout=30,
                )
                if resp.status_code == 429:
                    continue
                if resp.status_code == 200:
                    result = resp.json()['choices'][0]['message']['content'].strip()
                    if result.startswith('```'):
                        result = result.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
                    return result
            except Exception as e:
                log.warning(f'AI error: {e}')
        return None

    def _ask_ai_vision(self, screenshot_b64, prompt):
        """Send a screenshot to vision model for analysis."""
        keys = [self.groq_api_key]
        if self.groq_backup_key:
            keys.append(self.groq_backup_key)

        for key in keys:
            try:
                resp = requests.post(
                    'https://api.groq.com/openai/v1/chat/completions',
                    headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
                    json={
                        'model': 'llama-3.2-90b-vision-preview',
                        'messages': [{
                            'role': 'user',
                            'content': [
                                {'type': 'text', 'text': prompt},
                                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{screenshot_b64}'}}
                            ]
                        }],
                        'max_tokens': 2000,
                    },
                    timeout=30,
                )
                if resp.status_code == 429:
                    continue
                if resp.status_code == 200:
                    return resp.json()['choices'][0]['message']['content'].strip()
            except Exception as e:
                log.warning(f'Vision AI error: {e}')
        return None

    def _is_page_alive(self):
        """Check if the browser page is still usable."""
        try:
            return self.page and not self.page.is_closed()
        except Exception:
            return False

    def _take_screenshot(self):
        """Take a screenshot of the current page."""
        if not self._is_page_alive():
            return None
        try:
            screenshot_bytes = self.page.screenshot(type='jpeg', quality=75)
            return base64.b64encode(screenshot_bytes).decode('utf-8')
        except Exception as e:
            log.error(f'Screenshot error: {e}')
            return None

    def _get_page_context(self):
        """Extract page info with indexed selectors for reliable targeting."""
        if not self._is_page_alive():
            return {'url': '', 'title': '', 'inputs': [], 'buttons': [], 'errors': [], 'issues': [], 'bodyText': '', 'hasForms': False}
        try:
            context = self.page.evaluate('''() => {
                const getVisible = (el) => {
                    try {
                        const rect = el.getBoundingClientRect();
                        if (rect.width === 0 && rect.height === 0) return false;
                        const style = window.getComputedStyle(el);
                        return style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
                    } catch { return false; }
                };

                // Build a unique CSS selector for any element
                const getSelector = (el) => {
                    if (el.id) return '#' + CSS.escape(el.id);
                    if (el.name) return el.tagName.toLowerCase() + '[name="' + el.name + '"]';
                    if (el.type && el.tagName === 'INPUT') {
                        const sameType = [...document.querySelectorAll('input[type="' + el.type + '"]')];
                        const idx = sameType.indexOf(el);
                        if (sameType.length === 1) return 'input[type="' + el.type + '"]';
                        return 'input[type="' + el.type + '"]:nth-of-type(' + (idx+1) + ')';
                    }
                    if (el.placeholder) return el.tagName.toLowerCase() + '[placeholder="' + el.placeholder + '"]';
                    // Fallback: nth-child path
                    const parent = el.parentElement;
                    if (parent) {
                        const children = [...parent.children];
                        const idx = children.indexOf(el);
                        const parentSel = parent.id ? '#' + CSS.escape(parent.id) : parent.tagName.toLowerCase();
                        return parentSel + ' > ' + el.tagName.toLowerCase() + ':nth-child(' + (idx+1) + ')';
                    }
                    return el.tagName.toLowerCase();
                };

                const inputs = [...document.querySelectorAll('input, select, textarea')].filter(getVisible).map((el, i) => ({
                    index: i,
                    tag: el.tagName.toLowerCase(),
                    type: el.type || '',
                    name: el.name || '',
                    id: el.id || '',
                    placeholder: el.placeholder || '',
                    value: el.value || '',
                    label: el.labels?.[0]?.textContent?.trim() || '',
                    required: el.required,
                    ariaLabel: el.getAttribute('aria-label') || '',
                    selector: getSelector(el),
                }));

                const buttons = [...document.querySelectorAll('button, [role="button"], a, [type="submit"]')].filter(getVisible).slice(0, 40).map((el, i) => ({
                    index: i,
                    tag: el.tagName.toLowerCase(),
                    text: (el.textContent || '').trim().replace(/\\s+/g, ' ').substring(0, 80),
                    href: el.href || '',
                    type: el.type || '',
                    id: el.id || '',
                    ariaLabel: el.getAttribute('aria-label') || '',
                    selector: getSelector(el),
                }));

                const errors = [...document.querySelectorAll('[class*="error"], [class*="alert"], [class*="warning"], [role="alert"], .invalid-feedback, .form-error')]
                    .filter(getVisible)
                    .map(el => el.textContent?.trim().substring(0, 200))
                    .filter(t => t && t.length > 2);

                const bodyText = document.body?.innerText?.substring(0, 3000) || '';

                const issues = [];
                if (!document.title) issues.push('Page has no title');
                const noAlt = [...document.querySelectorAll('img')].filter(i => !i.alt).length;
                if (noAlt > 0) issues.push(noAlt + ' images without alt text');
                const brokenLinks = [...document.querySelectorAll('a[href=""], a[href="#"]')].length;
                if (brokenLinks > 0) issues.push(brokenLinks + ' empty/broken links');

                return {
                    url: window.location.href,
                    title: document.title || '',
                    inputs, buttons, errors, issues,
                    bodyText: bodyText.substring(0, 2000),
                    hasForms: document.querySelectorAll('form').length > 0,
                    hasPassword: !!document.querySelector('input[type="password"]'),
                    hasEmail: !!document.querySelector('input[type="email"]'),
                };
            }''')
            return context
        except Exception as e:
            log.error(f'Page context error: {e}')
            return {'url': self.page.url, 'title': '', 'inputs': [], 'buttons': [], 'errors': [], 'issues': [], 'bodyText': '', 'hasForms': False}

    def _log_step(self, action, result, screenshot=None, bug=None):
        """Log a test step."""
        self.step_count += 1
        entry = {
            'step': self.step_count,
            'action': action,
            'result': result,
            'timestamp': datetime.now().isoformat(),
            'url': self.page.url if self.page else '',
        }
        if screenshot:
            entry['screenshot'] = screenshot
        if bug:
            entry['bug'] = bug
            self.bugs.append({
                'step': self.step_count,
                'description': bug,
                'url': self.page.url if self.page else '',
                'timestamp': datetime.now().isoformat(),
            })
        self.test_log.append(entry)
        log.info(f'[Step {self.step_count}] {action} → {result[:100]}')

    def _safe_wait_for_load(self, timeout=8000):
        """Wait for page to settle, with fallback."""
        if not self._is_page_alive():
            return
        try:
            self.page.wait_for_load_state('networkidle', timeout=timeout)
        except Exception:
            try:
                self.page.wait_for_load_state('domcontentloaded', timeout=3000)
            except Exception:
                pass

    def _try_click(self, selector, value, description):
        """Try multiple click strategies. Returns True on success."""
        strategies = []

        # Strategy 1: CSS selector
        if selector:
            strategies.append(('css', lambda: self.page.click(selector, timeout=6000)))

        # Strategy 2: Playwright get_by_role for links/buttons
        if value:
            clean_text = value.strip()
            strategies.append(('role_link', lambda: self.page.get_by_role('link', name=clean_text).first.click(timeout=6000)))
            strategies.append(('role_button', lambda: self.page.get_by_role('button', name=clean_text).first.click(timeout=6000)))
            strategies.append(('get_by_text', lambda: self.page.get_by_text(clean_text, exact=False).first.click(timeout=6000)))
            # Partial text match
            if len(clean_text) > 10:
                # Use a substring (first few words)
                short = ' '.join(clean_text.split()[:3])
                strategies.append(('partial_text', lambda: self.page.get_by_text(short, exact=False).first.click(timeout=6000)))

        # Strategy 3: text= locator (Playwright's text selector)
        if value:
            strategies.append(('text_locator', lambda: self.page.locator(f'text=/{value[:30]}/i').first.click(timeout=6000)))

        for name, fn in strategies:
            try:
                fn()
                log.info(f'Click succeeded via {name}: {description}')
                return True
            except Exception as e:
                log.debug(f'Click strategy {name} failed: {str(e)[:100]}')
                continue

        return False

    def _execute_action(self, action_data):
        """Execute a single browser action decided by AI."""
        if not self._is_page_alive():
            self._log_step('Action skipped', 'Browser/page is closed')
            return False  # Stop the loop

        action = action_data.get('action', '')
        selector = action_data.get('selector', '')
        value = action_data.get('value', '')
        description = action_data.get('description', action)

        # Track for loop detection — key on action + selector ONLY (ignore value to catch AI tricks)
        if action == 'navigate':
            from urllib.parse import urlparse
            try:
                parsed = urlparse(value)
                action_key = f'navigate:{parsed.path}'
            except Exception:
                action_key = f'navigate:{value[:40]}'
        elif action in ('fill', 'click', 'select'):
            action_key = f'{action}:{selector[:40]}'
        else:
            action_key = f'{action}'
        self._last_actions.append(action_key)
        if len(self._last_actions) > 10:
            self._last_actions = self._last_actions[-10:]

        # Count how many times this exact action+target has been tried
        repeat_count = self._last_actions.count(action_key)
        if repeat_count >= 2:
            self._log_step(f'{action}: {description}', 'SKIPPED: Already tried this — moving on', bug=f'Agent stuck: {description}')
            self._consecutive_failures += 1
            return self._consecutive_failures < 3

        # URL-based stuck detection: if we've spent 8+ steps on the same page, force done
        current_url = self.page.url if self._is_page_alive() else ''
        self._url_visit_count[current_url] = self._url_visit_count.get(current_url, 0) + 1
        if self._url_visit_count[current_url] > 8:
            self._log_step('Stuck on page', f'Spent {self._url_visit_count[current_url]} steps on {current_url} — moving on',
                           bug=f'Stuck on page: {current_url}')
            return False

        try:
            if action == 'click':
                success = self._try_click(selector, value, description)
                if success:
                    self._consecutive_failures = 0
                    self._log_step(f'Click: {description}', 'OK')
                    self._safe_wait_for_load()
                    return True
                else:
                    self._consecutive_failures += 1
                    self._log_step(f'Click: {description}', f'FAILED: Could not find element. Selector: {selector}, Text: {value}',
                                   bug=f'Element not clickable: {description}')
                    return self._consecutive_failures < 3

            elif action == 'fill':
                if selector:
                    # Skip if we already filled this exact field
                    if selector in self._filled_fields:
                        self._log_step(f'Fill "{description}"', f'SKIPPED: Already filled with "{self._filled_fields[selector][:30]}"')
                        return True
                    try:
                        self.page.fill(selector, value, timeout=6000)
                    except Exception:
                        # Fallback: try clicking first then typing
                        try:
                            self.page.click(selector, timeout=3000)
                            self.page.keyboard.type(value, delay=30)
                        except Exception as e2:
                            self._log_step(f'Fill "{description}"', f'FAILED: {str(e2)[:150]}')
                            self._consecutive_failures += 1
                            return self._consecutive_failures < 3
                self._consecutive_failures = 0
                if selector:
                    self._filled_fields[selector] = value
                self._log_step(f'Fill "{description}": {value}', 'OK')
                return True

            elif action == 'select':
                if selector:
                    try:
                        self.page.select_option(selector, value, timeout=6000)
                    except Exception:
                        # Try clicking select then option text
                        try:
                            self.page.click(selector, timeout=3000)
                            time.sleep(0.5)
                            self.page.get_by_text(value, exact=False).first.click(timeout=3000)
                        except Exception as e2:
                            self._log_step(f'Select "{description}"', f'FAILED: {str(e2)[:150]}')
                            self._consecutive_failures += 1
                            return self._consecutive_failures < 3
                self._consecutive_failures = 0
                self._log_step(f'Select "{description}": {value}', 'OK')
                return True

            elif action == 'press':
                self.page.keyboard.press(value or 'Enter')
                self._consecutive_failures = 0
                self._log_step(f'Press key: {value or "Enter"}', 'OK')
                self._safe_wait_for_load()
                return True

            elif action == 'navigate':
                try:
                    self.page.goto(value, wait_until='domcontentloaded', timeout=20000)
                    self._safe_wait_for_load(5000)
                except Exception as e:
                    self._log_step(f'Navigate: {value}', f'FAILED: {str(e)[:150]}')
                    self._consecutive_failures += 1
                    return self._consecutive_failures < 3
                self._consecutive_failures = 0
                self._log_step(f'Navigate: {value}', 'OK')
                return True

            elif action == 'wait':
                seconds = min(int(value or 3), 10)
                time.sleep(seconds)
                self._log_step(f'Wait {seconds}s', 'OK')
                return True

            elif action == 'scroll':
                self.page.evaluate('window.scrollBy(0, 500)')
                self._log_step('Scroll down', 'OK')
                return True

            elif action == 'check_email':
                subject = value or None
                log.info('Checking temp email for verification...')
                email_data = self.temp_mail.wait_for_email(subject_contains=subject, timeout=45)
                if email_data:
                    self._log_step(f'Email received: {email_data["subject"]}', 'OK')
                    import re
                    links = re.findall(r'https?://[^\s<>"\']+(?:verify|confirm|activate|token|auth)[^\s<>"\']*',
                                       email_data.get('html', '') + email_data.get('text', ''))
                    if links:
                        self.page.goto(links[0], wait_until='domcontentloaded', timeout=20000)
                        self._log_step('Clicked verification link', 'OK')
                    return True
                else:
                    self._log_step('Check email', 'No verification email within 45s', bug='Verification email not received')
                    return True  # Continue testing anyway

            elif action == 'done':
                self._log_step('Testing complete', description)
                return False

            elif action == 'report_bug':
                self._log_step(f'Bug found: {description}', 'Bug reported', bug=description)
                return True

            else:
                self._log_step(f'Unknown action: {action}', 'Skipped')
                return True

        except Exception as e:
            error_msg = str(e)[:200]
            # Check if browser crashed — let main loop try recovery
            if 'closed' in error_msg.lower() or 'Target page' in error_msg:
                self._log_step(f'{action}: {description}', f'FAILED: Page closed', bug='Page closed unexpectedly')
                return True  # Return True so main loop can attempt recovery
            self._consecutive_failures += 1
            self._log_step(f'{action}: {description}', f'FAILED: {error_msg}', bug=f'Action failed - {action}: {error_msg}')
            return self._consecutive_failures < 3

    def run_test(self, url, test_instructions=None):
        """
        Main entry point. Launches browser, navigates to URL, and autonomously
        tests the website using AI-driven decisions.
        Returns a structured test report.
        """
        from playwright.sync_api import sync_playwright

        self.test_log = []
        self.bugs = []
        self.step_count = 0

        # Create temp email for signups
        temp_email = self.temp_mail.create()
        temp_password = self.temp_mail.password or TempMail._random_password(self.temp_mail)

        user_profile = {
            'email': temp_email,
            'password': temp_password,
            'first_name': 'Test',
            'last_name': 'User',
            'username': f'testuser_{random.randint(1000, 9999)}',
            'phone': f'+1{random.randint(2000000000, 9999999999)}',
        }

        log.info(f'Starting web test: {url}')
        log.info(f'Test email: {temp_email}')

        default_instructions = (
            'Explore the website, find signup/register, create an account, '
            'complete onboarding, and test all major features you can find. '
            'Report any bugs, errors, or UX issues.'
        )
        instructions = test_instructions or default_instructions

        screenshots_for_report = []

        try:
            with sync_playwright() as p:
                self.browser = p.chromium.launch(headless=False, args=['--start-maximized'])
                self.context = self.browser.new_context(
                    viewport={'width': 1366, 'height': 768},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                )
                # Capture console errors
                console_errors = []
                self.page = self.context.new_page()
                self.page.on('console', lambda msg: console_errors.append(msg.text) if msg.type == 'error' else None)
                self.page.on('pageerror', lambda err: console_errors.append(str(err)))

                # Handle popups — if a new page opens, switch to it
                def handle_popup(popup):
                    log.info(f'Popup opened: {popup.url}')
                    self.page = popup  # Switch to the new page
                self.context.on('page', handle_popup)

                # Navigate to the URL (use domcontentloaded — networkidle hangs on SPAs)
                try:
                    self.page.goto(url, wait_until='domcontentloaded', timeout=30000)
                    self._safe_wait_for_load(8000)
                    self._log_step(f'Navigate to {url}', 'OK')
                except Exception as e:
                    self._log_step(f'Navigate to {url}', f'FAILED: {e}', bug=f'Website failed to load: {e}')
                    return self._build_report(url, user_profile, console_errors, screenshots_for_report)

                # Initial screenshot
                initial_ss = self._take_screenshot()
                if initial_ss:
                    screenshots_for_report.append(initial_ss)

                # AI-driven testing loop
                while self.step_count < self.max_steps:
                    # If current page died, try to recover from browser context
                    if not self._is_page_alive():
                        pages = self.context.pages
                        if pages:
                            self.page = pages[-1]  # Switch to last open page
                            log.info(f'Recovered to page: {self.page.url}')
                        else:
                            # No pages left — open a new one in same browser
                            try:
                                self.page = self.context.new_page()
                                self.page.goto(url, wait_until='domcontentloaded', timeout=20000)
                                log.info(f'Reopened page in same browser: {url}')
                            except Exception:
                                self._log_step('Loop ended', 'Browser closed — could not recover')
                                break

                    # Check consecutive failures — bail early
                    if self._consecutive_failures >= 3:
                        self._log_step('Loop ended', f'Too many consecutive failures ({self._consecutive_failures})')
                        break

                    page_context = self._get_page_context()
                    if not page_context.get('url'):
                        # Page context extraction failed — page is dead
                        self._log_step('Loop ended', 'Cannot read page — browser likely closed')
                        break

                    screenshot_b64 = self._take_screenshot()

                    # Throttle to reduce flickering — wait for page to settle
                    time.sleep(2)

                    # Build readable history with failures highlighted
                    history_lines = []
                    for e in self.test_log[-12:]:
                        status = '✓' if 'FAILED' not in e['result'] and 'SKIPPED' not in e['result'] else '✗'
                        history_lines.append(f"  {status} Step {e['step']}: {e['action']} → {e['result'][:120]}")
                    history = '\n'.join(history_lines)

                    # Build input/button tables for clarity
                    input_table = '\n'.join([
                        f"  [{inp['index']}] {inp['type'] or inp['tag']} | selector: {inp['selector']} | placeholder: {inp.get('placeholder','')} | label: {inp.get('label','')}"
                        for inp in page_context.get('inputs', [])
                    ]) or '  (no inputs found)'

                    button_table = '\n'.join([
                        f"  [{btn['index']}] {btn['tag']} | text: \"{btn['text']}\" | selector: {btn['selector']} | href: {btn.get('href','')[:80]}"
                        for btn in page_context.get('buttons', [])
                    ]) or '  (no buttons/links found)'

                    # Count consecutive failures for AI context
                    recent_failures = sum(1 for e in self.test_log[-5:] if 'FAILED' in e.get('result', ''))

                    ai_prompt = f"""You are an expert QA tester autonomously testing: {url}
Goal: {instructions}

TEST ACCOUNT (use for signup/login):
  Email: {user_profile['email']}
  Password: {user_profile['password']}
  Name: {user_profile['first_name']} {user_profile['last_name']}
  Username: {user_profile['username']}

══ CURRENT PAGE ══
URL: {page_context.get('url', '')}
Title: {page_context.get('title', '')}

══ FORM INPUTS (use the 'selector' field in your response) ══
{input_table}

══ BUTTONS & LINKS (use 'selector' or 'text' for clicking) ══
{button_table}

══ ERRORS ON PAGE ══
{json.dumps(page_context.get('errors', []))[:500]}

══ PAGE TEXT (first 800 chars) ══
{page_context.get('bodyText', '')[:800]}

══ TEST HISTORY (last 12 steps) ══
{history}

══ STATUS ══
Steps: {self.step_count}/{self.max_steps} | Recent failures: {recent_failures}/5 | Console errors: {len(console_errors)}
Steps on current URL: {self._url_visit_count.get(page_context.get('url', ''), 0)} (will auto-stop at 8)
Fields already filled: {json.dumps(self._filled_fields)[:300]}

{'⚠️ MULTIPLE RECENT FAILURES! You MUST try a COMPLETELY DIFFERENT approach:' + chr(10) + '  - If fill keeps failing, try clicking a submit/next button instead' + chr(10) + '  - If click keeps failing, try navigating directly to a URL' + chr(10) + '  - If nothing works on this page, use "done" action' + chr(10) + '  - Do NOT retry the same action with different values — that is a LOOP' if recent_failures >= 2 else ''}

Respond with ONLY a valid JSON object — no other text:
{{
  "action": "click|fill|select|press|navigate|wait|scroll|check_email|report_bug|done",
  "selector": "the CSS selector from the inputs/buttons tables above",
  "value": "text to fill, button text to click, URL to navigate, or key to press",
  "description": "what this action does and why",
  "bugs_noticed": ["any bugs or UX issues you see right now"]
}}

SELECTOR RULES:
- ALWAYS use the exact 'selector' value from the tables above — they are guaranteed to work
- For clicking links/buttons: set "selector" from the table. Also set "value" to the button text as fallback
- For fill: use the input's selector from the table
- If no selector matches, use "navigate" to go to a URL directly (e.g., /register, /signup)
- NEVER invent selectors — only use ones from the tables or standard patterns like a[href="/signup"]
- If the same action failed before, you MUST try a DIFFERENT ACTION TYPE (not just different value)
- Do NOT fill the same field twice with different values — that is a loop
- If you're stuck on a page (many failures), use "done" to finish testing
- Fill ALL form fields first, THEN click submit — do not keep re-filling the same field"""

                    system = (
                        'You are an autonomous QA testing agent. You analyze web pages and decide one action at a time. '
                        'You must respond with ONLY a valid JSON object. No markdown, no explanation, just JSON. '
                        'Use the exact CSS selectors provided in the tables — they are extracted from the live DOM. '
                        'Be strategic: if something fails, try a different approach.'
                    )

                    ai_response = self._ask_ai(ai_prompt, system)

                    if not ai_response:
                        self._log_step('AI decision', 'AI failed to respond')
                        self._consecutive_failures += 1
                        break

                    # Parse AI decision
                    try:
                        clean = ai_response.strip()
                        if clean.startswith('```'):
                            clean = clean.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
                        json_start = clean.find('{')
                        json_end = clean.rfind('}') + 1
                        if json_start >= 0 and json_end > json_start:
                            clean = clean[json_start:json_end]
                        action_data = json.loads(clean)
                    except json.JSONDecodeError:
                        log.warning(f'AI returned non-JSON: {ai_response[:200]}')
                        self._consecutive_failures += 1
                        if self._consecutive_failures >= 2:
                            break
                        continue

                    # Log bugs the AI noticed
                    for bug in action_data.get('bugs_noticed', []):
                        if bug and isinstance(bug, str) and len(bug.strip()) > 3:
                            self.bugs.append({
                                'step': self.step_count + 1,
                                'description': bug,
                                'url': page_context.get('url', ''),
                                'timestamp': datetime.now().isoformat(),
                            })

                    # Execute the action
                    should_continue = self._execute_action(action_data)

                    # Save screenshot after navigation-type actions
                    if action_data.get('action') in ('click', 'navigate', 'press', 'check_email') and self._is_page_alive():
                        time.sleep(1.5)
                        ss = self._take_screenshot()
                        if ss:
                            screenshots_for_report.append(ss)
                            if len(screenshots_for_report) > 15:
                                screenshots_for_report = screenshots_for_report[-15:]

                    if not should_continue:
                        break

                    time.sleep(1)

                # Final screenshot
                final_ss = self._take_screenshot()
                if final_ss:
                    screenshots_for_report.append(final_ss)

                self.browser.close()

        except Exception as e:
            log.error(f'Web test error: {e}')
            self._log_step('Test engine', f'Fatal error: {e}', bug=f'Test engine crashed: {e}')

        return self._build_report(url, user_profile, console_errors if 'console_errors' in locals() else [], screenshots_for_report)

    def _build_report(self, url, user_profile, console_errors, screenshots):
        """Build the final test report using AI summarization."""
        # Compile raw data
        steps_text = '\n'.join([
            f"Step {e['step']}: {e['action']} → {e['result']}" + (f" [BUG: {e.get('bug', '')}]" if e.get('bug') else '')
            for e in self.test_log
        ])

        bugs_text = '\n'.join([
            f"• [{b['step']}] {b['description']} (at {b['url']})"
            for b in self.bugs
        ]) or 'No bugs found.'

        console_text = '\n'.join(console_errors[-10:]) if console_errors else 'No console errors.'

        # Ask AI to write a professional summary
        summary_prompt = f"""Write a concise QA test report for this website test.

WEBSITE: {url}
TEST EMAIL USED: {user_profile['email']}
TOTAL STEPS: {self.step_count}
TOTAL BUGS FOUND: {len(self.bugs)}

ALL TEST STEPS:
{steps_text}

BUGS FOUND:
{bugs_text}

CONSOLE ERRORS:
{console_text}

Write a professional but concise report with these sections:
1. **Summary** — 2-3 sentence overview
2. **Test Flow** — What was tested (bullet points)
3. **Bugs & Issues** — Each bug with severity (Critical/High/Medium/Low)
4. **Console Errors** — Any JS errors found
5. **UX Observations** — Usability issues, missing features
6. **Overall Score** — Rate the site /10 with brief justification

Keep it practical and actionable."""

        summary = self._ask_ai(summary_prompt, 'You are a senior QA engineer writing a test report. Be precise and professional.', max_tokens=3000)

        if not summary:
            # Fallback: build manual summary
            summary = f"""## Web Test Report: {url}

**Steps Completed:** {self.step_count}
**Bugs Found:** {len(self.bugs)}

### Test Steps:
{steps_text}

### Bugs:
{bugs_text}

### Console Errors:
{console_text}"""

        report = {
            'url': url,
            'summary': summary,
            'total_steps': self.step_count,
            'total_bugs': len(self.bugs),
            'bugs': self.bugs,
            'steps': [{'step': e['step'], 'action': e['action'], 'result': e['result']} for e in self.test_log],
            'console_errors': console_errors[-10:] if console_errors else [],
            'test_email': user_profile['email'],
            'timestamp': datetime.now().isoformat(),
        }

        # Include last screenshot for display
        if screenshots:
            report['screenshot'] = screenshots[-1]

        return report
