import requests
import json
import os
import sqlite3
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from mistralai import Mistral

# Настройка логирования
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_KEY = os.environ.get("MISTRAL_API_KEY")
MODEL = "mistral-large-latest"

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Инициализация базы данных
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# Создание пользовательских папок
def create_user_folder(username, user_id):
    folder_name = f"{username}@{user_id}"
    folder_path = os.path.join("user_data", folder_name)
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(os.path.join(folder_path, "saves"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "characters"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "chats"), exist_ok=True)
    return folder_path

def get_user_folder(username, user_id):
    folder_name = f"{username}@{user_id}"
    return os.path.join("user_data", folder_name)

# Проверка аутентификации
def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "Требуется авторизация", "need_login": True})
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

class ContextManager:
    def __init__(self, max_tokens=8000, summary_threshold=15):
        self.max_tokens = max_tokens
        self.summary_threshold = summary_threshold

    def estimate_tokens(self, text):
        """Примерная оценка количества токенов (1 токен ≈ 4 символа для русского)"""
        return len(text) // 3

    def create_summary(self, messages):
        """Создает краткое резюме старых сообщений"""
        summary_messages = []
        for msg in messages:
            if msg["role"] == "user":
                summary_messages.append(f"Игрок: {msg['content'][:100]}...")
            else:
                summary_messages.append(f"ГМ: {msg['content'][:200]}...")

        return {
            "role": "system", 
            "content": f"РЕЗЮМЕ ПРЕДЫДУЩИХ СОБЫТИЙ:\n" + "\n".join(summary_messages[-5:])
        }

    def optimize_context(self, conversation_history):
        """Оптимизирует контекст, балансируя полноту и размер"""
        if len(conversation_history) <= 6:
            return conversation_history

        total_tokens = sum(self.estimate_tokens(msg["content"]) for msg in conversation_history)

        if total_tokens <= self.max_tokens:
            return conversation_history

        # Сохраняем последние важные сообщения
        recent_messages = conversation_history[-8:]
        older_messages = conversation_history[:-8]

        # Создаем резюме старых сообщений
        if older_messages:
            summary = self.create_summary(older_messages)
            return [summary] + recent_messages

        return recent_messages

def load_gm_rules():
    """Загружает правила ГМ из JSON файла"""
    try:
        with open("attached_assets/2. Правила для гейм мастера_1751298976539.json", "r", encoding="utf-8") as f:
            rules = json.load(f)
        return rules
    except FileNotFoundError:
        print("Предупреждение: Файл с правилами ГМ не найден")
        return None

def create_gm_system_prompt(rules):
    """Создает системный промпт для ГМ на основе правил"""
    if not rules:
        return ""

    system_prompt = """Ты - Гейм Мастер (ГМ) для нарративной ролевой игры. Твоя задача - вести игру согласно следующим правилам:

ПРАВИЛА ИГРЫ:
"""

    # Добавляем правила из JSON
    for category, content in rules["rules"].items():
        system_prompt += f"\n{category.upper().replace('_', ' ')}:\n"
        if isinstance(content, dict):
            for key, value in content.items():
                if isinstance(value, list):
                    system_prompt += f"- {key}: {', '.join(value)}\n"
                else:
                    system_prompt += f"- {key}: {value}\n"
        else:
            system_prompt += f"- {content}\n"

    system_prompt += """
ВАЖНО:
- Если игрок НЕ использует тег 'ГМ:', ты можешь только описывать мир, NPC и их действия
- НЕ предлагай варианты действий, если игрок их не запрашивает
- Мир живет своей жизнью независимо от игрока
- Все описания могут быть максимально детальными без ограничений
- Согласие всегда подразумевается по умолчанию

Отвечай кратко и по существу для веб-интерфейса."""

    return system_prompt

def process_content(content):
    # Более аккуратная обработка тегов мышления
    import re
    # Удаляем только содержимое между тегами <think>...</think>, сохраняя остальной текст
    cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
    # Также удаляем пустые строки и лишние пробелы
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    cleaned = cleaned.strip()

    # Если после обработки контент стал пустым или очень коротким, возвращаем исходный
    if not cleaned or len(cleaned) < 10:
        return content

    return cleaned

def chat_with_ai(prompt, system_prompt="", conversation_history=[]):
    if not API_KEY:
        return "🔑 **Ошибка**: API ключ Mistral не найден. Добавьте MISTRAL_API_KEY в Secrets."

    try:
        client = Mistral(api_key=API_KEY)

        # Оптимизируем контекст
        context_manager = ContextManager()
        optimized_history = context_manager.optimize_context(conversation_history)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.extend(optimized_history)
        messages.append({"role": "user", "content": prompt})

        chat_response = client.chat.complete(
            model=MODEL,
            messages=messages
        )

        content = chat_response.choices[0].message.content
        processed_content = process_content(content)

        return processed_content

    except Exception as e:
        return f"Ошибка Mistral AI: {str(e)}"

# Веб-интерфейс
@app.route('/')
def index():
    if 'user_id' in session:
        return render_template('game.html')
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    if not username or not password:
        return jsonify({"error": "Логин и пароль не могут быть пустыми"})

    if len(username) < 3:
        return jsonify({"error": "Логин должен содержать минимум 3 символа"})

    if len(password) < 6:
        return jsonify({"error": "Пароль должен содержать минимум 6 символов"})

    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()

        # Проверяем, существует ли пользователь
        c.execute("SELECT id FROM users WHERE username = ?", (username,))
        if c.fetchone():
            conn.close()
            return jsonify({"error": "Пользователь с таким логином уже существует"})

        # Создаем пользователя
        password_hash = generate_password_hash(password)
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                 (username, password_hash))
        user_id = c.lastrowid
        conn.commit()
        conn.close()

        # Создаем папку пользователя
        create_user_folder(username, user_id)

        # Логинимся
        session['user_id'] = user_id
        session['username'] = username

        return jsonify({"success": True, "message": "Регистрация успешна!"})

    except Exception as e:
        return jsonify({"error": f"Ошибка регистрации: {str(e)}"})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    remember_me = data.get('remember_me', False)

    if not username or not password:
        return jsonify({"error": "Логин и пароль не могут быть пустыми"})

    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()

        if not user or not check_password_hash(user[1], password):
            return jsonify({"error": "Неверный логин или пароль"})

        session['user_id'] = user[0]
        session['username'] = username
        
        # Устанавливаем срок жизни сессии в зависимости от чекбокса
        if remember_me:
            # Сессия на 30 дней
            session.permanent = True
            app.permanent_session_lifetime = timedelta(days=30)
        else:
            # Сессия до закрытия браузера
            session.permanent = False

        return jsonify({"success": True, "message": "Вход выполнен успешно!"})

    except Exception as e:
        return jsonify({"error": f"Ошибка входа: {str(e)}"})

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"success": True})

@app.route('/get_user_info', methods=['GET'])
@login_required
def get_user_info():
    return jsonify({
        "username": session['username'],
        "user_id": session['user_id']
    })

@app.route('/get_saves', methods=['GET'])
@login_required
def get_saves():
    """Получает список сохранений пользователя"""
    user_folder = get_user_folder(session['username'], session['user_id'])
    saves_folder = os.path.join(user_folder, "saves")

    saves = []
    if os.path.exists(saves_folder):
        for filename in os.listdir(saves_folder):
            if filename.endswith('.json'):
                filepath = os.path.join(saves_folder, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        save_data = json.load(f)
                    saves.append({
                        "filename": filename[:-5],  # убираем .json
                        "timestamp": save_data.get('timestamp', 'Неизвестно'),
                        "character_name": save_data.get('character_name', 'Неизвестный персонаж')
                    })
                except:
                    continue

    return jsonify({"saves": saves})

@app.route('/get_characters', methods=['GET'])
@login_required
def get_characters():
    """Получает список персонажей пользователя"""
    user_folder = get_user_folder(session['username'], session['user_id'])
    characters_folder = os.path.join(user_folder, "characters")

    characters = []
    if os.path.exists(characters_folder):
        for filename in os.listdir(characters_folder):
            if filename.endswith('.json'):
                filepath = os.path.join(characters_folder, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        char_data = json.load(f)
                    characters.append({
                        "filename": filename[:-5],  # убираем .json
                        "name": char_data.get('name', filename[:-5]),
                        "description": char_data.get('description', '')[:100] + '...'
                    })
                except:
                    continue

    return jsonify({"characters": characters})

@app.route('/get_chats', methods=['GET'])
@login_required
def get_chats():
    """Получает список чатов пользователя с информацией о персонажах"""
    user_folder = get_user_folder(session['username'], session['user_id'])
    chats_folder = os.path.join(user_folder, "chats")

    if not os.path.exists(chats_folder):
        os.makedirs(chats_folder, exist_ok=True)

    chats = {}
    for filename in os.listdir(chats_folder):
        if filename.endswith('.json'):
            filepath = os.path.join(chats_folder, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    chat_data = json.load(f)
                
                chat_id = filename[:-5]  # убираем .json
                
                # Добавляем информацию о персонаже для UI
                character_desc, character_name = get_chat_character(chat_data)
                if character_name:
                    chat_data['character_name'] = character_name
                
                chats[chat_id] = chat_data
            except:
                continue

    # Если нет чатов, создаем основной
    if not chats:
        default_chat = {
            "name": "Основной чат",
            "messages": [],
            "character_id": None,
            "created_at": datetime.now().isoformat()
        }
        chats['default'] = default_chat
        save_chat_file('default', default_chat)

    return jsonify({"chats": chats})

# УБИРАЕМ ИЗБЫТОЧНУЮ ФУНКЦИЮ save_chat - теперь сохранение только при необходимости
def save_chat_file(chat_id, chat_data):
    """Сохраняет файл чата (только когда реально нужно)"""
    try:
        user_folder = get_user_folder(session['username'], session['user_id'])
        chats_folder = os.path.join(user_folder, "chats")
        os.makedirs(chats_folder, exist_ok=True)

        filepath = os.path.join(chats_folder, f"{chat_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Ошибка сохранения чата: {e}")

@app.route('/create_chat', methods=['POST'])
@login_required
def create_chat():
    """Создает новый чат"""
    data = request.get_json()
    chat_name = data.get('chat_name', 'Новый чат')
    chat_id = data.get('chat_id', f"chat_{int(datetime.now().timestamp())}")

    chat_data = {
        "name": chat_name,
        "messages": [],
        "character": session.get('character'),
        "character_name": session.get('character_name'),
        "created_at": datetime.now().isoformat()
    }

    save_chat_file(chat_id, chat_data)
    return jsonify({"success": True, "chat_id": chat_id, "chat_data": chat_data})

@app.route('/delete_chat', methods=['POST'])
@login_required
def delete_chat():
    """Удаляет чат"""
    data = request.get_json()
    chat_id = data.get('chat_id')

    if not chat_id:
        return jsonify({"error": "ID чата не указан"})

    user_folder = get_user_folder(session['username'], session['user_id'])
    filepath = os.path.join(user_folder, "chats", f"{chat_id}.json")

    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({"success": True, "message": "Чат удален"})
        else:
            return jsonify({"error": "Чат не найден"})
    except Exception as e:
        return jsonify({"error": f"Ошибка удаления чата: {str(e)}"})

@app.route('/start_game', methods=['POST'])
@login_required
def start_game():
    if not API_KEY:
        return jsonify({"error": "API ключ не найден. Добавьте MISTRAL_API_KEY в переменные окружения."})

    data = request.get_json()
    chat_id = data.get('chat_id', 'default')
    character = data.get('character')  # Персонаж может быть передан сразу

    rules = load_gm_rules()
    system_prompt = create_gm_system_prompt(rules)

    session['conversation_history'] = []
    session['system_prompt'] = system_prompt
    session['current_chat_id'] = chat_id

    # Если персонаж передан, используем его
    if character:
        session['character'] = character
        # Сразу начинаем игру с персонажем
        response = chat_with_ai(f"Начни захватывающее приключение для персонажа: {character}", system_prompt, [])
    else:
        # Загружаем данные чата
        chat_data = load_chat_data(chat_id)
        if chat_data and chat_data.get('character'):
            session['character'] = chat_data['character']
            character = chat_data['character']
            response = chat_with_ai(f"Начни захватывающее приключение для персонажа: {character}", system_prompt, [])
        else:
            # Нет персонажа - просим создать или загрузить
            response = "🎭 **Добро пожаловать в игру!**\n\nПрежде чем начать, выберите персонажа из списка или создайте нового."
            session['character'] = None

    if response and response.strip():
        session['conversation_history'] = [
            {"role": "user", "content": "Начни игру"},
            {"role": "assistant", "content": response}
        ]

        # Сохраняем в чат только если есть реальные изменения
        update_chat_messages(chat_id, [
            {"role": "user", "content": "Начни игру", "timestamp": datetime.now().isoformat()},
            {"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()}
        ])

    return jsonify({"response": response, "game_started": bool(character)})

def load_chat_data(chat_id):
    """Загружает данные чата"""
    try:
        user_folder = get_user_folder(session['username'], session['user_id'])
        filepath = os.path.join(user_folder, "chats", f"{chat_id}.json")

        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Ошибка загрузки чата: {e}")
    return None

def update_chat_messages(chat_id, messages):
    """Обновляет сообщения в чате (ТОЛЬКО при реальных изменениях)"""
    try:
        chat_data = load_chat_data(chat_id)
        if not chat_data:
            chat_data = {
                "name": f"Чат {chat_id}",
                "messages": [],
                "character": session.get('character'),
                "character_name": None,
                "created_at": datetime.now().isoformat()
            }

        # Проверяем, есть ли реально новые сообщения
        old_count = len(chat_data['messages'])
        chat_data['messages'].extend(messages)

        # Сохраняем только если действительно что-то изменилось
        if len(chat_data['messages']) > old_count:
            save_chat_file(chat_id, chat_data)
    except Exception as e:
        print(f"Ошибка обновления чата: {e}")

@app.route('/send_message', methods=['POST'])
@login_required
def send_message():
    data = request.get_json()
    user_message = data.get('message', '')
    chat_id = data.get('chat_id', 'default')

    logger.debug(f"send_message вызван: message='{user_message}', chat_id='{chat_id}'")

    if not user_message:
        logger.warning("Попытка отправить пустое сообщение")
        return jsonify({"error": "Пустое сообщение"})

    session['current_chat_id'] = chat_id

    # Проверяем, находимся ли в режиме создания персонажа
    if session.get('character_creation_mode'):
        logger.debug("В режиме создания персонажа")
        return create_character_continue(user_message, chat_id)

    # Проверяем, запрашивает ли игрок создание персонажа
    if 'создать персонажа' in user_message.lower() or 'создание персонажа' in user_message.lower():
        logger.debug("Запрос на создание персонажа")
        return create_character_start(chat_id)

    # Проверяем персонажа в чате
    chat_data = load_chat_data(chat_id)
    chat_character, chat_character_name = get_chat_character(chat_data)
    
    logger.debug(f"Проверка персонажа в чате: {bool(chat_character)}")

    # Проверяем, есть ли персонаж
    if not chat_character:
        logger.warning("Персонаж не найден в чате")
        return jsonify({
            "response": "⚠️ Сначала нужно создать или загрузить персонажа! Напишите 'создать персонажа' или выберите персонажа из списка."
        })

    conversation_history = session.get('conversation_history', [])
    system_prompt = session.get('system_prompt', '')

    # Добавляем информацию о персонаже в контекст
    enhanced_prompt = f"{user_message}\n\n[ПЕРСОНАЖ ИГРОКА: {chat_character}]"

    response = chat_with_ai(enhanced_prompt, system_prompt, conversation_history)

    if response and response.strip():
        conversation_history.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response}
        ])
        session['conversation_history'] = conversation_history

        # Сохраняем в чат
        update_chat_messages(chat_id, [
            {"role": "user", "content": user_message, "timestamp": datetime.now().isoformat()},
            {"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()}
        ])

    return jsonify({"response": response})

@app.route('/edit_message', methods=['POST'])
@login_required
def edit_message():
    """Редактирует сообщение и генерирует новый ответ ИИ"""
    data = request.get_json()
    message_id = data.get('message_id')
    new_content = data.get('new_content', '')
    chat_id = data.get('chat_id', 'default')
    
    if not new_content:
        return jsonify({"error": "Пустое сообщение"})
    
    conversation_history = session.get('conversation_history', [])
    system_prompt = session.get('system_prompt', '')
    
    # Обрезаем историю до редактируемого сообщения
    if message_id < len(conversation_history):
        conversation_history = conversation_history[:message_id]
        conversation_history.append({"role": "user", "content": new_content})
    
    # Добавляем информацию о персонаже в контекст
    character_info = session.get('character')
    if character_info:
        enhanced_prompt = f"{new_content}\n\n[ПЕРСОНАЖ ИГРОКА: {character_info}]"
    else:
        enhanced_prompt = new_content
    
    response = chat_with_ai(enhanced_prompt, system_prompt, conversation_history[:-1])
    
    if response and response.strip():
        conversation_history.append({"role": "assistant", "content": response})
        session['conversation_history'] = conversation_history
        
        # Обновляем чат
        chat_data = load_chat_data(chat_id)
        if chat_data:
            # Обрезаем сообщения в чате и добавляем новые
            if message_id < len(chat_data['messages']):
                chat_data['messages'] = chat_data['messages'][:message_id]
            
            chat_data['messages'].extend([
                {"role": "user", "content": new_content, "timestamp": datetime.now().isoformat()},
                {"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()}
            ])
            save_chat_file(chat_id, chat_data)
    
    return jsonify({"response": response})

def create_character_start(chat_id='default'):
    """Начинает процесс создания персонажа"""
    session['character_creation_history'] = []
    session['character_creation_mode'] = True
    session['current_chat_id'] = chat_id

    response = """🎭 **СОЗДАНИЕ ПЕРСОНАЖА**

Отлично! Давайте создадим вашего персонажа. Я задам вам несколько вопросов, чтобы лучше понять, кого вы хотите играть.

**Первый вопрос:** Как зовут вашего персонажа и в каком мире или сеттинге вы хотели бы играть? (фэнтези, современность, киберпанк, космос и т.д.)"""

    return jsonify({
        "response": response,
        "character_creation": True
    })

def create_character_continue(user_input, chat_id='default'):
    """Продолжает процесс создания персонажа"""
    system_prompt = session.get('system_prompt', '')
    creation_history = session.get('character_creation_history', [])

    # Специальный промпт для создания персонажа
    character_creation_prompt = f"""
{system_prompt}

РЕЖИМ СОЗДАНИЯ ПЕРСОНАЖА:
Ты помогаешь игроку создать персонажа. Задавай вопросы о:
- Имени и внешности
- Предыстории и характере  
- Навыках и способностях
- Снаряжении и особенностях

Когда персонаж будет готов (после 4-5 вопросов), заверши описанием в формате:
=== ПЕРСОНАЖ СОЗДАН ===
Имя: [имя]
[Полное описание персонажа]
=== КОНЕЦ ОПИСАНИЯ ===
"""

    response = chat_with_ai(user_input, character_creation_prompt, creation_history)

    # Проверяем, завершено ли создание персонажа
    if "=== ПЕРСОНАЖ СОЗДАН ===" in response:
        # Извлекаем описание персонажа
        start_marker = "=== ПЕРСОНАЖ СОЗДАН ==="
        end_marker = "=== КОНЕЦ ОПИСАНИЯ ==="

        start_idx = response.find(start_marker) + len(start_marker)
        end_idx = response.find(end_marker)

        if end_idx > start_idx:
            character_description = response[start_idx:end_idx].strip()

            # Извлекаем имя персонажа
            character_name = "Безымянный"
            lines = character_description.split('\n')
            for line in lines:
                if line.startswith('Имя:'):
                    character_name = line.replace('Имя:', '').strip()
                    break

            session['character'] = character_description
            session['character_creation_mode'] = False
            session.pop('character_creation_history', None)

            # Сохраняем персонажа и получаем его ID
            character_id = save_character_to_file(character_description, character_name)

            # Обновляем чат с ID персонажа
            chat_data = load_chat_data(chat_id)
            if chat_data:
                chat_data['character_id'] = character_id
                # Удаляем старые поля
                chat_data.pop('character', None)
                chat_data.pop('character_name', None)
                save_chat_file(chat_id, chat_data)

            # Сохраняем в чат
            update_chat_messages(chat_id, [
                {"role": "user", "content": user_input, "timestamp": datetime.now().isoformat()},
                {"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()}
            ])

            return jsonify({
                "response": response,
                "character_created": True,
                "character": character_description,
                "character_name": character_name
            })

    # Продолжаем процесс создания
    creation_history.extend([
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response}
    ])
    session['character_creation_history'] = creation_history

    # Сохраняем в чат
    update_chat_messages(chat_id, [
        {"role": "user", "content": user_input, "timestamp": datetime.now().isoformat()},
        {"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()}
    ])

    return jsonify({"response": response, "character_created": False})

def save_character_to_file(character_description, character_name=None):
    """Сохраняет персонажа в файл"""
    try:
        if not character_name:
            # Извлекаем имя персонажа
            lines = character_description.split('\n')
            character_name = "Безымянный"
            for line in lines:
                if line.startswith('Имя:'):
                    character_name = line.replace('Имя:', '').strip()
                    break

        user_folder = get_user_folder(session['username'], session['user_id'])
        characters_folder = os.path.join(user_folder, "characters")

        # Генерируем уникальный ID
        character_id = f"char_{int(datetime.now().timestamp() * 1000)}"

        character_data = {
            "id": character_id,
            "name": character_name,
            "description": character_description,
            "created_at": datetime.now().isoformat()
        }

        # Создаем безопасное имя файла БЕЗ системных цифр
        safe_name = "".join(c for c in character_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_name}.json"

        with open(os.path.join(characters_folder, filename), 'w', encoding='utf-8') as f:
            json.dump(character_data, f, ensure_ascii=False, indent=2)

        return character_id  # Возвращаем ID персонажа

    except Exception as e:
        print(f"Ошибка сохранения персонажа: {e}")
        return None

def get_character_by_id(character_id):
    """Загружает персонажа по ID из файла"""
    try:
        user_folder = get_user_folder(session['username'], session['user_id'])
        characters_folder = os.path.join(user_folder, "characters")
        
        # Ищем файл с нужным ID
        for filename in os.listdir(characters_folder):
            if filename.endswith('.json'):
                filepath = os.path.join(characters_folder, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    char_data = json.load(f)
                
                if char_data.get('id') == character_id:
                    return char_data
        
        return None
    except Exception as e:
        logger.error(f"Ошибка загрузки персонажа по ID {character_id}: {e}")
        return None

def get_chat_character(chat_data):
    """Получает полные данные персонажа для чата"""
    if not chat_data:
        return None, None
    
    character_id = chat_data.get('character_id')
    if character_id and character_id != 'None':
        char_data = get_character_by_id(character_id)
        if char_data:
            return char_data['description'], char_data['name']
    
    # Для обратной совместимости со старыми чатами
    old_character = chat_data.get('character')
    old_name = chat_data.get('character_name')
    if old_character:
        return old_character, old_name
    
    return None, None

@app.route('/load_character', methods=['POST'])
@login_required
def load_character():
    """Загружает персонажа из сохраненных и сохраняет ID в чат"""
    data = request.get_json()
    filename = data.get('filename')
    chat_id = data.get('chat_id', 'default')

    logger.debug(f"load_character вызван: filename='{filename}', chat_id='{chat_id}'")

    if not filename:
        logger.error("Не указано имя файла персонажа")
        return jsonify({"error": "Не указано имя файла"})

    # Проверяем, есть ли уже персонаж в текущем чате
    chat_data = load_chat_data(chat_id)
    if chat_data and chat_data.get('character_id'):
        logger.warning(f"Персонаж уже выбран для чата {chat_id}")
        return jsonify({"error": "Персонаж для этой истории уже выбран"})

    user_folder = get_user_folder(session['username'], session['user_id'])
    filepath = os.path.join(user_folder, "characters", f"{filename}.json")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            character_data = json.load(f)

        character_id = character_data.get('id')
        character_name = character_data.get('name', filename)
        character_description = character_data['description']

        logger.debug(f"Загружен персонаж: {character_name} (ID: {character_id})")

        # Получаем ID персонажа, если его нет - создаем
        if not character_id:
            character_id = f"char_{int(datetime.now().timestamp() * 1000)}"
            # Пересохраняем персонажа с новым ID
            with open(filepath, 'w', encoding='utf-8') as f:
                character_data['id'] = character_id
                json.dump(character_data, f, ensure_ascii=False, indent=2)

        # Сохраняем только ID персонажа в чат
        if not chat_data:
            chat_data = {
                "name": f"Чат {character_name}",
                "messages": [],
                "character_id": character_id,
                "created_at": datetime.now().isoformat()
            }
        else:
            chat_data['character_id'] = character_id
            # Удаляем старые поля для совместимости
            chat_data.pop('character', None)
            chat_data.pop('character_name', None)

        save_chat_file(chat_id, chat_data)

        # Загружаем правила ГМ для будущего использования
        rules = load_gm_rules()
        system_prompt = create_gm_system_prompt(rules)
        session['system_prompt'] = system_prompt

        logger.info(f"Персонаж '{character_name}' (ID: {character_id}) успешно привязан к чату {chat_id}")

        return jsonify({
            "success": True,
            "character": character_description,
            "character_name": character_name,
            "message": f"Персонаж '{character_name}' выбран"
        })

    except FileNotFoundError:
        return jsonify({"error": "Файл персонажа не найден"})
    except Exception as e:
        return jsonify({"error": f"Ошибка загрузки персонажа: {str(e)}"})

def create_chat_name_from_response(response):
    """Создает название чата из первых слов ответа ИИ"""
    import re
    # Убираем разметку и получаем первые слова
    clean_response = re.sub(r'[*#_\-\[\]()]', '', response)
    words = clean_response.split()[:4]  # Берем первые 4 слова
    chat_name = ' '.join(words)
    
    # Ограничиваем длину
    if len(chat_name) > 30:
        chat_name = chat_name[:27] + '...'
    
    return chat_name or "Новое приключение"

@app.route('/start_game_with_character', methods=['POST'])
@login_required
def start_game_with_character():
    """Начинает игру с уже выбранным персонажем"""
    data = request.get_json()
    chat_id = data.get('chat_id', 'default')

    logger.debug(f"start_game_with_character вызван для chat_id='{chat_id}'")

    # Загружаем данные чата для проверки персонажа
    chat_data = load_chat_data(chat_id)
    character, character_name = get_chat_character(chat_data)
    
    if not character:
        logger.error(f"Персонаж не найден в чате {chat_id}")
        return jsonify({"error": "Персонаж не выбран"})
    
    logger.info(f"Начинаем игру с персонажем: {character_name}")
    
    # Загружаем правила ГМ
    rules = load_gm_rules()
    system_prompt = create_gm_system_prompt(rules)
    session['system_prompt'] = system_prompt

    # Начинаем игру
    enhanced_prompt = f"Начни игру\n\n[ПЕРСОНАЖ ИГРОКА: {character}]"
    response = chat_with_ai(enhanced_prompt, system_prompt, [])

    if response and response.strip():
        # Создаем название чата из первых слов ответа
        chat_name = create_chat_name_from_response(response)
        
        # Обновляем данные чата
        chat_data['name'] = chat_name

        # Добавляем сообщения
        messages = [
            {"role": "user", "content": "Начни игру", "timestamp": datetime.now().isoformat()},
            {"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()}
        ]
        chat_data['messages'] = messages

        # Сохраняем чат
        save_chat_file(chat_id, chat_data)
        
        # Обновляем сессию
        session['conversation_history'] = messages

        return jsonify({
            "success": True,
            "response": response,
            "chat_name": chat_name,
            "game_started": True
        })

    return jsonify({"error": "Не удалось начать игру"})

@app.route('/save_game', methods=['POST'])
@login_required
def save_game():
    """Сохраняет текущую игру"""
    data = request.get_json()
    save_name = data.get('save_name', f"save_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    chat_id = data.get('chat_id', 'default')

    # Загружаем данные из чата
    chat_data = load_chat_data(chat_id)
    if not chat_data:
        return jsonify({"error": "Данные чата не найдены"})

    conversation_history = chat_data.get('messages', [])
    character = chat_data.get('character')
    character_name = chat_data.get('character_name', "Неизвестный персонаж")

    save_data = {
        "timestamp": datetime.now().isoformat(),
        "conversation_history": conversation_history,
        "character": character,
        "character_name": character_name,
        "save_name": save_name,
        "chat_id": chat_id
    }

    user_folder = get_user_folder(session['username'], session['user_id'])
    saves_folder = os.path.join(user_folder, "saves")
    
    # Создаем папку saves если она не существует
    os.makedirs(saves_folder, exist_ok=True)
    
    save_path = os.path.join(saves_folder, f"{save_name}.json")

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        return jsonify({"success": True, "message": "Игра сохранена"})
    except Exception as e:
        return jsonify({"error": f"Ошибка сохранения: {str(e)}"})

@app.route('/load_game', methods=['POST'])
@login_required
def load_game():
    """Загружает сохраненную игру"""
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({"error": "Не указано имя файла"})

    user_folder = get_user_folder(session['username'], session['user_id'])
    save_path = os.path.join(user_folder, "saves", f"{filename}.json")

    try:
        with open(save_path, "r", encoding="utf-8") as f:
            save_data = json.load(f)

        session['conversation_history'] = save_data.get('conversation_history', [])
        session['character'] = save_data.get('character', None)

        return jsonify({
            "success": True,
            "message": "Игра загружена",
            "timestamp": save_data.get('timestamp'),
            "character": save_data.get('character'),
            "character_name": save_data.get('character_name'),
            "history": save_data.get('conversation_history', [])
        })

    except FileNotFoundError:
        return jsonify({"error": "Файл сохранения не найден"})
    except Exception as e:
        return jsonify({"error": f"Ошибка загрузки: {str(e)}"})

@app.route('/delete_character', methods=['POST'])
@login_required
def delete_character():
    """Удаляет персонажа"""
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({"error": "Не указано имя файла"})

    user_folder = get_user_folder(session['username'], session['user_id'])
    filepath = os.path.join(user_folder, "characters", f"{filename}.json")

    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({"success": True, "message": "Персонаж удален"})
        else:
            return jsonify({"error": "Файл персонажа не найден"})
    except Exception as e:
        return jsonify({"error": f"Ошибка удаления персонажа: {str(e)}"})

@app.route('/get_character_by_id', methods=['POST'])
@login_required
def get_character_by_id_route():
    """Получает персонажа по ID"""
    data = request.get_json()
    character_id = data.get('character_id')

    if not character_id:
        return jsonify({"error": "ID персонажа не указан"})

    character_data = get_character_by_id(character_id)
    if character_data:
        return jsonify({
            "success": True,
            "character": character_data['description'],
            "character_name": character_data['name']
        })
    else:
        return jsonify({"error": "Персонаж не найден"})

@app.route('/delete_save', methods=['POST'])
@login_required
def delete_save():
    """Удаляет сохранение"""
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({"error": "Не указано имя файла"})

    user_folder = get_user_folder(session['username'], session['user_id'])
    filepath = os.path.join(user_folder, "saves", f"{filename}.json")

    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({"success": True, "message": "Сохранение удалено"})
        else:
            return jsonify({"error": "Файл сохранения не найден"})
    except Exception as e:
        return jsonify({"error": f"Ошибка удаления сохранения: {str(e)}"})

@app.route('/upload_character', methods=['POST'])
@login_required
def upload_character():
    """Загружает файл персонажа с пользовательским именем"""
    data = request.get_json()
    file_content = data.get('file_content')
    character_name = data.get('character_name', '').strip()

    if not file_content:
        return jsonify({"error": "Содержимое файла не получено"})

    if not character_name:
        return jsonify({"error": "Имя персонажа не указано"})

    try:
        # Пытаемся распарсить как JSON
        try:
            character_data = json.loads(file_content)
            # Создаем читаемое описание персонажа
            character_description = format_character_description(character_data)
        except json.JSONDecodeError:
            # Если не JSON, используем как текст
            character_description = file_content

        # Сохраняем персонажа с указанным именем
        filename = save_character_to_file(character_description, character_name)

        if filename:
            return jsonify({
                "success": True, 
                "character": character_description,
                "character_name": character_name,
                "filename": filename,
                "message": f"Персонаж '{character_name}' сохранен успешно"
            })
        else:
            return jsonify({"error": "Ошибка сохранения персонажа"})

    except Exception as e:
        return jsonify({"error": f"Ошибка при обработке файла: {str(e)}"})

def format_character_description(character_data):
    """Форматирует данные персонажа из JSON в читаемый текст"""
    if isinstance(character_data, dict):
        description = "=== ПЕРСОНАЖ ===\n"

        # Основная информация
        if 'name' in character_data:
            description += f"Имя: {character_data['name']}\n"
        if 'race' in character_data:
            description += f"Раса: {character_data['race']}\n"
        if 'class' in character_data:
            description += f"Класс: {character_data['class']}\n"
        if 'level' in character_data:
            description += f"Уровень: {character_data['level']}\n"

        # Характеристики
        if 'stats' in character_data:
            description += "\nХарактеристики:\n"
            for stat, value in character_data['stats'].items():
                description += f"- {stat}: {value}\n"

        # Навыки
        if 'skills' in character_data:
            description += "\nНавыки:\n"
            for skill in character_data['skills']:
                description += f"- {skill}\n"

        # Снаряжение
        if 'equipment' in character_data:
            description += "\nСнаряжение:\n"
            for item in character_data['equipment']:
                description += f"- {item}\n"

        # Предыстория
        if 'background' in character_data:
            description += f"\nПредыстория: {character_data['background']}\n"

        return description

    return str(character_data)

# Инициализация при запуске
init_db()
os.makedirs("user_data", exist_ok=True)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'web':
        print("🌐 Запуск веб-сервера на http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("🌐 Запуск веб-сервера на http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)