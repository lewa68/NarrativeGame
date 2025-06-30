
import pip
pip.main(['install', 'flask'])
pip.main(['install', 'werkzeug'])
import requests
import json
import os
import sqlite3
import hashlib
import secrets
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "deepseek/deepseek-r1"

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
    
    # Логирование для отладки
    if len(content) > len(cleaned) + 100:  # Если удалили много текста
        print(f"[DEBUG] ВНИМАНИЕ: Удален большой блок текста!")
        print(f"[DEBUG] Исходный текст начало: {content[:200]}...")
        print(f"[DEBUG] Обработанный текст: {cleaned[:200]}...")
    
    return cleaned

def chat_with_ai(prompt, system_prompt="", conversation_history=[]):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # Оптимизируем контекст
    context_manager = ContextManager()
    optimized_history = context_manager.optimize_context(conversation_history)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.extend(optimized_history)
    messages.append({"role": "user", "content": prompt})

    data = {
        "model": MODEL,
        "messages": messages,
        "stream": False
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 402:
            return "💳 **Ошибка оплаты**: На вашем аккаунте OpenRouter закончились средства или достигнут лимит. Пожалуйста, пополните баланс на https://openrouter.ai/"
        elif response.status_code == 401:
            return "🔑 **Ошибка авторизации**: Проверьте правильность API ключа OpenRouter"
        elif response.status_code != 200:
            return f"⚠️ **Ошибка API**: {response.status_code} - {response.text}"

        result = response.json()
        content = result["choices"][0]["message"]["content"]
        processed_content = process_content(content)
        
        # Логирование для отладки
        print(f"[DEBUG] Исходное сообщение длина: {len(content)}")
        print(f"[DEBUG] Обработанное сообщение длина: {len(processed_content)}")
        
        # Если после обработки контент стал пустым или очень коротким, возвращаем исходный
        if not processed_content or len(processed_content) < 10:
            print(f"[DEBUG] ВНИМАНИЕ: Обработанный контент слишком короткий, возвращаем исходный")
            return content
        
        return processed_content
    
    except Exception as e:
        return f"Ошибка: {str(e)}"

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

@app.route('/start_game', methods=['POST'])
@login_required
def start_game():
    if not API_KEY:
        return jsonify({"error": "API ключ не найден. Добавьте OPENROUTER_API_KEY в переменные окружения."})
    
    rules = load_gm_rules()
    system_prompt = create_gm_system_prompt(rules)
    
    session['conversation_history'] = []
    session['system_prompt'] = system_prompt
    session['character'] = None
    session['character_creation_mode'] = False
    
    # Проверяем, есть ли персонаж
    if 'character' not in session or not session['character']:
        response = "🎭 **Добро пожаловать в игру!**\n\nПрежде чем начать, мне нужно знать вашего персонажа. У вас есть три варианта:\n\n1. **Загрузить готового персонажа** - выберите из списка созданных персонажей\n2. **Создать нового персонажа** - напишите 'создать персонажа' и я помогу вам создать уникального героя\n3. **Загрузить файл персонажа** - используйте кнопку загрузки файла\n\nЧто выберете?"
    else:
        character_info = session['character']
        response = chat_with_ai(f"Начни игру для персонажа: {character_info}", system_prompt, [])
    
    if response and response.strip():
        session['conversation_history'] = [
            {"role": "user", "content": "Начни игру"},
            {"role": "assistant", "content": response}
        ]
    
    return jsonify({"response": response})

@app.route('/send_message', methods=['POST'])
@login_required
def send_message():
    data = request.get_json()
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({"error": "Пустое сообщение"})
    
    # Проверяем, находимся ли в режиме создания персонажа
    if session.get('character_creation_mode'):
        return create_character_continue(user_message)
    
    # Проверяем, запрашивает ли игрок создание персонажа
    if 'создать персонажа' in user_message.lower() or 'создание персонажа' in user_message.lower():
        return create_character_start()
    
    # Проверяем, есть ли персонаж
    if not session.get('character'):
        return jsonify({
            "response": "⚠️ Сначала нужно создать или загрузить персонажа! Напишите 'создать персонажа' или выберите персонажа из списка."
        })
    
    conversation_history = session.get('conversation_history', [])
    system_prompt = session.get('system_prompt', '')
    
    # Добавляем информацию о персонаже в контекст
    character_info = session.get('character')
    enhanced_prompt = f"{user_message}\n\n[ПЕРСОНАЖ ИГРОКА: {character_info}]"
    
    response = chat_with_ai(enhanced_prompt, system_prompt, conversation_history)
    
    if response and response.strip():
        conversation_history.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response}
        ])
        session['conversation_history'] = conversation_history
    
    return jsonify({"response": response})

def create_character_start():
    """Начинает процесс создания персонажа"""
    session['character_creation_history'] = []
    session['character_creation_mode'] = True
    
    response = """🎭 **СОЗДАНИЕ ПЕРСОНАЖА**

Отлично! Давайте создадим вашего персонажа. Я задам вам несколько вопросов, чтобы лучше понять, кого вы хотите играть.

**Первый вопрос:** Как зовут вашего персонажа и в каком мире или сеттинге вы хотели бы играть? (фэнтези, современность, киберпанк, космос и т.д.)"""
    
    return jsonify({
        "response": response,
        "character_creation": True
    })

def create_character_continue(user_input):
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
            session['character'] = character_description
            session['character_creation_mode'] = False
            session.pop('character_creation_history', None)
            
            # Сохраняем персонажа
            save_character_to_file(character_description)
            
            return jsonify({
                "response": response,
                "character_created": True,
                "character": character_description
            })
    
    # Продолжаем процесс создания
    creation_history.extend([
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response}
    ])
    session['character_creation_history'] = creation_history
    
    return jsonify({"response": response, "character_created": False})

def save_character_to_file(character_description):
    """Сохраняет персонажа в файл"""
    try:
        # Извлекаем имя персонажа
        lines = character_description.split('\n')
        character_name = "Безымянный"
        for line in lines:
            if line.startswith('Имя:'):
                character_name = line.replace('Имя:', '').strip()
                break
        
        user_folder = get_user_folder(session['username'], session['user_id'])
        characters_folder = os.path.join(user_folder, "characters")
        
        character_data = {
            "name": character_name,
            "description": character_description,
            "created_at": datetime.now().isoformat()
        }
        
        # Создаем безопасное имя файла
        safe_name = "".join(c for c in character_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(os.path.join(characters_folder, filename), 'w', encoding='utf-8') as f:
            json.dump(character_data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Ошибка сохранения персонажа: {e}")

@app.route('/load_character', methods=['POST'])
@login_required
def load_character():
    """Загружает персонажа из сохраненных"""
    data = request.get_json()
    filename = data.get('filename')
    
    if not filename:
        return jsonify({"error": "Не указано имя файла"})
    
    user_folder = get_user_folder(session['username'], session['user_id'])
    filepath = os.path.join(user_folder, "characters", f"{filename}.json")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            character_data = json.load(f)
        
        session['character'] = character_data['description']
        return jsonify({
            "success": True,
            "character": character_data['description'],
            "message": f"Персонаж '{character_data['name']}' загружен"
        })
        
    except FileNotFoundError:
        return jsonify({"error": "Файл персонажа не найден"})
    except Exception as e:
        return jsonify({"error": f"Ошибка загрузки персонажа: {str(e)}"})

@app.route('/save_game', methods=['POST'])
@login_required
def save_game():
    """Сохраняет текущую игру"""
    data = request.get_json()
    save_name = data.get('save_name', f"save_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    conversation_history = session.get('conversation_history', [])
    character = session.get('character')
    
    # Извлекаем имя персонажа
    character_name = "Неизвестный персонаж"
    if character:
        lines = character.split('\n')
        for line in lines:
            if line.startswith('Имя:'):
                character_name = line.replace('Имя:', '').strip()
                break
    
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "conversation_history": conversation_history,
        "character": character,
        "character_name": character_name,
        "save_name": save_name
    }
    
    user_folder = get_user_folder(session['username'], session['user_id'])
    save_path = os.path.join(user_folder, "saves", f"{save_name}.json")
    
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
            "history": save_data.get('conversation_history', [])
        })
    
    except FileNotFoundError:
        return jsonify({"error": "Файл сохранения не найден"})
    except Exception as e:
        return jsonify({"error": f"Ошибка загрузки: {str(e)}"})

@app.route('/upload_character', methods=['POST'])
@login_required
def upload_character():
    """Загружает файл персонажа"""
    if 'character_file' not in request.files:
        return jsonify({"error": "Файл не выбран"})
    
    file = request.files['character_file']
    if file.filename == '':
        return jsonify({"error": "Файл не выбран"})
    
    try:
        # Читаем содержимое файла
        content = file.read().decode('utf-8')
        
        # Пытаемся распарсить как JSON
        try:
            character_data = json.loads(content)
            # Создаем читаемое описание персонажа
            character_description = format_character_description(character_data)
        except json.JSONDecodeError:
            # Если не JSON, используем как текст
            character_description = content
        
        session['character'] = character_description
        
        # Сохраняем загруженного персонажа
        save_character_to_file(character_description)
        
        return jsonify({"success": True, "character": character_description, "message": "Персонаж загружен успешно"})
    
    except Exception as e:
        return jsonify({"error": f"Ошибка при загрузке файла: {str(e)}"})

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
