import pip
pip.main(['install', 'flask'])
import requests
import json
import os
from flask import Flask, render_template, request, jsonify, session
import secrets
from datetime import datetime

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "deepseek/deepseek-r1"

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

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
        
        if response.status_code != 200:
            return f"Ошибка API: {response.status_code}"

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
    return render_template('index.html')

@app.route('/start_game', methods=['POST'])
def start_game():
    if not API_KEY:
        return jsonify({"error": "API ключ не найден. Добавьте OPENROUTER_API_KEY в Secrets."})
    
    rules = load_gm_rules()
    system_prompt = create_gm_system_prompt(rules)
    
    session['conversation_history'] = []
    session['system_prompt'] = system_prompt
    session['character'] = None
    
    # Проверяем, есть ли персонаж
    if 'character' not in session or not session['character']:
        response = "🎭 **Добро пожаловать в игру!**\n\nПрежде чем начать, мне нужно знать вашего персонажа. У вас есть два варианта:\n\n1. **Загрузить готового персонажа** - используйте кнопку загрузки файла выше\n2. **Создать нового персонажа** - напишите 'создать персонажа' и я помогу вам создать уникального героя\n\nЧто выберете?"
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
def send_message():
    data = request.get_json()
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({"error": "Пустое сообщение"})
    
    # Проверяем, запрашивает ли игрок создание персонажа
    if 'создать персонажа' in user_message.lower() or 'создание персонажа' in user_message.lower():
        return create_character_start()
    
    # Проверяем, есть ли персонаж
    if not session.get('character'):
        return jsonify({
            "response": "⚠️ Сначала нужно создать или загрузить персонажа! Напишите 'создать персонажа' или используйте кнопку загрузки файла."
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
    
    response = """🎭 **СОЗДАНИЕ ПЕРСОНАЖА**

Отлично! Давайте создадим вашего персонажа. Я задам вам несколько вопросов, чтобы лучше понять, кого вы хотите играть.

**Первый вопрос:** Как зовут вашего персонажа и в каком мире или сеттинге вы хотели бы играть? (фэнтези, современность, киберпанк, космос и т.д.)"""
    
    return jsonify({
        "response": response,
        "character_creation": True
    })

@app.route('/save_game', methods=['POST'])
def save_game():
    """Сохраняет текущую игру"""
    conversation_history = session.get('conversation_history', [])
    character = session.get('character')
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "conversation_history": conversation_history,
        "character": character
    }
    
    with open("game_save.json", "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    return jsonify({"message": "Игра сохранена"})

@app.route('/load_game', methods=['POST'])
def load_game():
    """Загружает сохраненную игру"""
    try:
        with open("game_save.json", "r", encoding="utf-8") as f:
            save_data = json.load(f)
        
        session['conversation_history'] = save_data.get('conversation_history', [])
        session['character'] = save_data.get('character', None)
        return jsonify({"message": "Игра загружена", "timestamp": save_data.get('timestamp')})
    
    except FileNotFoundError:
        return jsonify({"error": "Файл сохранения не найден"})

@app.route('/upload_character', methods=['POST'])
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
        return jsonify({"message": "Персонаж загружен успешно", "character": character_description})
    
    except Exception as e:
        return jsonify({"error": f"Ошибка при загрузке файла: {str(e)}"})

@app.route('/create_character', methods=['POST'])
def create_character():
    """Создает персонажа через взаимодействие с ГМ"""
    data = request.get_json()
    user_input = data.get('input', '')
    
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

Когда персонаж будет готов, заверши описанием в формате:
=== ПЕРСОНАЖ СОЗДАН ===
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
            session.pop('character_creation_history', None)  # Очищаем историю создания
            
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

# Консольная версия (для обратной совместимости)
def console_main():
    if not API_KEY:
        print("❌ Ошибка: API ключ не найден. Добавьте OPENROUTER_API_KEY в Secrets.")
        return
        
    rules = load_gm_rules()
    system_prompt = create_gm_system_prompt(rules)
    
    print("╔" + "="*78 + "╗")
    print("║" + " "*25 + "НАРРАТИВНАЯ РОЛЕВАЯ ИГРА" + " "*25 + "║")
    print("║" + " "*78 + "║")
    print("║  🤖 ГМ: DeepSeek-R1 (специализированная версия для RPG)" + " "*14 + "║")
    print("║  📝 Для выхода введите 'exit'" + " "*41 + "║")
    print("║  🌐 Для веб-версии запустите: python main.py web" + " "*23 + "║")
    print("╚" + "="*78 + "╝")
    
    conversation_history = []
    context_manager = ContextManager()
    
    # Первое сообщение от ГМ
    print("\n🎲 ГЕЙМ МАСТЕР:")
    print("-" * 40)
    first_response = chat_with_ai("Начни игру", system_prompt, conversation_history)
    print(first_response)
    
    if first_response:
        conversation_history.extend([
            {"role": "user", "content": "Начни игру"},
            {"role": "assistant", "content": first_response}
        ])

    while True:
        print("\n🎮 ВАШ ХОД:")
        print("-" * 20)
        user_input = input(">>> ")

        if user_input.lower() == 'exit':
            print("\n🚪 Завершение работы...")
            print("Спасибо за игру! 🎲")
            break

        print("\n" + "="*80 + "\n")
        print("🎲 ГЕЙМ МАСТЕР:")
        print("-" * 40)
        
        # Оптимизируем контекст перед отправкой
        conversation_history = context_manager.optimize_context(conversation_history)
        
        response = chat_with_ai(user_input, system_prompt, conversation_history)
        print(response)
        
        if response:
            conversation_history.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response}
            ])

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'web':
        print("🌐 Запуск веб-сервера на http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        console_main()
