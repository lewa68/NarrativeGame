import requests
import json
import os

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "deepseek/deepseek-r1"

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

Начни игру с предложения выбрать сеттинг."""
    
    return system_prompt

def process_content(content):
    return content.replace('<think>', '').replace('</think>', '')

def chat_stream(prompt, system_prompt="", conversation_history=[]):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Добавляем историю разговора
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": prompt})

    data = {
        "model": MODEL,
        "messages": messages,
        "stream": True
    }

    with requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data,
        stream=True
    ) as response:
        if response.status_code != 200:
            print("Ошибка API:", response.status_code)
            return ""

        full_response = []

        for chunk in response.iter_lines():
            if chunk:
                chunk_str = chunk.decode('utf-8').replace('data: ', '')
                try:
                    chunk_json = json.loads(chunk_str)
                    if "choices" in chunk_json:
                        content = chunk_json["choices"][0]["delta"].get("content", "")
                        if content:
                            cleaned = process_content(content)
                            print(cleaned, end='', flush=True)
                            full_response.append(cleaned)
                except:
                    pass

        print()  # Перенос строки после завершения потока
        return ''.join(full_response)
def main():
    if not API_KEY:
        print("Ошибка: API ключ не найден. Добавьте OPENROUTER_API_KEY в Secrets.")
        return
        
    # Загружаем правила ГМ
    rules = load_gm_rules()
    system_prompt = create_gm_system_prompt(rules)
    
    print("=== НАРРАТИВНАЯ РОЛЕВАЯ ИГРА ===")
    print("ГМ: DeepSeek-R1 (специализированная версия для RPG)")
    print("Для выхода введите 'exit'")
    print("Используйте тег 'ГМ:' если хотите получить варианты действий\n")
    
    conversation_history = []
    
    # Первое сообщение от ГМ
    print("ГМ:", end=' ', flush=True)
    first_response = chat_stream("Начни игру", system_prompt, conversation_history)
    if first_response:
        conversation_history.extend([
            {"role": "user", "content": "Начни игру"},
            {"role": "assistant", "content": first_response}
        ])

    while True:
        user_input = input("\nВы: ")

        if user_input.lower() == 'exit':
            print("Завершение работы...")
            break

        print("ГМ:", end=' ', flush=True)
        response = chat_stream(user_input, system_prompt, conversation_history)
        
        if response:
            conversation_history.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response}
            ])
            
            # Ограничиваем историю последними 20 сообщениями для экономии токенов
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]

if __name__ == "__main__":
    main()