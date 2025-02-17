# РАБОТАЕТ 1 АЛГОРИТМ + ЯЗЫКИ + ДИАГРАММА

import os
import re
import telebot
import numpy as np
from datetime import datetime
from telebot import types
from deep_translator import GoogleTranslator
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import PyPDF2
from sklearn.tree import DecisionTreeClassifier
import joblib
import matplotlib.pyplot as plt

# === Настройки ML ===
MODEL_PATH = "ml_model.pkl"

# Функция обучения базовой модели
def train_basic_model():
    # Пример данных: тесты, значения и классы интерпретации (в пределах нормы = 0, отклонение = 1)
    X = [
        [20], [50],  # АлАТ
        [23], [40],  # АсАТ
        [3], [200],  # Билирубин общий
        [4], [10],  # Глюкоза
        [87], [150],  # Креатинин
        [200], [500],  # Мочевая кислота
        [60], [100],  # Общий белок
        [1], [200],  # Холестерин
    ]
    y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    model = DecisionTreeClassifier()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

# Загрузка модели
def load_model():
    if not os.path.exists(MODEL_PATH):
        train_basic_model()
    return joblib.load(MODEL_PATH)

# Предобработка данных для модели
def preprocess_data(data):
    try:
        numeric_data = [[float(value)] for value in data]
        return numeric_data
    except ValueError:
        return []

# Интерпретация результатов на основе модели
def interpret_results(data, model):
    preprocessed_data = preprocess_data(data)
    if not preprocessed_data:
        return ["Ошибка: данные для анализа не обнаружены."]

    # Преобразование в двумерный массив
    preprocessed_data = np.array(preprocessed_data).reshape(-1, 1)
    predictions = model.predict(preprocessed_data)
    return ["Отклонение" if pred else "В пределах нормы" for pred in predictions]

# Анализ текста и извлечение данных
def parse_analysis_text(text):
    results = []
    for line in text.splitlines():
        if not line.strip():
            continue

        # Адаптированное регулярное выражение для обработки лишних символов
        match = re.match(r"(.+?)\s+\[?([\d\.\*]+)\s*\|?\]?\s+([а-яА-Яa-zA-Z/]+)\s+([\d\.\-\>\<\s]+)", line)
        if match:
            test_name = match.group(1).strip()  # Название теста
            value = match.group(2).strip().replace("*", "")  # Значение (без звёздочек)
            unit = match.group(3).strip()  # Единицы измерения
            ref_range = match.group(4).strip()  # Референсные значения

            results.append((test_name, value, unit, ref_range))
        else:
            print(f"Строка не соответствует формату: {line}")
    return results

# Функция анализа данных
def analyze_results(parsed_results, model):
    interpretations = []
    for test_name, value, unit, ref_range in parsed_results:
        try:
            # Обработка референсных значений
            if '<' in ref_range:  # Верхняя граница, например: <41
                ref_low = 0.0
                ref_high = float(re.findall(r"[\d\.]+", ref_range)[0])
            elif '>' in ref_range:  # Нижняя граница, например: >64
                ref_low = float(re.findall(r"[\d\.]+", ref_range)[0])
                ref_high = float('inf')
            else:  # Полный диапазон, например: 4.1 - 6.0
                ref_values = re.findall(r"[\d\.]+", ref_range)
                ref_low = float(ref_values[0])
                ref_high = float(ref_values[1])

            value = float(value)

            # Проверяем отклонения
            if value < ref_low or value > ref_high:
                status = "Отклонение"
            else:
                status = "В пределах нормы"

            interpretations.append(f"{test_name}: {status} (Значение: {value} {unit}, Референсные: {ref_low}-{ref_high})")
        except (ValueError, IndexError):
            interpretations.append(f"{test_name}: Ошибка: данные для анализа не обнаружены. (Значение: {value} {unit})")
    return interpretations

# Построение графиков
def plot_results(parsed_results):
    test_names = []
    values = []
    ref_lows = []
    ref_highs = []

    for test_name, value, unit, ref_range in parsed_results:
        try:
            if '<' in ref_range:
                ref_low = 0.0
                ref_high = float(re.findall(r"[\d\.]+", ref_range)[0])
            elif '>' in ref_range:
                ref_low = float(re.findall(r"[\d\.]+", ref_range)[0])
                ref_high = float('inf')
            else:
                ref_values = re.findall(r"[\d\.]+", ref_range)
                ref_low = float(ref_values[0])
                ref_high = float(ref_values[1])

            test_names.append(test_name)
            values.append(float(value))
            ref_lows.append(ref_low)
            ref_highs.append(ref_high)
        except (ValueError, IndexError):
            continue

    plt.figure(figsize=(10, 6))
    plt.bar(test_names, values, color='blue', label='Значения')
    plt.bar(test_names, ref_highs, color='red', alpha=0.2, label='Референсные значения')
    plt.xticks(rotation=90)
    plt.ylabel('Значения')
    plt.title('Результаты анализов')
    plt.legend()
    plt.tight_layout()
    plt.savefig('analysis_results.png')
    plt.close()

# === Работа с файлами ===
def preprocess_image(image):
    # Увеличение контрастности и резкости
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)
    # Применение фильтра для уменьшения шума
    image = image.filter(ImageFilter.MedianFilter(size=3))
    return image

def extract_text_from_file(file_path):
    _, ext = os.path.splitext(file_path)
    try:
        if ext.lower() == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        elif ext.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                image = Image.open(file_path)
                image = preprocess_image(image)
                text = pytesseract.image_to_string(image, lang='rus')
            except Exception as img_err:
                raise ValueError(f"Ошибка обработки изображения: {img_err}")
        else:
            raise ValueError("Неподдерживаемый формат файла. Пожалуйста, используйте PDF или изображение.")
    except Exception as e:
        raise ValueError(f"Ошибка обработки файла: {e}")
    return text

# === Телеграм-бот ===
def main():
    model = load_model()
    bot = telebot.TeleBot('5663307481:AAGjpvtziVPtz8KQ5WxdjbAPqjDLiQOQMOM')
    user_states = {}

    def translate_message(message, language_key):
        lang_map = {
            "button1": "russian",
            "button2": "english",
            "button3": "german",
            "button4": "hebrew"
        }
        target_language = lang_map.get(language_key, "russian")
        try:
            return GoogleTranslator(source='auto', target=target_language).translate(message)
        except Exception:
            return message

    @bot.message_handler(commands=['start'])
    def start_command(message):
        show_language_menu(message.chat.id)

    def show_language_menu(chat_id):
        keyboard = types.InlineKeyboardMarkup()
        buttons = [
            types.InlineKeyboardButton(text="Русский", callback_data="button1"),
            types.InlineKeyboardButton(text="English", callback_data="button2"),
            types.InlineKeyboardButton(text="Deutsch", callback_data="button3"),
            types.InlineKeyboardButton(text="עברית", callback_data="button4")
        ]
        keyboard.add(*buttons)
        bot.send_message(chat_id, "Choose your language", reply_markup=keyboard)

    @bot.callback_query_handler(func=lambda call: call.data in ["button1", "button2", "button3", "button4"])
    def language_selected(call: types.CallbackQuery):
        user_states[call.message.chat.id] = call.data
        bot.send_message(call.message.chat.id, translate_message("Пожалуйста, отправьте файл для анализа.", call.data))

    @bot.message_handler(content_types=['document'])
    def handle_file(message):
        user_language = user_states.get(message.chat.id, "button1")
        temp_file_path = None  # Инициализация переменной для временного файла
        try:
            # Загрузка файла
            file_info = bot.get_file(message.document.file_id)
            file_content = bot.download_file(file_info.file_path)
            _, ext = os.path.splitext(file_info.file_path)

            # Создание временного файла
            temp_file_path = f"{message.document.file_id}{ext}"
            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(file_content)

            # Извлечение текста
            extracted_text = extract_text_from_file(temp_file_path)
            parsed_results = parse_analysis_text(extracted_text)

            # Анализ результатов
            interpretations = analyze_results(parsed_results, model)
            translated_results = "\n".join(translate_message(line, user_language) for line in interpretations)
            bot.send_message(message.chat.id, f"Результаты анализа:\n{translated_results}")

            # Построение и отправка графика
            plot_results(parsed_results)
            with open('analysis_results.png', 'rb') as plot:
                bot.send_photo(message.chat.id, plot)

        except Exception as e:
            bot.send_message(message.chat.id, f"Ошибка обработки файла: {str(e)}")
        finally:
            # Удаление временного файла
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists('analysis_results.png'):
                os.remove('analysis_results.png')

    bot.polling()

if __name__ == "__main__":
    main()
