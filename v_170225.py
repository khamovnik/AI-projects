# БЕЗ ЯЗЫКОВ, КОСТЫЛЬ С МЕНЮ. НО РАБОТАЮТ ДВА АЛГОРИТМА

import os
import re
import telebot
import numpy as np
from telebot import types
from deep_translator import GoogleTranslator
import pytesseract
import cv2
import joblib
import PyPDF2
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.tree import DecisionTreeClassifier

# === Настройки ML ===
MODEL_PATH = "ml_model.pkl"

def train_basic_model():
    X = [[20], [50], [23], [40], [3], [200], [4], [10], [87], [150], [200], [500], [60], [100], [1], [200]]
    y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    model = DecisionTreeClassifier()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

def load_model():
    if not os.path.exists(MODEL_PATH):
        train_basic_model()
    return joblib.load(MODEL_PATH)

# === Функции обработки Коагулограммы ===
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text_koagulogram(image_path):
    processed_img = preprocess_image(image_path)
    return pytesseract.image_to_string(processed_img, lang='rus')

def parse_results_koagulogram(text):
    text = re.sub(r'\b(изм|т)\b', '', text)
    pattern = r"([А-Яа-я]+[\sА-Яа-я]*)\s+(\d+[.,]?\d*)\s+(\d+[.,]?\d*-\d+[.,]?\d*)"
    matches = re.findall(pattern, text)
    results = [(match[0].strip(), float(match[1].replace(',', '.')),
                list(map(lambda x: float(x.replace(',', '.')), match[2].split('-')))) for match in matches]
    return results

def classify_results_koagulogram(results):
    data, labels = [], []
    for _, value, ref_range in results:
        data.append([value])
        labels.append(1 if ref_range[0] <= value <= ref_range[1] else 0)

    clf = DecisionTreeClassifier()
    clf.fit(data, labels)

    classifications = []
    for test_name, value, ref_range in results:
        prediction = clf.predict([[value]])[0]
        status = "В пределах нормы" if prediction == 1 else "Отклонение"
        classifications.append(f"{test_name}: {status} (Значение: {value}, Референсные: {ref_range[0]}-{ref_range[1]})")
    return classifications

# === Функции обработки Биохимии ===
def preprocess_image_bio(image):
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    return image

def extract_text_bio(file_path):
    _, ext = os.path.splitext(file_path)
    if ext.lower() == '.pdf':
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return "".join(page.extract_text() or "" for page in pdf_reader.pages)
    elif ext.lower() in ['.jpg', '.jpeg', '.png']:
        image = Image.open(file_path)
        image = preprocess_image_bio(image)
        return pytesseract.image_to_string(image, lang='rus')
    else:
        raise ValueError("Неподдерживаемый формат файла. Используйте PDF или изображение.")

def parse_results_bio(text):
    results = []
    for line in text.splitlines():
        match = re.match(r"(.+?)\s+\[?([\d\.\*]+)\s*\|?\]?\s+([а-яА-Яa-zA-Z/]+)\s+([\d\.\-\>\<\s]+)", line)
        if match:
            results.append((match.group(1).strip(), match.group(2).strip().replace("*", ""),
                            match.group(3).strip(), match.group(4).strip()))
    return results

def analyze_results_bio(parsed_results, model):
    interpretations = []
    for test_name, value, unit, ref_range in parsed_results:
        try:
            if '<' in ref_range:
                ref_low, ref_high = 0.0, float(re.findall(r"[\d\.]+", ref_range)[0])
            elif '>' in ref_range:
                ref_low, ref_high = float(re.findall(r"[\d\.]+", ref_range)[0]), float('inf')
            else:
                ref_values = re.findall(r"[\d\.]+", ref_range)
                ref_low, ref_high = float(ref_values[0]), float(ref_values[1])

            value = float(value)
            status = "Отклонение" if value < ref_low or value > ref_high else "В пределах нормы"
            interpretations.append(f"{test_name}: {status} (Значение: {value} {unit}, Референсные: {ref_low}-{ref_high})")
        except:
            interpretations.append(f"{test_name}: Ошибка анализа. (Значение: {value} {unit})")
    return interpretations

# === Телеграм-бот ===
bot = telebot.TeleBot('5663307481:AAGjpvtziVPtz8KQ5WxdjbAPqjDLiQOQMOM')
user_states = {}

def translate_message(message, language_key):
    lang_map = {"button1": "russian", "button2": "english", "button3": "german", "button4": "hebrew"}
    return GoogleTranslator(source='auto', target=lang_map.get(language_key, "russian")).translate(message)

@bot.message_handler(commands=['start'])
def start_command(message):
    keyboard = types.InlineKeyboardMarkup()
    buttons = [types.InlineKeyboardButton(text="Coagulogram", callback_data="coagulogram"),
               types.InlineKeyboardButton(text="BBT", callback_data="biochemistry")]
    keyboard.add(*buttons)
    bot.send_message(message.chat.id, "Choose a type of analysis:", reply_markup=keyboard)

@bot.callback_query_handler(func=lambda call: call.data in ["coagulogram", "biochemistry"])
def analysis_selected(call):
    user_states[call.message.chat.id] = call.data
    bot.send_message(call.message.chat.id, "Пожалуйста, отправьте файл для анализа.")

@bot.message_handler(content_types=['document'])
def handle_file(message):
    analysis_type = user_states.get(message.chat.id, "biochemistry")
    model = load_model() if analysis_type == "biochemistry" else None
    temp_file_path = None
    try:
        file_info = bot.get_file(message.document.file_id)
        file_content = bot.download_file(file_info.file_path)
        _, ext = os.path.splitext(file_info.file_path)

        temp_file_path = f"{message.document.file_id}{ext}"
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(file_content)

        if analysis_type == "coagulogram":
            extracted_text = extract_text_koagulogram(temp_file_path)
            parsed_results = parse_results_koagulogram(extracted_text)
            interpretations = classify_results_koagulogram(parsed_results)
        else:
            extracted_text = extract_text_bio(temp_file_path)
            parsed_results = parse_results_bio(extracted_text)
            interpretations = analyze_results_bio(parsed_results, model)

        bot.send_message(message.chat.id, f"Результаты анализа:\n" + "\n".join(interpretations))

    except Exception as e:
        bot.send_message(message.chat.id, f"Ошибка обработки файла: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

bot.polling()
