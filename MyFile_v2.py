import os
import PyPDF2
import telebot
import pandas as pd
import pytesseract
from PIL import Image
from telebot import types
from googletrans import Translator
from deep_translator import GoogleTranslator
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Словарь терминов
terminology_dict = {
    "АсАТ": "аспартатаминотрансфераза",
    "АлАТ": "аланинаминотрансфераза",
    "ГГТ": "гамма-глутамилтрансфераза",
    "АЛП": "щелочная фосфатаза",
    "ЛДГ": "лактатдегидрогеназа",
    "КПК": "креатинфосфокиназа",
    "БСК": "общий белок крови",
    "Альбумин": "альбумин",
    "Глобулины": "глобулины",
    "ЩФ": "щелочная фосфатаза",
    "СОЭ": "скорость оседания эритроцитов",
    "ХСЛ": "холестерол",
    "ЛПНП": "липопротеины низкой плотности",
    "ЛПВП": "липопротеины высокой плотности",
    "ТГ": "триглицериды",
    "ГАД": "глюкоза",
    "Креатинин": "креатинин",
    "Урея": "урея",
    "Натрий": "Na+",
    "Калий": "K+",
    "Хлор": "Cl-",
    "Кальций": "Ca2+",
    "Фосфор": "P"
}

# Глобальные переменные для модели и токенизатора
model = None
tokenizer = None

def create_model(input_length):
    model = keras.Sequential([
        layers.Embedding(input_dim=1000, output_dim=64, input_length=input_length),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')  # 3 класса для рекомендаций
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model():
    # Пример данных (в реальности, данные должны быть загружены из файлов)
    data = {
        'text': [
            "Blood test result: Normal",
            "Blood test result: High cholesterol",
            "Blood test result: Low hemoglobin"
        ],
        'recommendation': [
            "No action needed.",
            "Consider dietary changes and exercise.",
            "Consult a doctor."
        ]
    }
    df = pd.DataFrame(data)

    # Кодирование текстов
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df['text'])
    sequences = tokenizer.texts_to_sequences(df['text'])
    padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences)

    labels = np.array([0, 1, 2])  # Пример меток для рекомендаций

    model = create_model(padded_sequences.shape[1])
    model.fit(padded_sequences, labels, epochs=10)

    return model, tokenizer

def predict_recommendation(text, user_language):
    sequences = tokenizer.texts_to_sequences([text])
    padded = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=tokenizer.num_words)
    prediction = model.predict(padded)
    recommendation_index = np.argmax(prediction)
    
    recommendations_1 = [
        "Действий не нужно.", 
        "Подумайте об изменении диеты и физических упражнениях.",
        "Обратитесь к врачу."
     ]
    recommendations_2 = [
        "No action needed.",
        "Consider dietary changes and exercise.",
        "Consult a doctor."
     ]
    recommendations_3 = [
        "Keine Aktion erforderlich.",
        "Erwägen Sie Ernährungsumstellungen und Bewegung.",
        "Konsultieren Sie einen Arzt."
     ]
    recommendations_4 = [
        "אין צורך בפעולה.",
        "שקול שינויים תזונתיים ופעילות גופנית.",
        "תתייעצי עם רופא."
     ]

    if user_language == "button1":
       return recommendations_1[recommendation_index]

    if user_language == "button2":
       return recommendations_2[recommendation_index]

    if user_language == "button3":
       return recommendations_3[recommendation_index]
    
    if user_language == "button4":
       return recommendations_4[recommendation_index]

def main():
    global model, tokenizer
    token1 = '5663307481:AAGjpvtziVPtz8KQ5WxdjbAPqjDLiQOQMOM'  # Ваш токен
    bot = telebot.TeleBot(token1)
    user_states = {}
    translator = Translator()

    # Обучаем модель
    model, tokenizer = train_model()

    @bot.message_handler(commands=['start'])
    def handle_start(message):
        show_language_menu(message.chat.id)

    def show_language_menu(chat_id):
        keyboard = types.InlineKeyboardMarkup()
        button1 = types.InlineKeyboardButton(text="Русский", callback_data="button1")
        button2 = types.InlineKeyboardButton(text="English", callback_data="button2")
        button3 = types.InlineKeyboardButton(text="Deutsch", callback_data="button3")
        button4 = types.InlineKeyboardButton(text="עברית", callback_data="button4")
        keyboard.add(button1, button2, button3, button4)
        bot.send_message(chat_id, "Choose your language", reply_markup=keyboard)

    @bot.callback_query_handler(func=lambda call: call.data in ["button1", "button2", "button3", "button4"])
    def language_selected(call: types.CallbackQuery):
        user_states[call.message.chat.id] = call.data
        show_request_file_message(call.message.chat.id)

    def show_request_file_message(chat_id):
        user_language = user_states.get(chat_id)
        if user_language == "button1":
            bot.send_message(chat_id, "Пожалуйста, отправьте файл для обработки")
        elif user_language == "button2":
            bot.send_message(chat_id, "Please, send the file for processing")
        elif user_language == "button3":
            bot.send_message(chat_id, "Bitte, senden Sie die Datei zur Verarbeitung")
        elif user_language == "button4":
            bot.send_message(chat_id, "נא לשלוח את הקובץ לעיבוד")

    @bot.message_handler(content_types=['document'])
    def handle_document(message):
        user_language = user_states.get(message.chat.id)
        if not user_language:
            show_language_menu(message.chat.id)
            return

        file_info = bot.get_file(message.document.file_id)
        file_path = file_info.file_path
        downloaded_file = bot.download_file(file_path)
        file_extension = os.path.splitext(message.document.file_name)[1].lower()

        temp_file_path = f"{message.document.file_id}{file_extension}"
        with open(temp_file_path, 'wb') as file:
            file.write(downloaded_file)
        try:
            extracted_text = run_program_if_valid_file(temp_file_path)
            processed_text = ReplaceTermin(extracted_text, terminology_dict)
            translated_rows = translate_text(processed_text.split('\n'), user_language)

            # Получение рекомендации
            recommendation = predict_recommendation(processed_text, user_language)
            bot.send_message(message.chat.id, recommendation)

            csv_file_path = f"{message.document.file_id}.csv"
            CreateCsv(csv_file_path, translated_rows)
            with open(csv_file_path, 'rb') as file:
                bot.send_document(message.chat.id, file)

            os.remove(csv_file_path)
        except ValueError as e:
            bot.reply_to(message, str(e))
        finally:
            os.remove(temp_file_path)

    def translate_text(rows, language_key):
        lang_map = {
            "button1": "russian",
            "button2": "english",
            "button3": "german",
            "button4": "hebrew"
        }
        target_language = lang_map.get(language_key)
        if language_key == "button1":
            return rows
        translated_rows = [GoogleTranslator(source='auto', target=target_language).translate(row) for row in rows]
        return translated_rows

    bot.polling()

def ReadImage(file_path):
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image, lang='rus')
    return text

def ReadFile(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def CreateCsv(file_path, data):
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(file_path, index=False)

def ReplaceTermin(text, terminology_dict):
    lines = text.splitlines()
    result = []
    for line in lines:
        for key, value in terminology_dict.items():
            if key in line:
                line = line.replace(key, f"{value} ({get_value_from_line(line, key)})")
        result.append(line)
    return "\n".join(result)

def get_value_from_line(line, key):
    parts = line.split()
    for i, part in enumerate(parts):
        if part == key and i + 1 < len(parts):
            return parts[i + 1]
    return None

def run_program_if_valid_file(file_path):
    _, ext = os.path.splitext(file_path)
    if ext.lower() == '.pdf':
        extracted_text = ReadFile(file_path)
    elif ext.lower() in ['.jpg', '.jpeg', '.png']:
        extracted_text = ReadImage(file_path)
    else:
        raise ValueError("Неподдерживаемый формат файла. Пожалуйста, используйте PDF или изображение (JPG, JPEG, PNG).")

    return extracted_text

if __name__ == '__main__':
    main()
