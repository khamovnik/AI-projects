import os
import PyPDF2
import telebot
import pandas as pd
import pytesseract
from PIL import Image
from telebot import types
from deep_translator import GoogleTranslator

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

def main():
    token1 = '5663307481:AAGjpvtziVPtz8KQ5WxdjbAPqjDLiQOQMOM'     
    bot = telebot.TeleBot(token1)
    user_states = {}
    user_file_format = {}

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
        show_file_format_menu(call.message.chat.id, call.data)

    def show_file_format_menu(chat_id, user_language):
        keyboard = types.InlineKeyboardMarkup()
        button_csv = types.InlineKeyboardButton(text="CSV", callback_data="csv")
        button_excel = types.InlineKeyboardButton(text="Excel", callback_data="excel")
        keyboard.add(button_csv, button_excel)
        bot.send_message(chat_id, translate_message("Выберите формат для конвертации файла", user_language), reply_markup=keyboard)

    @bot.callback_query_handler(func=lambda call: call.data in ["csv", "excel"])
    def file_format_selected(call: types.CallbackQuery):
        user_file_format[call.message.chat.id] = call.data
        show_request_file_message(call.message.chat.id)

    def show_request_file_message(chat_id):
        user_language = user_states.get(chat_id)
        bot.send_message(chat_id, translate_message("Пожалуйста, отправьте файл для обработки", user_language))

    @bot.message_handler(content_types=['document'])
    def handle_document(message):
        user_language = user_states.get(message.chat.id)
        file_format = user_file_format.get(message.chat.id, "csv")
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
            extracted_text = run_program_if_valid_file(temp_file_path, user_language)
            processed_text = ReplaceTermin(extracted_text, terminology_dict)
            translated_rows = translate_text(processed_text.split('\n'), user_language)

            if file_format == "csv":
                csv_file_path = f"{message.document.file_id}.csv"
                CreateCsv(csv_file_path, translated_rows)
                with open(csv_file_path, 'rb') as file:
                    bot.send_document(message.chat.id, file)
                os.remove(csv_file_path)
            else:  # Excel
                excel_file_path = f"{message.document.file_id}.xlsx"
                CreateExcel(excel_file_path, translated_rows)
                with open(excel_file_path, 'rb') as file:
                    bot.send_document(message.chat.id, file)
                os.remove(excel_file_path)

        except ValueError as e:
            bot.reply_to(message, str(e))
        finally:
            os.remove(temp_file_path)

    def translate_message(message, language_key):
        lang_map = {
            "button1": "russian",
            "button2": "english",
            "button3": "german",
            "button4": "hebrew"
        }
        target_language = lang_map.get(language_key, "ru")  # по умолчанию русский
        try:
            translated_message = GoogleTranslator(source='auto', target=target_language).translate(message)
            return translated_message
        except Exception:
            return message  # если перевод не удался, возвращаем оригинал

    def translate_text(rows, language_key):
        return [translate_message(row, language_key) for row in rows]

    def CreateExcel(file_path, data):
        dataframe = pd.DataFrame(data)
        dataframe.to_excel(file_path, index=False)

    def CreateCsv(file_path, data):
        dataframe = pd.DataFrame(data)
        dataframe.to_csv(file_path, index=False)

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

    def run_program_if_valid_file(file_path, user_language):
        _, ext = os.path.splitext(file_path)
        if ext.lower() == '.pdf':
            extracted_text = ReadFile(file_path)
        elif ext.lower() in ['.jpg', '.jpeg', '.png']:
            extracted_text = ReadImage(file_path)
        else:
            raise ValueError(translate_message("Неподдерживаемый формат файла. Пожалуйста, используйте PDF или изображение (JPG, JPEG, PNG).", user_language))
        return extracted_text
        
    bot.polling()

if __name__ == '__main__':
    main()
