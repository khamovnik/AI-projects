from dict import terminology_dict
from functions import ReadFile, ReplaceTermin, CreateCsv
import telebot
import os


def main():
    token1 = '5663307481:AAGjpvtziVPtz8KQ5WxdjbAPqjDLiQOQMOM'
# создаем экземпляр бота с указанным токеном
    bot = telebot.TeleBot(token1)


# обработчик команды /start
    @bot.message_handler(commands=['start'])
    def handle_start(message):
        bot.reply_to(message, "Привет! Я бот для обработки PDF файлов")

# обработчик команды /process
    @bot.message_handler(commands=['process'])
    def handle_process(message):
    # отправляем сообщение с просьбой отправить PDF файл
        bot.reply_to(message, "Пожалуйста, отправьте PDF файл для обработки")

# обработчик приема PDF файла
    @bot.message_handler(content_types=['document'])
    def handle_document(message):
    # получаем информацию о файле
        file_info = bot.get_file(message.document.file_id)
        file_path = file_info.file_path

    # скачиваем файл
        downloaded_file = bot.download_file(file_path)

    # сохраняем файл с расширением .pdf
        pdf_file_path = f"{message.document.file_id}.pdf"
        with open(pdf_file_path, 'wb') as file:
            file.write(downloaded_file)

    # вызываем функции для обработки PDF файла
        pdf_text = ReadFile(pdf_file_path)
        processed_text = ReplaceTermin(pdf_text, terminology_dict)
        rows = processed_text.split('\n')
        csv_file_path = f"{message.document.file_id}.csv"
        CreateCsv(csv_file_path, rows)

    # отправляем обработанный CSV файл
        with open(csv_file_path, 'rb') as file:
            bot.send_document(message.chat.id, file)

    # удаляем временные файлы
        os.remove(pdf_file_path)
        os.remove(csv_file_path)

# запускаем бота
    bot.polling()
