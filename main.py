from dict import terminology_dict
from functions import ReadFile, ReplaceTermin, CreateCsv

pdf_name = 'Semen.pdf'
csv_name = 'output_new.csv'
pdf_text = ReadFile(pdf_name)
processed_text = ReplaceTermin(pdf_text, terminology_dict)
rows = processed_text.split('\n')
CreateCsv(csv_name, rows)
print("Первый файл обработан!")

pdf_name_2 = 'Dnkom.pdf'
csv_name_2 = 'output_new2.csv'
pdf_text_2 = ReadFile(pdf_name_2)
processed_text_2 = ReplaceTermin(pdf_text_2, terminology_dict)
rows_2 = processed_text_2.split('\n')
CreateCsv(csv_name_2, rows_2)
print("Второй файл обработан!")

pdf_name_3 = 'Vet.pdf'
csv_name_3 = 'output_new3.csv'
pdf_text_3 = ReadFile(pdf_name_3)
processed_text_3 = ReplaceTermin(pdf_text_3, terminology_dict)
rows_3 = processed_text_3.split('\n')
CreateCsv(csv_name_3, rows_3)
print("Третий файл обработан!")

pdf_name_4 = 'Eml.pdf'
csv_name_4 = 'output_new4.csv'
pdf_text_4 = ReadFile(pdf_name_4)
processed_text_4 = ReplaceTermin(pdf_text_4, terminology_dict)
rows_4 = processed_text_4.split('\n')
CreateCsv(csv_name_4, rows_4)
print("Четвёртый файл обработан!")

pdf_name_5 = 'Kiselev_1.pdf'
csv_name_5 = 'output_new5.csv'
pdf_text_5 = ReadFile(pdf_name_5)
processed_text_5 = ReplaceTermin(pdf_text_5, terminology_dict)
rows_5 = processed_text_5.split('\n')
CreateCsv(csv_name_5, rows_5)
print("Пятый файл обработан!")

pdf_name_6 = 'Kiselev_2.pdf'
csv_name_6 = 'output_new6.csv'
pdf_text_6 = ReadFile(pdf_name_6)
processed_text_6 = ReplaceTermin(pdf_text_6, terminology_dict)
rows_6 = processed_text_6.split('\n')
CreateCsv(csv_name_6, rows_6)
print("Пятый файл обработан!")

pdf_name_7 = 'Kiselev_3.pdf'
csv_name_7 = 'output_new7.csv'
pdf_text_7 = ReadFile(pdf_name_7)
processed_text_7 = ReplaceTermin(pdf_text_7, terminology_dict)
rows_7 = processed_text_7.split('\n')
CreateCsv(csv_name_7, rows_7)
print("Шестой файл обработан!")

pdf_name_8 = 'Kiselev_4.pdf'
csv_name_8 = 'output_new8.csv'
pdf_text_8 = ReadFile(pdf_name_8)
processed_text_8 = ReplaceTermin(pdf_text_8, terminology_dict)
rows_8 = processed_text_8.split('\n')
CreateCsv(csv_name_8, rows_8)
print("Седьмой файл обработан!")

pdf_name_9 = 'Kiselev_5.pdf'
csv_name_9 = 'output_new9.csv'
pdf_text_9 = ReadFile(pdf_name_9)
processed_text_9 = ReplaceTermin(pdf_text_9, terminology_dict)
rows_9 = processed_text_9.split('\n')
CreateCsv(csv_name_9, rows_9)
print("Восьмой файл обработан!")
