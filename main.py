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
