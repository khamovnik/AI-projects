from dict import terminology_dict
from functions import ReadFile, ReplaceTermin, CreateCsv

pdf_name = 'Semen.pdf'
csv_name = 'output_new.csv'
pdf_text = ReadFile(pdf_name)
processed_text = ReplaceTermin(pdf_text, terminology_dict)
rows = processed_text.split('\n')
CreateCsv(csv_name, rows)
print("Gemacht!")

