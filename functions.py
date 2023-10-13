import PyPDF2
import pandas as pd

def ReadFile(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
def CreateCsv(file_path, data):
    series = pd.Series(data)
    series.to_csv(file_path, index=False, header=False)
def ReplaceTermin(text, terminology_dict):
    for key, value in terminology_dict.items():
        text = text.replace(key, value)
    return text
