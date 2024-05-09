import os
import re

files = os.listdir('samples')

# Функция для извлечения числа из названия файла
def extract_number(file_name):
    match = re.search(r'\d+', file_name)
    if match:
        return int(match.group())
    return 0

# Находим файл с наибольшим числовым значением в названии
max_number_file = max(files, key=extract_number)

# Извлекаем число из названия файла
max_number = extract_number(max_number_file)

print(f"Файл с наибольшим числовым значением: {max_number_file}")
print(f"Значение числа: {max_number}")