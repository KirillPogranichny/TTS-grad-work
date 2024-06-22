import re
from num2words import num2words
import pymorphy2

morph = pymorphy2.MorphAnalyzer()

# Словари для склонения числительных по падежам и родам
units_masculine = {
    'nominative': ["один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"],
    'genitive': ["одного", "двух", "трех", "четырех", "пяти", "шести", "семи", "восьми", "девяти"],
    'dative': ["одному", "двум", "трем", "четырем", "пяти", "шести", "семи", "восьми", "девяти"],
    'accusative': ["один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"],
    'instrumental': ["одним", "двумя", "тремя", "четырьмя", "пятью", "шестью", "семью", "восемью", "девятью"],
    'prepositional': ["одном", "двух", "трех", "четырех", "пяти", "шести", "семи", "восьми", "девяти"]
}

units_feminine = {
    'nominative': ["одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"],
    'genitive': ["одной", "двух", "трех", "четырех", "пяти", "шести", "семи", "восьми", "девяти"],
    'dative': ["одной", "двум", "трем", "четырем", "пяти", "шести", "семи", "восьми", "девяти"],
    'accusative': ["одну", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"],
    'instrumental': ["одной", "двумя", "тремя", "четырьмя", "пятью", "шестью", "семью", "восемью", "девятью"],
    'prepositional': ["одной", "двух", "трех", "четырех", "пяти", "шести", "семи", "восьми", "девяти"]
}

units_neuter = {
    'nominative': ["одно", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"],
    'genitive': ["одного", "двух", "трех", "четырех", "пяти", "шести", "семи", "восьми", "девяти"],
    'dative': ["одному", "двум", "трем", "четырем", "пяти", "шести", "семи", "восьми", "девяти"],
    'accusative': ["одно", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"],
    'instrumental': ["одним", "двумя", "тремя", "четырьмя", "пятью", "шестью", "семью", "восемью", "девятью"],
    'prepositional': ["одном", "двух", "трех", "четырех", "пяти", "шести", "семи", "восьми", "девяти"]
}

tens = {
    'nominative': ["десять", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"],
    'genitive': ["десяти", "двадцати", "тридцати", "сорока", "пятидесяти", "шестидесяти", "семидесяти", "восьмидесяти", "девяноста"],
    'dative': ["десяти", "двадцати", "тридцати", "сорока", "пятидесяти", "шестидесяти", "семидесяти", "восьмидесяти", "девяноста"],
    'accusative': ["десять", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"],
    'instrumental': ["десятью", "двадцатью", "тридцатью", "сорока", "пятьюдесятью", "шестьюдесятью", "семьюдесятью", "восемьюдесятью", "девяноста"],
    'prepositional': ["десяти", "двадцати", "тридцати", "сорока", "пятидесяти", "шестидесяти", "семидесяти", "восьмидесяти", "девяноста"]
}

hundreds = {
    'nominative': ["сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"],
    'genitive': ["ста", "двухсот", "трехсот", "четырехсот", "пятисот", "шестисот", "семисот", "восьмисот", "девятисот"],
    'dative': ["ста", "двумстам", "тремстам", "четыремстам", "пятистам", "шестистам", "семистам", "восьмистам", "девятистам"],
    'accusative': ["сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"],
    'instrumental': ["ста", "двумястами", "тремястами", "четырьмястами", "пятьюстами", "шестьюстами", "семьюстами", "восемьюстами", "девятьюстами"],
    'prepositional': ["ста", "двухстах", "трехстах", "четырехстах", "пятистах", "шестистах", "семистах", "восьмистах", "девятистах"]
}

thousands = {
    'nominative': ["тысяча", "тысячи", "тысяч"],
    'genitive': ["тысячи", "тысяч", "тысяч"],
    'dative': ["тысяче", "тысячам", "тысячам"],
    'accusative': ["тысячу", "тысячи", "тысяч"],
    'instrumental': ["тысячей", "тысячами", "тысячами"],
    'prepositional': ["тысяче", "тысячах", "тысячах"]
}

millions = {
    'nominative': ["миллион", "миллиона", "миллионов"],
    'genitive': ["миллиона", "миллионов", "миллионов"],
    'dative': ["миллиону", "миллионам", "миллионам"],
    'accusative': ["миллион", "миллиона", "миллионов"],
    'instrumental': ["миллионом", "миллионами", "миллионами"],
    'prepositional': ["миллионе", "миллионах", "миллионах"]
}

cases = {
    'именительный': 'nominative',
    'родительный': 'genitive',
    'дательный': 'dative',
    'винительный': 'accusative',
    'творительный': 'instrumental',
    'предложный': 'prepositional'
}

def get_gender_and_case(word):
    parsed = morph.parse(word)[0]
    gender = parsed.tag.gender
    case = parsed.tag.case
    return gender, case

def decline_number(word, case, gender):
    words = word.split()
    declined_words = []

    for w in words:
        if gender == 'masc':
            units = units_masculine
        elif gender == 'femn':
            units = units_feminine
        elif gender == 'neut':
            units = units_neuter
        else:
            units = units_masculine  # Default to masculine if gender is unknown

        if w in units['nominative']:
            declined_words.append(units[case][units['nominative'].index(w)])
        elif w in tens['nominative']:
            declined_words.append(tens[case][tens['nominative'].index(w)])
        elif w in hundreds['nominative']:
            declined_words.append(hundreds[case][hundreds['nominative'].index(w)])
        elif w.endswith("тысяча") or w.endswith("тысячи") or w.endswith("тысяч"):
            if len(words) == 1:
                declined_words.append(thousands[case][thousands['nominative'].index(w)])
            else:
                declined_words.append(thousands[case][1])
        elif w.endswith("миллион") or w.endswith("миллиона") or w.endswith("миллионов"):
            if len(words) == 1:
                declined_words.append(millions[case][millions['nominative'].index(w)])
            else:
                declined_words.append(millions[case][1])
        else:
            declined_words.append(w)

    return ' '.join(declined_words)

def process_text(text):
    words = text.split()
    for i, word in enumerate(words):
        if word.isdigit():
            number_in_words = num2words(int(word), lang='ru')
            next_word = words[i + 1] if i + 1 < len(words) else ""
            gender, case = get_gender_and_case(next_word)
            declined_number = decline_number(number_in_words, cases[case], gender)
            words[i] = declined_number
    return ' '.join(words)

# Пример использования
text = "2017 год был хорошим"
processed_text = process_text(text)
print(processed_text)
