from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

menu = [
    [InlineKeyboardButton(text="🇬🇧 Английская речь", callback_data="generate_en"),
     InlineKeyboardButton(text="🇷🇺 Русская речь", callback_data="generate_ru")]
]
menu = InlineKeyboardMarkup(inline_keyboard=menu)
