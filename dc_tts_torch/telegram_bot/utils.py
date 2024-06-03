from aiogram import F, Router, types, Dispatcher, Bot
from aiogram.enums import ParseMode
from aiogram.types.input_file import FSInputFile
import os

import config


bot = Bot(token=config.BOT_TOKEN, parse_mode=ParseMode.HTML, disable_web_page_preview=True)


async def send_audiofile(chat_id: int, path: str):
    files = os.listdir(path)
    wav_files = [f for f in files if f.endswith('.wav')]
    sorted_files = sorted(wav_files, key=lambda x: int(x.split('-')[0]))
    max_file = sorted_files[-1]
    await bot.send_audio(chat_id, FSInputFile(os.path.join(path, max_file)))
