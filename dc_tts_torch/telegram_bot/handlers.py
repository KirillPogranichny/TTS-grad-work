import os
import subprocess
import sys
import asyncio

from aiogram import Router, Bot
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, CallbackQuery
from threading import Thread

import kb
import text
import config
from states import Form
from utils import send_audiofile

bot = Bot(token=config.BOT_TOKEN, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
router = Router()
process = None


# Обработчик команды /start
@router.message(Command("start"))
async def start_handler(message: Message):
    await message.answer(
        text.greet.format(name=message.from_user.full_name), reply_markup=kb.menu)


# Обработчик выбора языка
@router.callback_query(lambda c: c.data in ["generate_en", "generate_ru"])
async def choose_lang_handler(callback: CallbackQuery, state: FSMContext) -> None:
    await bot.answer_callback_query(callback.id)
    await state.set_state(Form.enter_text)
    if callback.data == "generate_en":
        language = 'en'
        lang = text.en_lang
    else:
        language = 'ru'
        lang = text.ru_lang
    await state.update_data(language=language)
    await callback.message.answer(lang)


def run_subprocess(command):
    subprocess.run(command, check=True)


async def send_typing_action(chat_id):
    while True:
        await bot.send_chat_action(chat_id, action="typing")
        await asyncio.sleep(1)


@router.message(Form.enter_text)
async def text_handler(message: Message, state: FSMContext) -> None:
    user_data = await state.get_data()
    language = user_data['language']
    user_text = message.text

    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    script = "dc_tts_torch/synthesize.py"
    dataset = "ljspeech" if language == 'en' else "ruspeech"

    command = [sys.executable, script, f"--dataset={dataset}", f"--text={user_text}"]

    typing_task = asyncio.create_task(send_typing_action(message.chat.id))

    try:
        thread = Thread(target=run_subprocess, args=(command,))
        thread.start()

        while thread.is_alive():
            await asyncio.sleep(1)
        await message.answer("Процесс успешно завершен!\n"
                             "Вот ваш аудиофайл:")
        await send_audiofile(message.chat.id, f'dc_tts_torch/samples/{language}')
    except subprocess.CalledProcessError as e:
        print(f"Произошла ошибка: {e.output}")
        await message.answer("Произошла ошибка при выполнении процесса.")
    finally:
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

    await state.clear()
    await message.answer(
        text.markup.format(name=message.from_user.full_name), reply_markup=kb.menu)
