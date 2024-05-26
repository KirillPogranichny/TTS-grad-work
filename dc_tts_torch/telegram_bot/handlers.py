import socket
import os
import subprocess
import sys
import asyncio

from aiogram import Router, types, Bot
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
# from aiogram.methods.send_chat_action import SendChatAction

import kb
import text
import config
from states import Form
from utils import send_message_to_script

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
async def choose_lang_handler(callback: types.CallbackQuery, state: FSMContext) -> None:
    global process
    await bot.answer_callback_query(callback.id)
    await state.set_state(Form.enter_text)
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if callback.data == "generate_en":
        await state.update_data(language='en')
        # await callback.message.answer(text.en_lang) # Вынести в следующую итерацию
        try:
            process = subprocess.Popen(
                [sys.executable, "dc_tts_torch/synthesize.py", "--dataset=ljspeech"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            # data['text'] = message.text
            # subprocess.run([sys.executable, "dc_tts_torch/synthesize.py", "--dataset=ljspeech"], check=True)
            # await send_message_to_script(sock)
            # await utils.send_audiofile(message.chat.id, 'dc_tts_torch/samples/en')
        except subprocess.CalledProcessError as e:
            print(f"Произошла ошибка: {e.output}")
    else:
        await state.update_data(language='ru')
        try:
            process = subprocess.Popen(
                [sys.executable, "dc_tts_torch/synthesize.py", "--dataset=ljspeech"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        # await callback.message.answer(text.ru_lang) # Вынести в следующую итерацию
        except subprocess.CalledProcessError as e:
            print(f"Произошла ошибка: {e.output}")


@router.message()
async def post_handler(message: Message, state: FSMContext) -> None:
    global process
    user_text = message.text
    await send_message_to_script(process, user_text)
    await state.finish()

    # data = await state.get_data()
    # if data['language'] == "en":
    #     try:
    #         data['text'] = message.text
    #
    #         await utils.send_audiofile(message.chat.id, 'dc_tts_torch/samples/en')
    #     except subprocess.CalledProcessError as e:
    #         print(f"Произошла ошибка: {e.output}")
    # elif data['language'] == "ru":
    #     try:
    #         subprocess.run([sys.executable, "dc_tts_torch/synthesize.py", "--dataset=ruspeech"], check=True)
    #         await utils.send_audiofile(message.chat.id, 'dc_tts_torch/samples/ru')
    #     except subprocess.CalledProcessError as e:
    #         print(f"Произошла ошибка: {e.output}")

    # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # sock.bind(('127.0.0.1', 12345))
    # sock.listen(1)
    #
    #
    # async def send_message_to_script(sock: socket.socket):
    #     # Отправка сообщения в скрипт через сокет
    #     sock.sendall(data['text'].encode())

    # @dp.message_handler()
    # async def handle_message(message: types.Message):
    #     await send_message_to_script(message)

    # await bot.send_chat_action(chat_id=message.chat.id, action=types.ChatAction.TYPING)
    # os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    # await bot.send_chat_action(chat_id=message.chat.id, action=types.ChatAction.READING)

# @router.message(state=Form.enter_text)
# async def process_text(message: types.Message, state: FSMContext) -> None:
#     global process
#     text = message.text
#     if process:
#         process.stdin.write(text + '\n')
#         process.stdin.flush()
#
#         # Reading output for debugging
#         output, error = process.communicate()
#         if output:
#             await bot.send_message(message.chat.id, f"Output: {output}")
#         if error:
#             await bot.send_message(message.chat.id, f"Error: {error}")
#
#         await state.clear()
#     else:
#         await bot.send_message(message.chat.id, "Process not running. Please restart the command.")
#         await state.clear()
