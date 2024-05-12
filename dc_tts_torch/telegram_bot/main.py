import asyncio
import logging
import os

from aiogram import Bot, Dispatcher
from aiogram.enums.parse_mode import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram import flags
from aiogram.fsm.context import FSMContext

import utils
from states import Form

import config
from handlers import router


async def main():
    bot = Bot(token=config.BOT_TOKEN, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    # path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # os.chdir(path)
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
