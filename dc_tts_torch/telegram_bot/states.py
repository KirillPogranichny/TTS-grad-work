from aiogram.fsm.state import StatesGroup, State


class Form(StatesGroup):
    enter_text = State()
