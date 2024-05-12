from aiogram.fsm.state import StatesGroup, State


class Form(StatesGroup):
    select_dataset = State()  # Состояние выбора датасета
    enter_text = State()      # Состояние ввода текста
