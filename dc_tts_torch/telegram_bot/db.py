import multiprocessing
import subprocess
import time
import os
import sys


def run_terminal_command(cmd, queue):
    # Открываем терминал и запускаем команду
    process = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)

    # Ждем данных для отправки
    data_to_send = queue.get()
    if data_to_send:
        process.stdin.write(data_to_send)
        process.stdin.flush()

    # Получаем вывод терминала
    stdout, stderr = process.communicate()
    queue.put((stdout, stderr))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    # Команда для запуска скрипта в терминале
    command = [sys.executable, "dc_tts_torch/synthesize.py", "--dataset=ljspeech"]

    # Данные, которые нужно отправить в терминал
    data_to_send = 'I wanna be yours q'

    # Запускаем терминал с командой в отдельном процессе
    queue = multiprocessing.Queue()

    # Запускаем терминал с командой в отдельном процессе
    terminal_process = multiprocessing.Process(target=run_terminal_command, args=(command, queue))

    # Запускаем процесс
    terminal_process.start()

    # Отправляем данные в процесс терминала через очередь
    queue.put(data_to_send)

    # Ждем завершения процесса терминала и получения данных из очереди
    terminal_process.join()

    # Получаем вывод терминала из очереди
    stdout, stderr = queue.get()
    print(f'STDOUT: {stdout}')
    print(f'STDERR: {stderr}')

    # Получаем вывод терминала
    stdout, stderr = terminal_process.communicate()
    print(f'STDOUT: {stdout}')
    print(f'STDERR: {stderr}')