# import librosa
# import soundfile
# from hparams import HParams as hp
#
#
# def main():
#     filename = 'samples/en/1-wav.wav'
#     y, _ = librosa.load(filename)
#     steps = float(input("Number of semitones to shift audio file: "))
#     new_y = librosa.effects.pitch_shift(y, sr=hp.sr, n_steps=steps)
#     new_y = librosa.effects.time_stretch(new_y, rate=0.9)
#     soundfile.write("samples/en/pitchShifted.wav", new_y, hp.sr)
#
#
# main()

import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict


class MetricsProcessor:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.event_files = self._find_event_files()
        self.all_metrics = {}
        self.filtered_all_metrics = {}

    def _find_event_files(self):
        return [os.path.join(self.log_dir, f) for f in os.listdir(self.log_dir) if f.startswith('events.out.tfevents')]

    def extract_and_sort_metrics(self, event_file):
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()

        metrics = {}
        for tag in event_acc.Tags()['scalars']:
            metrics[tag] = []
            for scalar_event in event_acc.Scalars(tag):
                metrics[tag].append((scalar_event.step, scalar_event.value))

            # Сортировка списка значений по второму элементу кортежа (значение метрики)
            if 'att' in tag:
                metrics[tag].sort(key=lambda x: x[1], reverse=True)
            elif 'loss' in tag:
                metrics[tag].sort(key=lambda x: x[1])  # Сортировка по возрастанию

        return metrics

    def compute_average(self, metrics):
        grouped_metrics = defaultdict(list)
        for tag, values in metrics.items():
            for step, value in values:
                grouped_metrics[step].append(value)

        averaged_metrics = {step: sum(values) / len(values) for step, values in grouped_metrics.items()}
        return averaged_metrics

    def filter_metrics(self, metrics):
        filtered_metrics = {tag: [item for item in values if item[0] % 5000 == 0] for tag, values in metrics.items()}
        return filtered_metrics

    def sort_metrics_by_value(self, metrics_list, reverse=False):
        return sorted(metrics_list, key=lambda x: x[1], reverse=reverse)

    def process(self):
        for event_file in self.event_files:
            metrics = self.extract_and_sort_metrics(event_file)
            self.all_metrics[event_file] = metrics

            # Фильтрация метрик
            filtered_metrics = self.filter_metrics(metrics)
            self.filtered_all_metrics[event_file] = filtered_metrics

        for event_file, metrics in self.all_metrics.items():
            print(f'Metrics from {event_file}:')
            for tag, values in metrics.items():
                print(f'  {tag}: {values[:5]}')  # Печать первых 5 элементов

            # Вычисление средних значений для att и loss
            att_metrics = {tag: values for tag, values in metrics.items() if 'att' in tag}
            loss_metrics = {tag: values for tag, values in metrics.items() if 'loss' in tag}

            averaged_att = self.compute_average(att_metrics)
            averaged_loss = self.compute_average(loss_metrics)

            # Сортировка средних значений для att по убыванию элемента с индексом 1
            sorted_averaged_att = self.sort_metrics_by_value(list(averaged_att.items()), reverse=True)
            sorted_averaged_loss = self.sort_metrics_by_value(list(averaged_loss.items()), reverse=False)

            # Вывод отсортированных средних значений
            print(f'Sorted averaged att metrics (descending): {sorted_averaged_att[:5]}')  # Печать первых 5 элементов
            print(f'Sorted averaged loss metrics (ascending): {sorted_averaged_loss[:5]}')  # Печать первых 5 элементов

            # Фильтрованные метрики
            filtered_metrics = self.filtered_all_metrics[event_file]
            filtered_att_metrics = {tag: values for tag, values in filtered_metrics.items() if 'att' in tag}
            filtered_loss_metrics = {tag: values for tag, values in filtered_metrics.items() if 'loss' in tag}

            filtered_averaged_att = self.compute_average(filtered_att_metrics)
            filtered_averaged_loss = self.compute_average(filtered_loss_metrics)

            # Сортировка отфильтрованных средних значений для att по убыванию элемента с индексом 1
            sorted_filtered_averaged_att = self.sort_metrics_by_value(list(filtered_averaged_att.items()), reverse=True)
            sorted_filtered_averaged_loss = self.sort_metrics_by_value(list(filtered_averaged_loss.items()), reverse=False)

            # Вывод отсортированных отфильтрованных средних значений
            print(f'Sorted filtered averaged att metrics (descending): {sorted_filtered_averaged_att[:5]}')  # Печать первых 5 элементов
            print(f'Sorted filtered averaged loss metrics (ascending): {sorted_filtered_averaged_loss[:5]}')  # Печать первых 5 элементов

            # Обобщенный лист значений для отсортированных средних значений
            combined_metrics = {}
            for step, att_value in sorted_averaged_att:
                if step in averaged_loss:
                    combined_metrics[step] = (att_value, averaged_loss[step])

            # Сортировка обобщенного списка по сумме индексов
            combined_metrics_list = sorted(
                combined_metrics.items(),
                key=lambda x: (sorted_averaged_att.index((x[0], x[1][0])) + sorted_averaged_loss.index((x[0], x[1][1])))
            )

            print(f'Final sorted combined metrics: {combined_metrics_list[:5]}')

            combined_metrics = {}
            for step, att_value in filtered_averaged_att.items():
                if step in filtered_averaged_loss:
                    combined_metrics[step] = (att_value, filtered_averaged_loss[step])

            # Сортировка обобщенного списка по сумме индексов
            combined_metrics_list = sorted(
                combined_metrics.items(), key=lambda x: (
                sorted_filtered_averaged_att.index((x[0], x[1][0])) + sorted_filtered_averaged_loss.index((x[0], x[1][1])))
            )

            print(f'Final sorted filtered combined metrics: {combined_metrics_list[:5]}')


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
logdir_path = os.path.join(parent_dir, 'logdir')

log_dirs = {
    'ljspeech-ssrn': os.path.join(logdir_path, 'ljspeech-ssrn'),
    'ljspeech-text2mel': os.path.join(logdir_path, 'ljspeech-text2mel'),
    'ruspeech-ssrn': os.path.join(logdir_path, 'ruspeech-ssrn'),
    'ruspeech-text2mel': os.path.join(logdir_path, 'ruspeech-text2mel')
}

key = 'ljspeech-ssrn'
selected_log_dir = log_dirs[key]

processor = MetricsProcessor(selected_log_dir)
processor.process()
