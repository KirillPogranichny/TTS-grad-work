import warnings
import sys
import time
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dnn_tts_torch.models.ssrn import SSRN
from hparams import HParams as hp
from logger import Logger
from utils import get_last_checkpoint_file_name, load_checkpoint, save_checkpoint
from dnn_tts_torch.datasets.data_loader import SSRNDataLoader

warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            'ljspeech',
            'ruspeech'],
        help='dataset name')
    args = parser.parse_args()

    if args.dataset == 'ljspeech':
        from dnn_tts_torch.datasets.lj_speech import LJSpeech as SpeechDataset
    else:
        from dnn_tts_torch.datasets.ru_speech import RUSpeech as SpeechDataset

    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    if use_gpu:
        torch.backends.cudnn.benchmark = True

    train_data_loader = SSRNDataLoader(ssrn_dataset=SpeechDataset(
        ['mags', 'mels']), batch_size=24, mode='train')
    valid_data_loader = SSRNDataLoader(ssrn_dataset=SpeechDataset(
        ['mags', 'mels']), batch_size=24, mode='valid')

    ssrn = SSRN().cuda()

    optimizer = torch.optim.Adam(ssrn.parameters(), lr=hp.ssrn_lr)

    start_timestamp = int(time.time() * 1000)
    start_epoch = 0
    global_step = 0

    logger = Logger(args.dataset, 'ssrn')

    # load the last checkpoint if exists
    last_checkpoint_file_name = get_last_checkpoint_file_name(logger.logdir)
    if last_checkpoint_file_name:
        print("loading the last checkpoint: %s" % last_checkpoint_file_name)
        start_epoch, global_step = load_checkpoint(
            last_checkpoint_file_name, ssrn, optimizer)

    def get_lr():
        return optimizer.param_groups[0]['lr']

    def lr_decay(step, warmup_steps=1000):
        new_lr = hp.ssrn_lr * warmup_steps ** 0.5 * \
            min((step + 1) * warmup_steps ** -1.5, (step + 1) ** -0.5)
        optimizer.param_groups[0]['lr'] = new_lr

    def train(train_epoch, phase='train'):
        global global_step

        lr_decay(global_step)
        print("epoch %3d with lr=%.02e" % (train_epoch, get_lr()))

        ssrn.train() if phase == 'train' else ssrn.eval()
        torch.set_grad_enabled(
            True) if phase == 'train' else torch.set_grad_enabled(False)
        data_loader = train_data_loader if phase == 'train' else valid_data_loader

        it = 0
        running_loss = 0.0
        running_l1_loss = 0.0

        pbar = tqdm(
            data_loader,
            unit="audios",
            unit_scale=data_loader.batch_size,
            disable=hp.disable_progress_bar)
        for batch in pbar:
            M, S = batch['mags'], batch['mels']
            M = M.permute(0, 2, 1)  # TODO: because of pre processing
            S = S.permute(0, 2, 1)  # TODO: because of pre processing

            M.requires_grad = False
            M = M.cuda()
            S = S.cuda()

            Z_logit, Z = ssrn(S)

            l1_loss = F.l1_loss(Z, M)

            loss = l1_loss

            if phase == 'train':
                lr_decay(global_step)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1

            it += 1

            loss = loss.item()
            l1_loss = l1_loss.item()
            running_loss += loss
            running_l1_loss += l1_loss

            if phase == 'train':
                # update the progress bar
                pbar.set_postfix({
                    'l1': "%.05f" % (running_l1_loss / it)
                })
                logger.log_step(phase, global_step, {'loss_l1': l1_loss}, {
                                'mags-true': M[:1, :, :], 'mags-pred': Z[:1, :, :], 'mels': S[:1, :, :]})
                if global_step % 5000 == 0:
                    # checkpoint at every 5000th step
                    save_checkpoint(
                        logger.logdir,
                        train_epoch,
                        global_step,
                        ssrn,
                        optimizer)

        epoch_loss = running_loss / it
        epoch_l1_loss = running_l1_loss / it

        logger.log_epoch(phase, global_step, {'loss_l1': epoch_l1_loss})

        return epoch_loss

    since = time.time()
    epoch = start_epoch
    while True:
        train_epoch_loss = train(epoch, phase='train')
        time_elapsed = time.time() - since
        time_str = (f'total time elapsed: '
                    f'{time_elapsed // 3600:.0f}h '
                    f'{time_elapsed % 3600 // 60:.0f}m '
                    f'{time_elapsed % 60:.0f}s')
        print(f'train epoch loss {train_epoch_loss}, step={
        global_step}, {time_str}')

        valid_epoch_loss = train(epoch, phase='valid')
        print(f'valid epoch loss {valid_epoch_loss}')

        epoch += 1
        if global_step >= hp.text2mel_max_iteration:
            print(f'max step {hp.text2mel_max_iteration} (current step {
            global_step}) reached, exiting...')
            sys.exit(0)
