# launch_dataset.py

import torchaudio
import torch
import torchaudio.transforms as transforms
import random
import numpy as np


class AudioUtil():

    @classmethod
    def open(self, file_path):
        return torchaudio.load(file_path)

    @classmethod
    def resample(cls, aud, newsr):
        sig, sr = aud
        num_channels = sig.shape[0]
        if sr != newsr:
            resig = torchaudio.transforms.Resample(sr, newsr)(sig)
        else:
            resig = sig
        return (resig, newsr)

    @classmethod
    def rechannel(cls, aud):
        resig, newsr = aud
        num_channels = resig.shape[0]
        if num_channels == 1:
            return torch.cat([resig, resig, resig]), newsr
        elif num_channels == 2:
            return (torch.cat([resig, torch.mean(resig, axis = 0).reshape(1, -1)]), newsr)

    @classmethod
    def pad_trunc(cls, aud, max_s):  # max_s = 5
        sig, sr = aud
        num_rows, sig_len = sig.shape
        if num_rows < 3:
            print(sig.shape)
        max_len = sr * max_s

        if (sig_len > max_len):
          # Truncate the signal to the given length
            sig = sig[:,:max_len]

        elif (sig_len < max_len):
          # Length of padding to add at the beginning and end of the signal

            pad_end_len = max_len - sig_len
            pad_end = torch.zeros((num_rows, pad_end_len))
            sig = torch.cat((sig, pad_end), 1)

        return (sig, sr)


    @classmethod
    def time_shift(cls, aud, shift_limit):
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    @classmethod
    def spectro_gram(cls, aud, n_mels=224, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc

        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
#         print('type of spec after melspectrogram: ', type(spec), spec.shape)
        return spec

    @classmethod
    def spectro_augment(cls, spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec

    @classmethod
    def normalize(cls, spec):
        mean, std = spec.mean(), spec.std()
        return (spec - mean)/std



class train_ds(torch.utils.data.Dataset):

    def __init__(self, df, BIRD_CODE, img_size = 224, sr = 32000, duration = 5):
        # df is pd.pandas(df)
        self.df = df
        self.img_size = img_size
        self.sr = sr
        self.duration = duration
        self.BIRD_CODE = BIRD_CODE  # 'apfly': 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):

        sample = self.df.loc[idx, :]
        file_path = '/kaggle/input/birdsong-recognition/train_audio/' + sample['ebird_code'] + '/' + sample['filename']

        aud = AudioUtil().open(file_path)
        reaud = AudioUtil().resample(aud, self.sr)
        chaaud = AudioUtil().rechannel(reaud)
        dur_aud = AudioUtil().pad_trunc(chaaud, self.duration)
        shift_aud = AudioUtil().time_shift(dur_aud, shift_limit = 0.5)
        sgram = AudioUtil().spectro_gram(shift_aud, n_mels=self.img_size, n_fft=1024, hop_len=None)  # size would be 3, 244, 313; tensor size
        aug_sgram = AudioUtil().spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return idx, aug_sgram, self.BIRD_CODE[sample['ebird_code']]


class test_ds(torch.utils.data.Dataset):
    def __init__(self, df, img_size = 224, sr = 32000, duration = 5, test_sample = False):
        self.df = df
        self.img_size = img_size
        self.sr = sr
        self.duration = duration
        self.test_sample = test_sample

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):

        sample = self.df.loc[idx, :]
        site = sample.site
        audio_id = sample.audio_id
        duration = sample.seconds
        if self.test_sample:
            file_path = '/kaggle/input/birdcall-check/test_audio/' + audio_id + '.mp3'
        else:
            file_path = '/kaggle/input/birdsong-recognition/test_audio/' + audio_id + '.mp3'
        aud = AudioUtil().open(file_path)
        reaud = AudioUtil().resample(aud, self.sr)  # reaud = aud, sample_rate


        if site == 'site_3':
            images = []
            start = 0
            while start <= (duration-5) * self.sr:
                end = 5 * self.sr + start

                if end > duration:
                    reaud1 = reaud[0][:, start:], reaud[1]
                    reaud1 = AudioUtil.pad_trunc(reaud1, self.duration)

                else:
                    reaud1 = reaud[0][:, start:end], reaud[1]

                reaud1 = AudioUtil.rechannel(reaud1)
                sgram = AudioUtil().spectro_gram(reaud1, n_mels=self.img_size, n_fft=1024, hop_len=None)  # size would be 3, 244, 313
                # 3 = channe
                # 224: number of mels
                # 313 = sample_rate / hop_len * time = sample_rate / (n_fft/2) * time
                images.append(sgram)
                start = end
            return np.array(images), audio_id, site

        else:
            end_seconds = int(sample.seconds)
            start_seconds = int(end_seconds - 5)

            start_index = self.sr * start_seconds
            end_index = self.sr * end_seconds
            print(reaud[0].shape, start_index, end_index, 'ii')
            reaud1 = reaud[0][:, start_index:end_index], reaud[1]
            reaud1 = AudioUtil().rechannel(reaud1)
            sgram = AudioUtil().spectro_gram(reaud1, n_mels=self.img_size, n_fft=1024, hop_len=None)  # size would be 3, 244, 313
            return sgram, audio_id, site  # 1, 3, 224, 313
        
