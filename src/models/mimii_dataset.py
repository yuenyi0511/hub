"""
MIMII 数据集加载器 (支持 ResNet 和 AutoEncoder)
核心修复：自动跳过启动/停止过渡段，只提取设备稳定工作的中间段音频。
"""
import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from pathlib import Path
import random

class MIMIIDataset(Dataset):
    def __init__(self, data_dir, device_type=None, n_mels=64, n_fft=1024,
                 hop_length=512, sr=16000, segment_duration=1.0, augment=False):
        self.data_dir = Path(data_dir)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.segment_duration = segment_duration
        self.segment_samples = int(segment_duration * sr)
        self.augment = augment

        self.file_paths = []
        self.labels = []

        if device_type:
            device_roots = [self.data_dir / device_type]
        else:
            device_roots = [d for d in self.data_dir.iterdir() if d.is_dir()]

        for dev_root in device_roots:
            nested = [d for d in dev_root.iterdir() if d.is_dir()]
            devs_to_scan = nested if nested else [dev_root]

            for dev in devs_to_scan:
                for split_dir in ['train', 'test']:
                    split_path = dev / split_dir
                    if not split_path.exists():
                        continue

                    for wav_path in split_path.glob("*.wav"):
                        fname = wav_path.name.lower()
                        if '_normal_' in fname:
                            self.file_paths.append(str(wav_path))
                            self.labels.append(0)
                        elif '_anomaly_' in fname:
                            self.file_paths.append(str(wav_path))
                            self.labels.append(1)

        print(f"[MIMIIDataset] {device_type or '全部'}: {len(self.file_paths)} 个文件")
        if self.file_paths:
            print(f"  示例: {self.file_paths[0]}")
        print(f"  - Normal:  {sum(1 for l in self.labels if l==0)} 个")
        print(f"  - Anomaly: {sum(1 for l in self.labels if l==1)} 个")

        # 逐设备计算归一化统计量
        self.global_mean = 0.0
        self.global_std = 1.0
        normal_specs = []
        for i in range(min(800, len(self.file_paths))):
            if self.labels[i] == 0:
                try:
                    mel = self._load_and_extract(i, training=False)
                    normal_specs.append(mel)
                except Exception:
                    pass
        if normal_specs:
            all_specs = np.concatenate(normal_specs, axis=1)
            self.global_mean = float(np.mean(all_specs))
            self.global_std = float(np.std(all_specs))
            print(f"  [归一化] mean={self.global_mean:.3f}, std={self.global_std:.3f}")

    def _load_and_extract(self, idx, training=True):
        filepath = self.file_paths[idx]
        audio, _ = librosa.load(filepath, sr=self.sr, duration=None)
        total_samples = len(audio)
        
        # MIMII 音频 10 秒，工作段约在 2~8 秒之间
        work_start = int(2.0 * self.sr)
        work_end = int(8.0 * self.sr)
        work_end = min(work_end, total_samples)
        work_len = work_end - work_start

        if work_len < self.segment_samples:
            audio = audio[:self.segment_samples]
            if len(audio) < self.segment_samples:
                audio = np.pad(audio, (0, self.segment_samples - len(audio)))
        else:
            if training and self.augment:
                offset = random.randint(0, work_len - self.segment_samples)
                audio = audio[work_start + offset : work_start + offset + self.segment_samples]
            else:
                center = (work_start + work_end) // 2
                half = self.segment_samples // 2
                audio = audio[max(0, center-half) : center+half]

        return self._extract_mel_spectrogram(audio)

    def _extract_mel_spectrogram(self, audio):
        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return log_mel

    def _add_noise(self, mel_spec, snr_db=30):
        noise = np.random.normal(0, 1e-4, mel_spec.shape)
        return mel_spec + noise

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filepath = self.file_paths[idx]
        label = self.labels[idx]

        try:
            mel_spec = self._load_and_extract(idx, training=self.augment)
            if self.augment:
                mel_spec = self._add_noise(mel_spec)

            mel_spec = (mel_spec - self.global_mean) / (self.global_std + 1e-8)
            mel_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
            return mel_tensor, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            time_frames = int(self.segment_duration * self.sr // self.hop_length) + 1
            fake_spec = torch.zeros((1, self.n_mels, time_frames), dtype=torch.float32)
            return fake_spec, torch.tensor(label, dtype=torch.long)


class MIMIIOnlyNormalDataset(Dataset):
    def __init__(self, data_dir, device_type=None, n_mels=64, n_fft=1024,
                 hop_length=512, sr=16000, segment_duration=1.0, augment=False):
        self.full_dataset = MIMIIDataset(
            data_dir, device_type, n_mels, n_fft, hop_length, sr, segment_duration, augment
        )
        self.indices = [i for i, l in enumerate(self.full_dataset.labels) if l == 0]
        print(f"[MIMIIOnlyNormal] {device_type or '全部'}: {len(self.indices)} 个 Normal")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.full_dataset[self.indices[idx]]