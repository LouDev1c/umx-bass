from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from nnAudio.features import CQT
import numpy as np

try:
    from asteroid_filterbanks.enc_dec import Encoder, Decoder
    from asteroid_filterbanks.transforms import to_torchaudio, from_torchaudio
    from asteroid_filterbanks import torch_stft_fb
except ImportError:
    pass


def make_filterbanks(n_fft=4096, n_hop=1024, center=False, sample_rate=44100.0, method=None, use_cqt=None):
    window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)

    if method == "stft":
        encoder = TorchSTFT(n_fft=n_fft, n_hop=n_hop, window=window, center=center)
        decoder = TorchISTFT(n_fft=n_fft, n_hop=n_hop, window=window, center=center)
    elif method == "cqt":
        encoder = nnAudioCQT(n_fft=n_fft, n_hop=n_hop, center=center, window=window, sample_rate=sample_rate)
        decoder = nnAudioICQT(n_fft=n_fft, n_hop=n_hop, center=center, window=window, sample_rate=sample_rate)
        use_cqt = True
    elif method == "asteroid":
        fb = torch_stft_fb.TorchSTFTFB.from_torch_args(
            n_fft=n_fft,
            hop_length=n_hop,
            win_length=n_fft,
            window=window,
            center=center,
            sample_rate=sample_rate,
        )
        encoder = AsteroidSTFT(fb)
        decoder = AsteroidISTFT(fb)
    else:
        raise NotImplementedError
    return encoder, decoder


class nnAudioCQT(nn.Module):
    def __init__(
        self,
        n_fft: int = 4096,
        n_hop: int = 1024,
        center: bool = False,
        window: Optional[nn.Parameter] = None,
        sample_rate: float = 44100.0
    ):
        super(nnAudioCQT, self).__init__()
        
        # CQT参数设置
        self.n_bins = int(np.ceil(np.log2(sample_rate/2 / 20) * 12))
        self.hop_length = n_hop
        self.center = center
        self.sample_rate = sample_rate
        
        # 使用nnAudio的CQT实现
        self.cqt = CQT(
            sr=sample_rate,
            hop_length=n_hop,
            n_bins=self.n_bins,
            bins_per_octave=12,  # 标准音阶
            fmin=20,  # 最低频率设为20Hz，适合贝斯
            window='hann',
            center=center,
            pad_mode='reflect'
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """CQT forward path
        Args:
            x (Tensor): audio waveform of
                shape (nb_samples, nb_channels, nb_timesteps)
        Returns:
            CQT (Tensor): complex cqt of
                shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
                last axis is stacked real and imaginary
        """
        shape = x.size()  # (nb_samples, nb_channels, nb_timesteps)
        
        # pack batch
        x = x.view(-1, shape[-1])
        
        # 对每个通道分别进行CQT
        complex_cqt = []
        for ch in range(shape[1]):
            # 获取当前通道的音频
            ch_audio = x[ch::shape[1]]
            # 计算CQT
            cqt_ch = self.cqt(ch_audio)  # 检查输出格式
            
            # 确保维度顺序正确
            cqt_ch = cqt_ch.permute(0, 2, 1)  # (batch, time, freq)
            
            # 如果CQT输出已经是复数形式
            if torch.is_complex(cqt_ch):
                # 直接转换为实部和虚部的堆叠
                cqt_ch = torch.stack([cqt_ch.real, cqt_ch.imag], dim=-1)
            else:
                # 如果输出是幅度谱，需要转换为复数形式
                # 这里假设相位为0，可以根据需要修改
                cqt_ch = torch.stack([cqt_ch, torch.zeros_like(cqt_ch)], dim=-1)
            
            complex_cqt.append(cqt_ch)
        
        # 合并所有通道
        cqt_f = torch.stack(complex_cqt, dim=1)  # (batch, channel, time, freq, complex)
        
        # 调整维度顺序以匹配STFT输出
        cqt_f = cqt_f.permute(0, 1, 3, 2, 4)  # (batch, channel, freq, time, complex)
        
        return cqt_f


class nnAudioICQT(nn.Module):
    def __init__(
        self,
        n_fft: int = 4096,
        n_hop: int = 1024,
        center: bool = False,
        window: Optional[nn.Parameter] = None,
        sample_rate: float = 44100.0
    ):
        super(nnAudioICQT, self).__init__()
        
        # 保持与CQT相同的参数
        self.n_bins = int(np.ceil(np.log2(sample_rate/2 / 20) * 12))
        self.hop_length = n_hop
        self.center = center
        self.sample_rate = sample_rate
        
        # 使用nnAudio的ICQT实现
        self.icqt = CQT(
            sr=sample_rate,
            hop_length=n_hop,
            n_bins=self.n_bins,
            bins_per_octave=12,
            fmin=20,
            window='hann',
            center=center,
            pad_mode='reflect'
        )
        
    def forward(self, X: Tensor, length: Optional[int] = None) -> Tensor:
        """Inverse CQT path
        Args:
            X (Tensor): complex cqt of
                shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
            length (int, optional): audio signal length to crop the signal
        Returns:
            x (Tensor): audio waveform of
                shape (nb_samples, nb_channels, nb_timesteps)
        """
        shape = X.size()
        X = X.reshape(-1, shape[-3], shape[-2], shape[-1])
        
        # 对每个通道分别进行ICQT
        audio_chunks = []
        for ch in range(shape[1]):
            # 获取当前通道的CQT
            ch_cqt = X[ch::shape[1]]
            # 转换为复数形式
            ch_cqt = torch.view_as_complex(ch_cqt)
            # 计算ICQT
            audio_ch = self.icqt.inverse(ch_cqt)
            audio_chunks.append(audio_ch)
            
        # 合并所有通道
        y = torch.stack(audio_chunks, dim=1)
        
        # 裁剪到指定长度
        if length is not None:
            y = y[..., :length]
            
        y = y.reshape(shape[:-3] + y.shape[-1:])
        return y


class TorchSTFT(nn.Module):
    def __init__(
        self,
        n_fft: int = 4096,
        n_hop: int = 1024,
        center: bool = False,
        window: Optional[nn.Parameter] = None,
    ):
        super(TorchSTFT, self).__init__()
        if window is None:
            self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
        else:
            self.window = window

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

    def forward(self, x: Tensor) -> Tensor:
        """STFT forward path
        Args:
            x (Tensor): audio waveform of
                shape (nb_samples, nb_channels, nb_timesteps)
        Returns:
            STFT (Tensor): complex stft of
                shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
                last axis is stacked real and imaginary
        """

        shape = x.size()

        # pack batch
        x = x.view(-1, shape[-1])

        complex_stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            window=self.window,
            center=self.center,
            normalized=False,
            onesided=True,
            pad_mode="reflect",
            return_complex=True,
        )
        stft_f = torch.view_as_real(complex_stft)
        # unpack batch
        stft_f = stft_f.view(shape[:-1] + stft_f.shape[-3:])
        return stft_f


class TorchISTFT(nn.Module):
    """Multichannel Inverse-Short-Time-Fourier functional
    wrapper for torch.istft to support batches
    Args:
        STFT (Tensor): complex stft of
            shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
            last axis is stacked real and imaginary
        n_fft (int, optional): transform FFT size. Defaults to 4096.
        n_hop (int, optional): transform hop size. Defaults to 1024.
        window (callable, optional): window function
        center (bool, optional): If True, the signals first window is
            zero padded. Centering is required for a perfect
            reconstruction of the signal. However, during training
            of spectrogram models, it can safely turned off.
            Defaults to `true`
        length (int, optional): audio signal length to crop the signal
    Returns:
        x (Tensor): audio waveform of
            shape (nb_samples, nb_channels, nb_timesteps)
    """
    def __init__(
        self,
        n_fft: int = 4096,
        n_hop: int = 1024,
        center: bool = False,
        sample_rate: float = 44100.0,
        window: Optional[nn.Parameter] = None,
    ) -> None:
        super(TorchISTFT, self).__init__()

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center
        self.sample_rate = sample_rate

        if window is None:
            self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
        else:
            self.window = window

    def forward(self, X: Tensor, length: Optional[int] = None) -> Tensor:
        shape = X.size()
        X = X.reshape(-1, shape[-3], shape[-2], shape[-1])
        y = torch.istft(
            torch.view_as_complex(X),
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            window=self.window,
            center=self.center,
            normalized=False,
            onesided=True,
            length=length,
        )
        y = y.reshape(shape[:-3] + y.shape[-1:])
        return y


class AsteroidSTFT(nn.Module):
    def __init__(self, fb):
        super(AsteroidSTFT, self).__init__()
        self.enc = Encoder(fb)

    def forward(self, x):
        aux = self.enc(x)
        return to_torchaudio(aux)


class AsteroidISTFT(nn.Module):
    def __init__(self, fb):
        super(AsteroidISTFT, self).__init__()
        self.dec = Decoder(fb)

    def forward(self, X: Tensor, length: Optional[int] = None) -> Tensor:
        aux = from_torchaudio(X)
        return self.dec(aux, length=length)


class ComplexNorm(nn.Module):
    r"""Compute the norm of complex tensor input.

    Extension of `torchaudio.functional.complex_norm` with mon
    Args:
        mono (bool): Downmix to single channel after applying power norm
            to maximize
    """

    def __init__(self, mono: bool = False):
        super(ComplexNorm, self).__init__()
        self.mono = mono

    def forward(self, spec: Tensor) -> Tensor:
        """
        Args:
            spec: complex_tensor (Tensor): Tensor shape of
                `(..., complex=2)`
        Returns:
            Tensor: Power/Mag of input
                `(...,)`
        """
        # take the magnitude
        spec = torch.abs(torch.view_as_complex(spec))
        # down-mix in the mag domain to preserve energy
        if self.mono:
            spec = torch.mean(spec, 1, keepdim=True)
        return spec
