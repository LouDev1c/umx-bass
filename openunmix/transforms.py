from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from nnAudio.Spectrogram import CQT2010v2
import torch.nn.functional as F


def make_filterbanks(
        n_fft: int = 4096,
        n_hop: int = 1024,
        center: bool = False,
        sample_rate: float = 44100.0,
        method: str = None,
):
    window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
    if method == "stft":
        encoder = TorchSTFT(n_fft=n_fft, n_hop=n_hop, window=window, center=center)
        decoder = TorchISTFT(n_fft=n_fft, n_hop=n_hop, window=window, center=center)
    elif method == "hybrid":
        encoder = Hybrid(n_fft=n_fft, n_hop=n_hop, window=window, center=center, sample_rate=sample_rate)
        decoder = Hybrid_Inv(n_fft=n_fft, hop_length=n_hop, window=window, center=center, sample_rate=sample_rate)
    else:
        raise NotImplementedError(f"Unknown method: {method}")
    return encoder, decoder


class Hybrid(nn.Module):
    def __init__(
            self,
            n_fft: int = 4096,
            n_hop: int = 1024,
            center: bool = False,
            window: Optional[nn.Parameter] = None,
            sample_rate: float = 44100.0,
    ):
        super(Hybrid, self).__init__()
        if window is None:
            self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
        else:
            self.window = window

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center
        self.sample_rate = sample_rate

        # 计算STFT的频带数
        self.stft_bins = n_fft // 2 + 1

        # 设置CQT参数
        self.cqt_bins = 84  # 7个八度，每个八度12个音符
        self.cqt = CQT2010v2(
            sr=int(sample_rate),
            hop_length=n_hop,
            fmin=27.5,
            n_bins=self.cqt_bins,
            bins_per_octave=12,
            verbose=False,
        )

        # 计算STFT和CQT的分界频率
        self.crossover_freq = 200  # 200Hz作为分界点
        self.crossover_bin = int(self.crossover_freq * n_fft / sample_rate)

        # 计算STFT部分需要保留的频带数
        self.stft_keep_bins = self.stft_bins - self.crossover_bin

    def forward(self, x: Tensor) -> Tensor:
        """Hybrid forward path
        Args:
            x (Tensor): audio waveform of
                shape (nb_samples, nb_channels, nb_timesteps)
        Returns:
            Tensor: complex hybrid transform of
                shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
        """
        shape = x.size()
        # pack batch
        x = x.view(-1, shape[-1])
        
        # 计算STFT
        stft = torch.stft(
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
        stft = torch.view_as_real(stft)
        
        # 计算CQT
        cqt = self.cqt(x)  # CQT输出已经是复数形式
        
        # 计算STFT的时间帧数
        stft_frames = stft.shape[-2]
        
        # 将CQT转换为与STFT相同的形状
        cqt_resized = F.interpolate(
            torch.abs(cqt).unsqueeze(1),  # [batch, 1, bins, time]
            size=(self.crossover_bin, stft_frames),  # 使用STFT的时间帧数
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [batch, bins, time]
        
        # 将CQT转换为复数形式
        cqt_complex = torch.stack([cqt_resized, torch.zeros_like(cqt_resized)], dim=-1)
        
        # 取STFT的高频部分
        stft_high = stft[:, self.crossover_bin:, :, :]
        
        # 拼接CQT和STFT
        hybrid = torch.cat([cqt_complex, stft_high], dim=1)
        
        # 重塑为正确的形状
        # 从 [batch, bins, time, 2] 转换为 [batch, channels, bins, time, 2]
        hybrid = hybrid.view(shape[0], shape[1], -1, hybrid.shape[2], 2)
        
        return hybrid


class Hybrid_Inv(nn.Module):
    def __init__(
        self,
        n_fft: int = 4096,
        hop_length: int = 1024,
        center: bool = False,
        sample_rate: float = 44100.0,
        window: Optional[nn.Parameter] = None,
    ):
        super(Hybrid_Inv, self).__init__()
        if window is None:
            self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
        else:
            self.window = window

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        self.sample_rate = sample_rate
        
        # 计算STFT的频带数
        self.stft_bins = n_fft // 2 + 1
        
        # 设置CQT参数
        self.cqt_bins = 84  # 7个八度，每个八度12个音符
        self.cqt = CQT2010v2(
            sr=sample_rate,
            hop_length=hop_length,
            n_bins=self.cqt_bins,
            bins_per_octave=12,
            verbose=False
        )
        
        # 计算STFT和CQT的分界频率
        self.crossover_freq = 200  # 200Hz作为分界点
        self.crossover_bin = int(self.crossover_freq * n_fft / sample_rate)

    def forward(self, X: Tensor, length: Optional[int] = None) -> Tensor:
        """Hybrid inverse transform
        Args:
            X (Tensor): complex hybrid transform of
                shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
            length (int, optional): audio signal length to crop the signal
        Returns:
            Tensor: audio waveform of
                shape (nb_samples, nb_channels, nb_timesteps)
        """
        shape = X.size()
        X = X.reshape(-1, shape[-3], shape[-2], shape[-1])
        
        # 分离低频和高频部分
        cqt_part = X[:, :self.crossover_bin, :, :]
        stft_part = X[:, self.crossover_bin:, :, :]
        
        # 将CQT部分转换为原始CQT形状
        cqt_original = F.interpolate(
            cqt_part.permute(0, 3, 1, 2),  # [batch, 2, bins, time]
            size=(self.cqt_bins, cqt_part.shape[-2]),
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1)  # [batch, bins, time, 2]
        
        # 重建STFT
        stft_full = torch.zeros(
            X.shape[0], self.stft_bins, X.shape[2], 2,
            device=X.device
        )
        stft_full[:, self.crossover_bin:, :, :] = stft_part
        
        # 使用ISTFT重建信号
        y = torch.istft(
            torch.view_as_complex(stft_full),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=self.center,
            normalized=False,
            onesided=True,
            length=length,
        )
        
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
