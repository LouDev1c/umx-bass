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


class GriffinLim(nn.Module):
    """Griffin-Lim 算法实现"""
    def __init__(self, n_iter=32):
        super(GriffinLim, self).__init__()
        self.n_iter = n_iter

    def forward(self, magnitude, stft_fn, istft_fn, length=None):
        """Griffin-Lim 算法
        Args:
            magnitude: 幅度谱
            stft_fn: STFT 函数
            istft_fn: ISTFT 函数
            length: 输出信号长度
        Returns:
            重建的时域信号
        """
        # 初始化随机相位
        angles = torch.randn_like(magnitude) * 2 * torch.pi
        complex_spec = magnitude * torch.exp(1j * angles)
        
        # 迭代优化
        for _ in range(self.n_iter):
            # 重建时域信号
            signal = istft_fn(complex_spec, length=length)
            # 重新计算STFT
            complex_spec = stft_fn(signal)
            # 保持幅度不变，更新相位
            complex_spec = magnitude * torch.exp(1j * torch.angle(complex_spec))
        
        # 最后一次重建
        signal = istft_fn(complex_spec, length=length)
        return signal


class Hybrid_Inv(nn.Module):
    def __init__(
        self,
        n_fft: int = 4096,
        hop_length: int = 1024,
        center: bool = False,
        sample_rate: float = 44100.0,
        window: Optional[nn.Parameter] = None,
        n_iter: int = 32,  # Griffin-Lim 迭代次数
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
        
        # 计算过渡带的宽度（以STFT的bin为单位）
        self.transition_width = 4  # 可以根据需要调整
        
        # 创建过渡带权重
        self.register_buffer('transition_weights', self._create_transition_weights())
        
        # 初始化Griffin-Lim
        self.griffin_lim = GriffinLim(n_iter=n_iter)

    def _create_transition_weights(self):
        """创建过渡带的权重函数"""
        weights = torch.ones(self.stft_bins, device=self.window.device)
        transition_start = self.crossover_bin - self.transition_width
        transition_end = self.crossover_bin + self.transition_width
        
        # 创建平滑的过渡函数
        transition = torch.linspace(0, 1, 2 * self.transition_width + 1, device=self.window.device)
        weights[transition_start:transition_end + 1] = transition
        
        return weights

    def _stft_fn(self, x):
        """STFT 函数包装器"""
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=self.center,
            normalized=False,
            onesided=True,
            pad_mode="reflect",
            return_complex=True,
        )

    def _istft_fn(self, X, length=None):
        """ISTFT 函数包装器"""
        return torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=self.center,
            normalized=False,
            onesided=True,
            length=length,
        )

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
        
        # 使用CQT重建低频部分
        cqt_complex = torch.view_as_complex(cqt_original)
        cqt_mag = torch.abs(cqt_complex)
        
        # 使用Griffin-Lim重建低频信号
        cqt_signal = self.griffin_lim(
            cqt_mag,
            self.cqt.forward,
            self.cqt.inverse,
            length=length
        )
        
        # 重建STFT
        stft_full = torch.zeros(
            X.shape[0], self.stft_bins, X.shape[2], 2,
            device=X.device
        )
        stft_full[:, self.crossover_bin:, :, :] = stft_part
        
        # 使用Griffin-Lim重建高频信号
        stft_mag = torch.abs(torch.view_as_complex(stft_full))
        stft_signal = self.griffin_lim(
            stft_mag,
            self._stft_fn,
            self._istft_fn,
            length=length
        )
        
        # 计算能量归一化因子
        cqt_energy = torch.mean(torch.abs(cqt_signal))
        stft_energy = torch.mean(torch.abs(stft_signal))
        energy_ratio = torch.sqrt(cqt_energy / stft_energy)
        
        # 应用能量归一化
        stft_signal = stft_signal * energy_ratio
        
        # 使用过渡带权重合并信号
        transition_weights = self.transition_weights.view(1, 1, -1, 1)
        stft_weights = transition_weights
        cqt_weights = 1 - transition_weights
        
        # 在频域应用权重
        stft_full = stft_full * stft_weights
        cqt_full = torch.zeros_like(stft_full)
        cqt_full[:, :self.crossover_bin, :, :] = cqt_part * cqt_weights[:self.crossover_bin]
        
        # 合并频域表示
        combined_stft = stft_full + cqt_full
        
        # 使用Griffin-Lim重建最终信号
        combined_mag = torch.abs(torch.view_as_complex(combined_stft))
        y = self.griffin_lim(
            combined_mag,
            self._stft_fn,
            self._istft_fn,
            length=length
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
