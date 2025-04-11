from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from nnAudio.features import CQT

try:
    from asteroid_filterbanks.enc_dec import Encoder, Decoder
    from asteroid_filterbanks.transforms import to_torchaudio, from_torchaudio
    from asteroid_filterbanks import torch_stft_fb
except ImportError:
    pass


def make_filterbanks(
        n_fft=4096,
        n_hop=1024,
        center=False,
        sr=44100.0,
        method="stft"):
    window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)

    if method == "stft":
        encoder = TorchSTFT(n_fft=n_fft, n_hop=n_hop, window=window, center=center)
        decoder = TorchISTFT(n_fft=n_fft, n_hop=n_hop, window=window, center=center)
    elif method == "cqt":
        encoder = nnAudioCQT(
            sample_rate=int(sr),  # 确保为int类型
            fmin=32.7,  # 贝斯最低频(C1音)
            n_bins=84,  # 7个八度(12 * 7=84)
            bins_per_octave=12,  # 每八度12个半音
            n_hop=n_hop
        )
        decoder = nnAudioICQT(
            n_fft=n_fft,  # 使用传入的n_fft参数
            n_hop=n_hop,
            n_iter=50  # Griffin-Lim迭代次数
        )
    elif method == "asteroid":
        fb = torch_stft_fb.TorchSTFTFB.from_torch_args(
            n_fft=n_fft,
            hop_length=n_hop,
            win_length=n_fft,
            window=window,
            center=center,
            sample_rate=sr,
        )
        encoder = AsteroidSTFT(fb)
        decoder = AsteroidISTFT(fb)
    else:
        raise NotImplementedError
    return encoder, decoder


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


class nnAudioCQT(nn.Module):
    def __init__(
            self,
            sample_rate: int = 44100,
            fmin: float = 32.7,
            n_bins: int = 84,
            bins_per_octave: int = 12,
            n_hop: int = 1024,
    ):
        super().__init__()
        self.cqt = CQT(
            sr=sample_rate,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            hop_length=n_hop,
            output_format="Complex",
            pad_mode='reflect',
            trainable=False,
            window='hann'
        )
        # 添加手动复数转换标志
        self.force_complex = True
        self.out_complex = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.reshape(-1, shape[-1])

        # 计算CQT
        cqt_out = self.cqt(x)

        # 处理复数输出
        if cqt_out.is_complex():
            cqt_complex = cqt_out
        else:
            if self.force_complex and cqt_out.shape[-1] == 2:
                cqt_complex = torch.view_as_complex(cqt_out.contiguous())
            else:
                raise RuntimeError(
                    f"CQT output format not supported. Shape: {cqt_out.shape}, is_complex: {cqt_out.is_complex()}")
        # 存储复数输出
        self.out_complex = cqt_complex
        magnitude = torch.view_as_real(cqt_complex)

        # 转换为实部/虚部表示
        return magnitude.reshape(shape[0], shape[1], -1, cqt_complex.shape[-1], 2)

    def get_complex(self):
        if self.out_complex is None:
            raise RuntimeError("No complex output available. Call forward() first.")
        return self.out_complex

class nnAudioICQT(nn.Module):
    def __init__(
            self,
            n_fft: int = 4096,
            n_hop: int = 1024,
            n_iter: int = 100,  # 增加迭代次数
            momentum: float = 0.99,  # 添加动量项
    ):
        super().__init__()
        self._prev_y = None
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.n_iter = n_iter
        self.momentum = momentum
        self.window = torch.hann_window(n_fft)

        # 存储对应的CQT编码器
        self._cqt_encoder = None

    def set_cqt_encoder(self, encoder):
        self._cqt_encoder = encoder

    def forward(self, X: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        if self._cqt_encoder is None:
            raise RuntimeError("CQT encoder not set. Call set_cqt_encoder() first.")
        shape = X.size()
        X = X.reshape(-1, *shape[-2:])
        # 获取原始复数输出
        cqt_complex = self._cqt_encoder.get_complex()
        if cqt_complex is None:
            raise RuntimeError("No complex output available from CQT encoder")

        # 提取相位信息
        phase = torch.angle(cqt_complex)

        # 将CQT幅度谱转换为STFT幅度谱
        stft_magnitude = self._cqt_to_stft_magnitude(X)

        # 确保相位和幅度谱的维度匹配
        if phase.shape[1] != stft_magnitude.shape[1]:
            # 使用插值调整相位谱的维度
            phase = torch.nn.functional.interpolate(
                phase.unsqueeze(1),
                size=(stft_magnitude.shape[1], phase.shape[2]),
                mode="linear",
                align_corners=True
            ).squeeze(1)

        # 使用原始相位重建复数谱
        stft_complex = stft_magnitude * torch.exp(1j * phase)

        # 使用ISTFT重构时域信号
        y = torch.istft(
            stft_complex,
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            win_length=self.n_fft,
            window=self.window.to(X.device),
            length=length
        )

        # 应用动量项
        if hasattr(self, '_prev_y'):
            y = self.momentum * self._prev_y + (1 - self.momentum) * y
        self._prev_y = y

        return y.reshape(shape[:-2] + y.shape[-1:])

    def _cqt_to_stft_magnitude(self, cqt_mag: torch.Tensor) -> torch.Tensor:
        stft_bins = self.n_fft // 2 + 1
        # 使用更好的插值方法
        return torch.nn.functional.interpolate(
            cqt_mag.unsqueeze(1),
            size=(stft_bins, cqt_mag.shape[-1]),
            mode="bicubic",  # 使用双三次插值
            align_corners=True
        ).squeeze(1)


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
