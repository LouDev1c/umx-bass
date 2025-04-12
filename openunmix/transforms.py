from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from nnAudio.features import CQT
import torch.nn.functional as F

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
            pad_mode='constant',
            trainable=False
        )
        # 添加手动复数转换标志
        self.force_complex = True

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
                # 如果输出是分开的实部和虚部
                cqt_complex = torch.view_as_complex(cqt_out.contiguous())
            else:
                raise RuntimeError(
                    f"CQT output format not supported. Shape: {cqt_out.shape}, is_complex: {cqt_out.is_complex()}")

        # 转换为实部/虚部表示
        return torch.view_as_real(cqt_complex).reshape(shape[0], shape[1], -1, cqt_complex.shape[-1], 2)


class nnAudioICQT(nn.Module):
    def __init__(
            self,
            sample_rate: int = 44100,
            n_fft: int = 4096,
            n_hop: int = 1024,
            n_iter: int = 50,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.n_iter = n_iter
        self.window = torch.hann_window(n_fft)

    def forward(self, X: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        shape = X.size()
        X = X.reshape(-1, *shape[-3:])
        target_magnitude = torch.view_as_complex(X).abs()
        target_magnitude_stft = self._cqt_to_stft_magnitude(target_magnitude)

        y = torch.randn(
            X.shape[0],
            length if length else shape[-2] * self.n_hop,
            device=X.device,
            dtype=torch.float32
        )

        for _ in range(self.n_iter):
            stft_complex = torch.stft(
                y,
                n_fft=self.n_fft,
                hop_length=self.n_hop,
                win_length=self.n_fft,
                window=self.window.to(y.device),
                return_complex=True
            )
            phase = torch.angle(stft_complex)
            new_spec = target_magnitude_stft * torch.exp(1j * phase)
            y = torch.istft(
                new_spec,
                n_fft=self.n_fft,
                hop_length=self.n_hop,
                win_length=self.n_fft,
                window=self.window.to(y.device),
                length=length
            )

        return y.reshape(shape[:-3] + y.shape[-1:])

    def _cqt_to_stft_magnitude(self, cqt_mag: torch.Tensor) -> torch.Tensor:
        stft_bins = self.n_fft // 2 + 1
        return torch.nn.functional.interpolate(
            cqt_mag.unsqueeze(1),
            size=(stft_bins, cqt_mag.shape[-1]),
            mode="bicubic"
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
