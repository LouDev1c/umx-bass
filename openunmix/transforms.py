from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import librosa
import numpy as np

try:
    from asteroid_filterbanks.enc_dec import Encoder, Decoder
    from asteroid_filterbanks.transforms import to_torchaudio, from_torchaudio
    from asteroid_filterbanks import torch_stft_fb
except ImportError:
    pass


def make_filterbanks(
        n_fft: int = 4096,
        n_hop: int = 1024,
        center: bool = False,
        sample_rate: float = 44100.0,
        n_bins: int = 60,
        hop_length: int = 512,
        f_min: float = 30,
        bins_per_octave: int = 12,
        method: str = "stft"):
    window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)

    if method == "stft":
        encoder = TorchSTFT(n_fft=n_fft, n_hop=n_hop, window=window, center=center)
        decoder = TorchISTFT(n_fft=n_fft, n_hop=n_hop, window=window, center=center)
    elif method == "cqt":
        encoder = Librosa_CQT(
            n_bins=n_bins,
            hop_length=hop_length,
            f_min=f_min,
            bins_per_octave=bins_per_octave,
            sample_rate=sample_rate
        )
        decoder = Librosa_ICQT(
            n_bins=n_bins,
            hop_length=hop_length,
            f_min=f_min,
            bins_per_octave=bins_per_octave,
            sample_rate=sample_rate
        )
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


class TorchSTFT(nn.Module):
    """Multichannel Short-Time-Fourier Forward transform uses hard coded hann_window.
    Args:
        n_fft (int, optional): transform FFT size. Defaults to 4096.
        n_hop (int, optional): transform hop size. Defaults to 1024.
        center (bool, optional): If True, the signals first window is
            zero padded. Centering is required for a perfect
            reconstruction of the signal. However, during training
            of spectrogram models, it can safely turned off.
            Defaults to `true`
        window (nn.Parameter, optional): window function
    """

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
        # nb_samples, nb_channels, nb_timesteps = shape

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


class Librosa_CQT(nn.Module):
    """Constant-Q Transform using Librosa's implementation with PyTorch compatibility"""
    def __init__(
            self,
            n_bins: int = 84,                   # 涵盖音符数
            hop_length: int = 512,
            f_min: float = 32.7,                  # 贝斯E1对应的频率
            bins_per_octave: int = 12,          # 每个八度的音符数量
            sample_rate: float = 44100.0,
            window: str = 'hann',
            scale: bool = True,
            pad_mode: str = 'constant',
            norm: int = 1,
            filter_scale: float = 0.8,
            tuning: float = 0.0,
            sparsity: float = 0.01,
            res_type: str = 'soxr_hq',
            return_complex: bool = True
    ):
        super().__init__()
        self.n_bins = n_bins
        self.hop_length = hop_length
        self.f_min = f_min
        self.bins_per_octave = bins_per_octave
        self.sample_rate = sample_rate
        self.window = window
        self.scale = scale
        self.pad_mode = pad_mode
        self.norm = norm
        self.filter_scale = filter_scale
        self.tuning = tuning
        self.sparsity = sparsity
        self.res_type = res_type
        self.return_complex = return_complex

    def forward(self, x: Tensor) -> Tensor:
        """Forward CQT transform similar to STFT output format
        Args:
            x (Tensor): Input audio of shape (batch, channels, time)
        Returns:
            Tensor: CQT coefficients of shape (batch, channels, bins, frames, complex=2)
        """
        # Convert to numpy array for librosa processing
        device = x.device
        batch, channels, _ = x.shape

        # Reshape to process all channels at once (batch * channels, time)
        x_flat = x.reshape(-1, x.size(-1)).detach().cpu().numpy()

        # Pre-allocate output array
        cqt_real_imag = np.zeros((len(x_flat), self.n_bins, (x.size(-1) // self.hop_length) + 1, 2),
                                 dtype=np.float32)

        # Process all audios in parallel
        for i, audio in enumerate(x_flat):
            cqt = librosa.cqt(
                audio,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                fmin=self.f_min,
                n_bins=self.n_bins,
                bins_per_octave=self.bins_per_octave,
                window=self.window,
                scale=self.scale,
                pad_mode=self.pad_mode,
                norm=self.norm,
                filter_scale=self.filter_scale,
                tuning=self.tuning,
                sparsity=self.sparsity,
                res_type=self.res_type
            )
            if self.return_complex:
                cqt_real_imag[i, ..., 0] = cqt.real
                cqt_real_imag[i, ..., 1] = cqt.imag
            else:
                mag, phase = librosa.magphase(cqt)
                cqt_real_imag[i, ..., 0] = mag
                cqt_real_imag[i, ..., 1] = phase

        # Reshape back to (batch, channels, bins, frames, 2)
        output = torch.from_numpy(cqt_real_imag).to(device)
        return output.view(batch, channels, self.n_bins, -1, 2)


class Librosa_ICQT(nn.Module):
    """Inverse Constant-Q Transform using Librosa's implementation with PyTorch compatibility"""

    def __init__(
            self,
            n_bins: int = 96,
            hop_length: int = 256,
            f_min: float = 27.0,
            bins_per_octave: int = 12,
            sample_rate: float = 44100.0,
            window: str = 'hann',
            scale: bool = True,
            norm: int = 1,
            filter_scale: int = 1,
            tuning: float = 0.0,
            sparsity: float = 0.01,
            res_type: str = 'soxr_hq',
            input_type: str = 'complex'
    ):
        super().__init__()
        self.n_bins = n_bins
        self.hop_length = hop_length
        self.f_min = f_min
        self.bins_per_octave = bins_per_octave
        self.sample_rate = sample_rate
        self.window = window
        self.scale = scale
        self.norm = norm
        self.filter_scale = filter_scale
        self.tuning = tuning
        self.sparsity = sparsity
        self.res_type = res_type
        self.input_type = input_type

    def forward(self, X: Tensor, length: Optional[int] = None) -> Tensor:
        """Inverse CQT transform similar to ISTFT

        Args:
            X (Tensor): CQT coefficients of shape (batch, channels, bins, frames, complex=2)
            length (int, optional): Target output length in samples

        Returns:
            Tensor: Reconstructed audio of shape (batch, channels, time)
        """
        # Convert to numpy array for librosa processing
        device = X.device
        batch, channels, _, _, _ = X.size()

        # Reshape to (batch * channels, bins, frames, 2)
        X_flat = X.reshape(-1, *X.shape[2:]).cpu().numpy()
        audio_list = []

        for i, x in enumerate(X_flat):
            if self.input_type == 'complex':
                cqt = x[..., 0] + 1j * x[..., 1]  # Reconstruct complex CQT
            else:  # 'magphase'
                cqt = x[..., 0] * np.exp(1j * x[..., 1])  # Mag * e^(j*phase)

            # Inverse CQT
            audio = librosa.icqt(
                cqt,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                fmin=self.f_min,
                bins_per_octave=self.bins_per_octave,
                window=self.window,
                scale=self.scale,
                norm=self.norm,
                filter_scale=self.filter_scale,
                tuning=self.tuning,
                sparsity=self.sparsity,
                res_type=self.res_type,
                length=length
            )
            audio_list.append(audio)

        # Stack and reshape
        audio_np = np.stack(audio_list).reshape(batch, channels, -1)
        return torch.from_numpy(audio_np).to(device)


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
    Extension of `torchaudio.functional.complex_norm` with mono
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
        # downmix in the mag domain to preserve energy
        if self.mono:
            spec = torch.mean(spec, 1, keepdim=True)
        return spec
