from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm

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
    sample_rate: int = 44100,
    n_bins: int = 84,
    hop_length: int = 512,
    f_min: float = 32.7,
    bins_per_octave: int = 12,
    method: str = "stft",
    window_type: str = "hann",
    n_iter: int = 32
):
    window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)

    if method == "stft":

        encoder = TorchSTFT(n_fft=n_fft, n_hop=n_hop, window=window, center=center)
        decoder = TorchISTFT(n_fft=n_fft, n_hop=n_hop, window=window, center=center)
    elif method == "cqt":
        encoder = CQT(
            sr=sample_rate,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            fmin=f_min,
            n_hop=hop_length,  # 使用专门的hop_length参数
            center=center,
            window=window
        )
        decoder = ICQT(
            sr=sample_rate,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            fmin=f_min,
            n_hop=hop_length,
            center=center,
            window=window,
            n_iter=n_iter
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


class CQT(nn.Module):
    def __init__(
            self,
            sr: int,
            n_bins=84,
            bins_per_octave=12,
            fmin: float = 32.7,
            n_hop=512,
            center: bool = False,
            window: Optional[nn.Parameter] = None
    ):
        super(CQT, self).__init__()

        self.sr = sr
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.fmin = fmin
        self.n_hop = n_hop
        self.center = center

        # Calculate Q value
        self.Q = 1.0 / (2.0 ** (1.0 / bins_per_octave) - 1.0)

        # Compute the frequencies for each bin
        self.frequencies = fmin * (2 ** (torch.arange(n_bins, dtype=torch.float32) / bins_per_octave))

        # Compute the window lengths for each bin
        self.window_lengths = torch.round(self.Q * sr / self.frequencies).long()

        # Compute maximum window length for padding
        self.max_window_length = self.window_lengths.max().item()

        # Create window function
        if window is None:
            # We'll create windows on the fly in forward pass since they have different lengths
            self.window = None
        else:
            self.window = window

        # Pre-compute complex exponential terms
        self.register_buffer('complex_exponentials', self._create_complex_exponentials())

    def _create_complex_exponentials(self):
        """Create complex exponential terms for each frequency bin."""
        t = torch.arange(self.max_window_length, dtype=torch.float32)
        t = t.unsqueeze(0)  # [1, max_window_length]

        # Compute angular frequencies
        omega = 2 * math.pi * self.frequencies.unsqueeze(1) / self.sr  # [n_bins, 1]

        # Compute complex exponentials
        complex_exp = torch.exp(-1j * omega * t)  # [n_bins, max_window_length]

        return complex_exp

    def _create_windows(self, device):
        """Create windows for each frequency bin."""
        windows = []
        for length in self.window_lengths:
            if self.window is None:
                # Use Hann window
                win = torch.hann_window(length, device=device)
            else:
                # Use provided window (truncated to appropriate length)
                win = self.window[:length]
            windows.append(win)
        return windows

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """CQT forward path

        Args:
            x (Tensor): audio waveform of shape (nb_samples, nb_channels, nb_timesteps)

        Returns:
            CQT (Tensor): complex CQT of shape
                (nb_samples, nb_channels, n_bins, nb_frames, complex=2)
                last axis is stacked real and imaginary
        """
        shape = x.size()
        nb_samples, nb_channels, nb_timesteps = shape

        # Pack batch and channels
        x = x.view(-1, nb_timesteps)

        if self.center:
            # Pad the signal on both sides
            x = F.pad(x, (self.max_window_length // 2, self.max_window_length // 2), mode='reflect')

        # Get device
        device = x.device

        # Create windows for each frequency bin
        windows = self._create_windows(device)

        # Compute number of frames
        n_frames = 1 + (x.shape[-1] - self.max_window_length) // self.n_hop

        # Initialize output tensor
        cqt = torch.zeros((x.shape[0], self.n_bins, n_frames),
                          dtype=torch.complex64, device=device)

        # Compute CQT for each frequency bin
        for k in range(self.n_bins):
            window_length = self.window_lengths[k].item()
            window = windows[k]

            # Extract frames
            frames = F.unfold(
                x.unsqueeze(1).unsqueeze(-1),
                kernel_size=(1, window_length),
                stride=(1, self.n_hop)
            )  # [batch, window_length, n_frames]

            frames = frames.transpose(1, 2)  # [batch, n_frames, window_length]

            # Apply window
            frames = frames * window.unsqueeze(0)

            # Compute dot product with complex exponential
            complex_exp = self.complex_exponentials[k, :window_length]
            cqt[:, k, :] = torch.sum(frames * complex_exp, dim=-1)

        # Convert to real/imaginary representation
        cqt = torch.view_as_real(cqt)

        # Unpack batch and channels
        cqt = cqt.view(nb_samples, nb_channels, self.n_bins, n_frames, 2)

        return cqt


class ICQT(nn.Module):
    """Multichannel Inverse Constant-Q Transform using PyTorch operations.

        Args:
            sr (int): Sample rate of the input audio
            n_bins (int, optional): Number of frequency bins. Defaults to 84.
            bins_per_octave (int, optional): Number of bins per octave. Defaults to 12.
            fmin (float, optional): Minimum frequency. Defaults to 32.7 (C1).
            n_hop (int, optional): Hop length between frames. Defaults to 1024.
            center (bool, optional): If True, the signals first window is zero padded.
                Defaults to False.
            window (nn.Parameter, optional): Window function. Defaults to Hann window.
            n_iter (int, optional): Number of Griffin-Lim iterations for phase reconstruction.
                Defaults to 32.
        """

    def __init__(
        self,
        sr: int,
        n_bins: int = 84,
        bins_per_octave: int = 12,
        fmin: float = 32.7,
        n_hop: int = 1024,
        center: bool = False,
        window: Optional[nn.Parameter] = None,
        n_iter: int = 32,
    ):
        super(ICQT, self).__init__()

        # Store parameters
        self.sr = sr
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.fmin = fmin
        self.n_hop = n_hop
        self.center = center
        self.n_iter = n_iter

        # Compute Q factor
        self.Q = 1.0 / (2 ** (1.0 / bins_per_octave) - 1)

        # Compute the frequencies for each bin
        self.frequencies = fmin * (2 ** (torch.arange(n_bins, dtype=torch.float32) / bins_per_octave))

        # Compute the window lengths for each bin
        self.window_lengths = torch.round(self.Q * sr / self.frequencies).long()

        # Compute maximum window length for padding
        self.max_window_length = self.window_lengths.max().item()

        # Create window function
        if window is None:
            # We'll create windows on the fly in forward pass since they have different lengths
            self.window = None
        else:
            self.window = window

        # Pre-compute complex exponential terms for reconstruction
        self.register_buffer('complex_exponentials', self._create_complex_exponentials())

        # Create a forward CQT for Griffin-Lim iterations
        self.cqt = CQT(
            sr=sr,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            fmin=fmin,
            n_hop=n_hop,
            center=center,
            window=window
        )

    def _create_complex_exponentials(self):
        """Create complex exponential terms for each frequency bin."""
        t = torch.arange(self.max_window_length, dtype=torch.float32)
        t = t.unsqueeze(0)  # [1, max_window_length]

        # Compute angular frequencies
        omega = 2 * math.pi * self.frequencies.unsqueeze(1) / self.sr  # [n_bins, 1]

        # Compute complex exponentials
        complex_exp = torch.exp(1j * omega * t)  # [n_bins, max_window_length] (note positive sign)

        return complex_exp

    def _create_windows(self, device):
        """Create windows for each frequency bin."""
        windows = []
        for length in self.window_lengths:
            if self.window is None:
                # Use Hann window
                win = torch.hann_window(length, device=device)
            else:
                # Use provided window (truncated to appropriate length)
                win = self.window[:length]
            windows.append(win)
        return windows

    def _griffin_lim(self, mag_cqt: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """Griffin-Lim algorithm for phase reconstruction.

        Args:
            mag_cqt (Tensor): Magnitude CQT of shape (batch, n_bins, n_frames)
            length (int, optional): Target output length

        Returns:
            Tensor: Reconstructed waveform
        """
        # Initialize random phase
        angles = 2 * math.pi * torch.rand_like(mag_cqt)
        complex_spec = mag_cqt * torch.exp(1j * angles)

        # Convert to time-frequency representation expected by ICQT
        cqt_input = torch.view_as_real(complex_spec.unsqueeze(-1).expand(*complex_spec.shape, 2))

        for _ in range(self.n_iter):
            # Inverse transform
            waveform = self.forward(cqt_input, length=length)

            # Forward transform
            new_cqt = self.cqt(waveform)
            new_angles = torch.angle(torch.view_as_complex(new_cqt))

            # Update phase while keeping magnitude
            complex_spec = mag_cqt * torch.exp(1j * new_angles)
            cqt_input = torch.view_as_real(complex_spec.unsqueeze(-1).expand(*complex_spec.shape, 2))

        return waveform

    def forward(self, X: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """ICQT forward path

        Args:
            X (Tensor): complex CQT of shape
                (nb_samples, nb_channels, n_bins, nb_frames, complex=2)
            length (int, optional): Target output length. Defaults to None.

        Returns:
            Tensor: audio waveform of shape (nb_samples, nb_channels, nb_timesteps)
        """
        shape = X.size()
        nb_samples, nb_channels, n_bins, n_frames, _ = shape

        # Check if input is magnitude-only (all imaginary parts zero)
        is_magnitude = torch.allclose(X[..., 1], torch.zeros_like(X[..., 1]))

        if is_magnitude:
            # Use Griffin-Lim algorithm for phase reconstruction
            mag_cqt = X[..., 0]
            mag_cqt = mag_cqt.reshape(-1, n_bins, n_frames)
            waveform = self._griffin_lim(mag_cqt, length=length)
            return waveform.reshape(nb_samples, nb_channels, -1)

        # Pack batch and channels
        X = X.reshape(-1, n_bins, n_frames, 2)
        complex_cqt = torch.view_as_complex(X)

        # Get device
        device = X.device

        # Create windows for each frequency bin
        windows = self._create_windows(device)

        # Compute output length if not provided
        if length is None:
            length = (n_frames - 1) * self.n_hop + self.max_window_length

        # Initialize output signal
        output = torch.zeros((X.shape[0], length), device=device)
        norm = torch.zeros_like(output)

        # Reconstruct signal for each frequency bin
        for k in range(n_bins):
            window_length = self.window_lengths[k].item()
            window = windows[k]

            # Compute time positions of frames
            time_steps = torch.arange(n_frames, device=device) * self.n_hop

            # Compute the complex sinusoids for this bin
            sinusoid = self.complex_exponentials[k, :window_length]

            # Compute the frames
            frames = complex_cqt[:, k].unsqueeze(-1) * sinusoid.unsqueeze(0) * window.unsqueeze(0)

            # Overlap-add the frames
            for t in range(n_frames):
                start = time_steps[t]
                end = start + window_length
                if end > length:
                    frames = frames[:, :length - start]
                    window_length = frames.shape[-1]
                    end = length

                output[:, start:end] += frames[:, t]
                norm[:, start:end] += window[:window_length].pow(2)

        # Normalize by window power
        norm = torch.where(norm > 1e-10, norm, torch.ones_like(norm))
        output = output / norm

        if self.center:
            # Remove padding if center was True
            pad = self.max_window_length // 2
            output = output[:, pad:-pad] if length is None else output[:, pad:pad+length]

        # Unpack batch and channels
        output = output.reshape(nb_samples, nb_channels, -1)

        return output

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
