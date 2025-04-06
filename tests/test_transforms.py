import pytest
import numpy as np
import torch
from openunmix import transforms

@pytest.fixture(params=[4096])
def nb_timesteps(request):
    return int(request.param)

@pytest.fixture(params=[2])
def nb_channels(request):
    return request.param

@pytest.fixture(params=[2])
def nb_samples(request):
    return request.param

@pytest.fixture(params=[2048])
def nfft(request):
    return int(request.param)

@pytest.fixture(params=[2])
def hop(request, nfft):
    return nfft // request.param

# 新增CQT专用参数
@pytest.fixture(params=[96])  # 默认84个频点（覆盖7个八度）
def n_bins(request):
    return request.param

@pytest.fixture(params=[256])  # CQT专用hop_length
def cqt_hop_length(request):
    return request.param

@pytest.fixture(params=[27.0])  # 最低频率（C2音符）
def f_min(request):
    return request.param

@pytest.fixture(params=[12])  # 每八度的bin数
def bins_per_octave(request):
    return request.param

@pytest.fixture(params=["stft", "cqt", "asteroid"])
def method(request):
    return request.param

@pytest.fixture
def audio(nb_samples, nb_channels, nb_timesteps):
    return torch.rand((nb_samples, nb_channels, nb_timesteps))


def test_trans(audio, nfft, hop, method, n_bins, cqt_hop_length, f_min, bins_per_octave):
    # 根据方法选择参数
    if method == "stft":
        encoder, decoder = transforms.make_filterbanks(
            n_fft=nfft,
            n_hop=hop,
            center=True,
            method=method
        )
    elif method == "asteroid":
        encoder, decoder = transforms.make_filterbanks(
            n_fft=nfft,
            n_hop=hop,
            center=True,
            method=method
        )
    elif method == "cqt":
        encoder, decoder = transforms.make_filterbanks(
            n_fft=nfft,
            n_hop=hop,
            center=True,
            n_bins=n_bins,
            hop_length=cqt_hop_length,  # 使用CQT专用hop
            f_min=f_min,
            bins_per_octave=bins_per_octave,
            method=method
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    X = encoder(audio)
    out = decoder(X.detach(), length=audio.shape[-1])

    tolerance = 1e-2 if method == "cqt" else 1e-6
    error = np.sqrt(np.mean((audio.detach().numpy() - out.detach().numpy()) ** 2))
    assert error < tolerance, f"{method}重建误差{error}超过阈值{tolerance}"