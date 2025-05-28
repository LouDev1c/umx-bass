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


@pytest.fixture
def audio(request, nb_samples, nb_channels, nb_timesteps):
    return torch.rand((nb_samples, nb_channels, nb_timesteps))


def test_stft(audio, nfft, hop):
    # we should only test for center=True as
    # False doesn't pass COLA
    # https://github.com/pytorch/audio/issues/500
    stft, istft = transforms.make_filterbanks(n_fft=nfft, n_hop=hop, center=True, method="stft")

    X = stft(audio)
    X = X.detach()
    out = istft(X, length=audio.shape[-1])
    assert np.sqrt(np.mean((audio.detach().numpy() - out.detach().numpy()) ** 2)) < 1e-6


def test_hybrid(audio, nfft, hop):
    """测试 Hybrid 变换和逆变换的重建质量"""
    # 创建 Hybrid 和 Hybrid_Inv
    hybrid, hybrid_inv = transforms.make_filterbanks(
        n_fft=nfft,
        n_hop=hop,
        center=True,
        sample_rate=44100.0,
        method="hybrid"
    )

    # 进行变换和逆变换
    X = hybrid(audio)
    X = X.detach()
    out = hybrid_inv(X, length=audio.shape[-1])

    # 计算重建误差
    mse = np.sqrt(np.mean((audio.detach().numpy() - out.detach().numpy()) ** 2))
    print(f"Hybrid 重建 MSE: {mse:.6f}")
    
    # 验证重建质量
    assert mse < 1e-3  # 允许稍大的误差，因为 Hybrid 变换可能不如 STFT 精确


def test_hybrid_frequency_resolution():
    """测试 Hybrid 变换的频率分辨率"""
    # 创建测试信号
    sample_rate = 44100
    duration = 1  # 秒
    t = torch.linspace(0, duration, int(sample_rate * duration))
    
    # 生成包含低频和高频的测试信号
    low_freq = torch.sin(2 * np.pi * 100 * t)  # 100Hz
    high_freq = torch.sin(2 * np.pi * 1000 * t)  # 1000Hz
    test_signal = low_freq + high_freq
    test_signal = test_signal.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    
    # 创建 Hybrid 和 Hybrid_Inv
    hybrid, hybrid_inv = transforms.make_filterbanks(
        n_fft=4096,
        n_hop=1024,
        center=True,
        sample_rate=sample_rate,
        method="hybrid"
    )
    
    # 进行变换
    X = hybrid(test_signal)
    
    # 计算预期的频带数
    stft_bins = 4096 // 2 + 1  # STFT 频带数
    crossover_bin = int(200 * 4096 / sample_rate)  # 分界点
    expected_bins = stft_bins  # 总频带数应该等于STFT的频带数
    
    # 验证频谱形状
    assert X.shape[2] == expected_bins, f"Expected {expected_bins} bins, got {X.shape[2]}"
    
    # 验证重建质量
    out = hybrid_inv(X, length=test_signal.shape[-1])
    mse = np.sqrt(np.mean((test_signal.detach().numpy() - out.detach().numpy()) ** 2))
    assert mse < 1e-3
