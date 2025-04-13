import pytest
import torch
import numpy as np

from openunmix import model
from openunmix import umxhq
from openunmix import umx
from openunmix import umxl


@pytest.fixture(params=[100])
def nb_frames(request):
    return int(request.param)


@pytest.fixture(params=[1, 2])
def nb_channels(request):
    return request.param


@pytest.fixture(params=[2])
def nb_samples(request):
    return request.param


@pytest.fixture(params=[1024])
def nb_bins(request):
    return request.param


@pytest.fixture
def spectrogram(request, nb_samples, nb_channels, nb_bins, nb_frames):
    return torch.rand((nb_samples, nb_channels, nb_bins, nb_frames))


@pytest.fixture(params=[False])
def unidirectional(request):
    return request.param


@pytest.fixture(params=[32])
def hidden_size(request):
    return request.param


def test_shape(spectrogram, nb_bins, nb_channels, unidirectional, hidden_size):
    unmix = model.OpenUnmix(
        nb_bins=nb_bins,
        nb_channels=nb_channels,
        unidirectional=unidirectional,
        nb_layers=1,  # speed up training
        hidden_size=hidden_size,
    )
    unmix.eval()
    Y = unmix(spectrogram)
    assert spectrogram.shape == Y.shape


@pytest.mark.parametrize("model_fn", [umx, umxhq, umxl])
def test_model_loading(model_fn):
    X = torch.rand((1, 2, 4096))
    model = model_fn(niter=0, pretrained=True)
    Y = model(X)
    assert Y[:, 0, ...].shape == X.shape


def test_attention_mechanism():
    """测试注意力机制对低频部分的增强效果"""
    # 创建一个简单的测试模型
    unmix = model.OpenUnmix(
        nb_bins=1024,
        nb_channels=2,
        hidden_size=32,
        nb_layers=1,
        unidirectional=False
    )
    
    # 获取注意力权重
    # 使用更接近实际场景的输入维度
    x = torch.randn(100, 2, 64)  # (nb_frames, nb_samples, hidden_size)
    
    # 应用注意力机制
    attention_weights = unmix.freq_attention(x)
    x = x * attention_weights
    
    # 应用贝斯频率掩码
    x = x * unmix.bass_mask.view(1, 1, -1)
    
    # 取绝对值，因为增强效果应该体现在幅度上
    x = torch.abs(x)
    
    # 归一化到[0, 1]范围
    x = x / (torch.max(x) + 1e-6)
    
    # 检查低频部分的权重
    low_freq_indices = torch.arange(x.shape[-1]) < int(x.shape[-1] * 0.1)
    low_freq_weights = x[..., low_freq_indices]
    
    # 验证低频部分的权重是否显著大于其他部分
    high_freq_indices = ~low_freq_indices
    high_freq_weights = x[..., high_freq_indices]
    
    # 计算相对增强比例
    low_freq_mean = torch.mean(low_freq_weights)
    high_freq_mean = torch.mean(high_freq_weights)
    enhancement_ratio = low_freq_mean / high_freq_mean
    
    # 验证增强比例是否合理
    assert enhancement_ratio > 1.5, f"低频部分增强不足，增强比例为: {enhancement_ratio:.3f}"
    
    # 验证权重范围是否合理
    assert torch.min(x) >= 0.0, "权重不应为负值"
    assert torch.max(x) <= 1.0, "权重不应大于1.0"
    
    # 打印测试结果
    print(f"低频部分平均权重: {low_freq_mean.item():.3f}")
    print(f"高频部分平均权重: {high_freq_mean.item():.3f}")
    print(f"增强比例: {enhancement_ratio.item():.3f}")
    print(f"权重范围: [{torch.min(x).item():.3f}, {torch.max(x).item():.3f}]")
