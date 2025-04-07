import pytest
import torch

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


def test_attention_mechanism(spectrogram, nb_bins, nb_channels, hidden_size):
    """测试频率感知注意力机制是否正常工作"""
    unmix = model.OpenUnmix(
        nb_bins=nb_bins,
        nb_channels=nb_channels,
        hidden_size=hidden_size,
        nb_layers=1,
    )
    unmix.eval()

    # 前向传播
    Y = unmix(spectrogram)

    # 检查输出形状
    assert spectrogram.shape == Y.shape

    # 检查注意力层的参数
    attention_params = list(unmix.freq_attention.parameters())
    assert len(attention_params) == 4  # 两个线性层的weight和bias

    # 检查注意力层的输入输出维度
    x = spectrogram.permute(3, 0, 1, 2)
    nb_frames, nb_samples, nb_channels, nb_bins = x.shape
    x = x[..., :unmix.nb_bins]
    x = x + unmix.input_mean
    x = x * unmix.input_scale
    x = unmix.fc1(x.reshape(-1, nb_channels * unmix.nb_bins))
    x = unmix.bn1(x)
    x = x.reshape(nb_frames, nb_samples, unmix.hidden_size)
    x = torch.tanh(x)
    lstm_out = unmix.lstm(x)
    x = torch.cat([x, lstm_out[0]], -1)

    # 应用注意力机制
    attention_weights = unmix.freq_attention(x)
    assert attention_weights.shape == x.shape
    assert torch.all(attention_weights >= 0) and torch.all(attention_weights <= 1)


def test_attention_mechanism(spectrogram, nb_bins, nb_channels, hidden_size):
    """测试频率感知注意力机制是否正常工作"""
    unmix = model.OpenUnmix(
        nb_bins=nb_bins,
        nb_channels=nb_channels,
        hidden_size=hidden_size,
        nb_layers=1,
    )
    unmix.eval()
    
    # 前向传播
    Y = unmix(spectrogram)
    
    # 检查输出形状
    assert spectrogram.shape == Y.shape
    
    # 检查注意力层的参数
    attention_params = list(unmix.freq_attention.parameters())
    assert len(attention_params) == 4  # 两个线性层的weight和bias
    
    # 检查注意力层的输入输出维度
    x = spectrogram.permute(3, 0, 1, 2)
    nb_frames, nb_samples, nb_channels, nb_bins = x.shape
    x = x[..., :unmix.nb_bins]
    x = x + unmix.input_mean
    x = x * unmix.input_scale
    x = unmix.fc1(x.reshape(-1, nb_channels * unmix.nb_bins))
    x = unmix.bn1(x)
    x = x.reshape(nb_frames, nb_samples, unmix.hidden_size)
    x = torch.tanh(x)
    lstm_out = unmix.lstm(x)
    x = torch.cat([x, lstm_out[0]], -1)
    
    # 应用注意力机制
    attention_weights = unmix.freq_attention(x)
    assert attention_weights.shape == x.shape
    assert torch.all(attention_weights >= 0) and torch.all(attention_weights <= 1)


@pytest.mark.parametrize("model_fn", [umx, umxhq, umxl])
def test_model_loading(model_fn):
    X = torch.rand((1, 2, 4096))
    model = model_fn(niter=0, pretrained=True)
    Y = model(X)
    assert Y[:, 0, ...].shape == X.shape
