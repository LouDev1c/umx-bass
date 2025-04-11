import pytest
import torch
from openunmix import transforms
from openunmix.transforms import nnAudioICQT, nnAudioCQT


@pytest.fixture(params=[32768])
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

@pytest.fixture(params=["stft", "cqt", "asteroid"])
def method(request):
    return request.param

@pytest.fixture
def audio(request, nb_samples, nb_channels, nb_timesteps):
    # 根据方法类型决定音频长度
    if method == "cqt":
        # CQT需要更长的音频才能有足够的低频分辨率
        return torch.rand((nb_samples, nb_channels, nb_timesteps * 2))
    else:
        return torch.rand((nb_samples, nb_channels, nb_timesteps))

@pytest.mark.parametrize(
    "nfft,hop,method",
    [(2048, 2, "stft"), (2048, 2, "cqt"), (2048, 2, "asteroid")]
)
def test_encoder_and_decoder(audio, nfft, hop, method):
    if method == "cqt":
        # 使用更适合CQT的参数
        encoder, decoder = transforms.make_filterbanks(
            n_fft=nfft,
            n_hop=hop,
            center=True,
            method=method,
            sr=44100.0
        )
        if isinstance(decoder, nnAudioICQT):
            decoder.set_cqt_encoder(encoder)
        tolerance = 0.6
    else:
        encoder, decoder = transforms.make_filterbanks(
            n_fft=nfft,
            n_hop=hop,
            center=True,
            method=method
        )
        tolerance = 1e-6

    X = encoder(audio)

    # 新增类型检查
    if method == "cqt":
        assert X.is_complex() or X.shape[-1] == 2, "CQT output must be complex"

    X = X.detach()
    out = decoder(X, length=audio.shape[-1])

    # 检查输出形状
    assert out.shape == audio.shape, f"Expected shape {audio.shape}, got {out.shape}"

    # 检查数据类型
    assert out.dtype == audio.dtype, f"Expected dtype {audio.dtype}, got {out.dtype}"

    # 检查数值范围
    assert torch.all(torch.isfinite(out)), "Output contains non-finite values"

    # 检查重建质量
    assert torch.allclose(audio, out, atol=tolerance), "Reconstruction quality is poor"


def test_nnAudioICQT():
    """测试nnAudioICQT类"""
    # 创建测试数据
    x = torch.randn(2, 2, 44100)  # (batch_size, channels, time)

    # 创建编码器和解码器
    encoder = nnAudioCQT()
    decoder = nnAudioICQT()

    # 设置编码器
    decoder.set_cqt_encoder(encoder)

    # 先进行CQT变换
    cqt_out = encoder(x)

    # 进行ICQT变换
    y = decoder(cqt_out)

    # 检查输出形状
    assert y.shape == x.shape, f"Expected shape {x.shape}, got {y.shape}"

    # 检查数据类型
    assert y.dtype == x.dtype, f"Expected dtype {x.dtype}, got {y.dtype}"

    # 检查数值范围
    assert torch.all(torch.isfinite(y)), "Output contains non-finite values"

    # 检查重建质量
    # 由于相位重建，可能会有一些差异，但幅度应该大致相同
    x_mag = torch.encoder(x, n_fft=2048, hop_length=512, return_complex=True).abs()
    y_mag = torch.encoder(y, n_fft=2048, hop_length=512, return_complex=True).abs()
    assert torch.allclose(x_mag, y_mag, rtol=0.1), "Reconstruction quality is poor"
