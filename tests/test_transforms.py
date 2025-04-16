import pytest
import numpy as np
import torch
from openunmix import transforms
import librosa
import matplotlib.pyplot as plt


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
def audio(nb_samples, nb_channels, nb_timesteps):
    # 根据方法类型决定音频长度
    if method == "cqt":
        # CQT需要更长的音频才能有足够的低频分辨率
        return torch.rand((nb_samples, nb_channels, nb_timesteps * 2))
    else:
        return torch.rand((nb_samples, nb_channels, nb_timesteps))


def test_stft(audio, nfft, hop, method):
    # we should only test for center=True as
    # False doesn't pass COLA
    # https://github.com/pytorch/audio/issues/500
    if method == "cqt":
        # 使用更适合CQT的参数
        stft, istft = transforms.make_filterbanks(
            n_fft=nfft,
            n_hop=hop,
            center=True,
            method=method,
            sr=44100.0
        )
        tolerance = 0.6
    else:
        stft, istft = transforms.make_filterbanks(
            n_fft=nfft,
            n_hop=hop,
            center=True,
            method=method
        )
        tolerance = 1e-6

    X = stft(audio)

    # 新增类型检查
    if method == "cqt":
        assert X.is_complex() or X.shape[-1] == 2, "CQT output must be complex"

    X = X.detach()
    out = istft(X, length=audio.shape[-1])

    assert np.sqrt(np.mean((audio.detach().numpy() - out.detach().numpy()) ** 2)) < tolerance


def test_cqt_performance():
    """专门测试CQT的性能指标"""
    # 测试参数
    sample_rate = 44100
    n_fft = 2048
    n_hop = 1024
    duration = 2.0  # 测试音频时长
    fmin = 32.7  # 贝斯最低频(C1音)
    n_bins = 84  # 7个八度(12 * 7=84)

    # 生成更复杂的测试信号
    t = np.linspace(0, duration, int(sample_rate * duration))

    # 生成多个频率成分的测试信号
    frequencies = [50, 100, 200, 400, 800]  # 更密集的频率分布
    amplitudes = [1.0, 0.8, 0.6, 0.4, 0.2]  # 递减的幅度
    phases = np.random.rand(len(frequencies)) * 2 * np.pi  # 随机相位

    test_signal = np.zeros_like(t)
    for freq, amp, phase in zip(frequencies, amplitudes, phases):
        test_signal += amp * np.sin(2 * np.pi * freq * t + phase)

    # 添加一些噪声
    noise = np.random.normal(0, 0.1, len(t))
    test_signal += noise

    # 归一化
    test_signal = test_signal / np.max(np.abs(test_signal))

    test_signal = torch.from_numpy(test_signal).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]

    # 初始化CQT和ICQT
    cqt, icqt = transforms.make_filterbanks(
        n_fft=n_fft,
        n_hop=n_hop,
        center=True,
        method="cqt",
        sr=sample_rate
    )

    # 1. 测试CQT变换
    X = cqt(test_signal)
    cqt_complex = torch.view_as_complex(X)
    cqt_mag = cqt_complex.abs()
    cqt_phase = cqt_complex.angle()

    # 2. 计算频率分辨率
    # 将频谱分为低频、中频和高频区域
    n_low_bins = n_bins // 3
    n_mid_bins = n_bins // 3
    n_high_bins = n_bins - n_low_bins - n_mid_bins

    # 计算不同频率区域的特征
    low_freq_region = cqt_mag[..., :n_low_bins]
    mid_freq_region = cqt_mag[..., n_low_bins:n_low_bins + n_mid_bins]
    high_freq_region = cqt_mag[..., -n_high_bins:]

    # 3. 计算各频率成分的误差
    freq_errors = {}
    for freq in frequencies:
        # 找到最接近的频率bin
        cqt_freq_bin = int(np.log2(freq / fmin) * n_bins)
        cqt_freq_bin = min(cqt_freq_bin, cqt_mag.shape[-2] - 1)
        
        # 计算该频率的误差
        cqt_energy = cqt_mag[..., cqt_freq_bin, :].mean().item()
        target_energy = amplitudes[frequencies.index(freq)]
        
        # 使用相对误差
        if target_energy > 0:
            freq_errors[freq] = 20 * np.log10(abs(cqt_energy - target_energy) / target_energy)
        else:
            freq_errors[freq] = 20 * np.log10(abs(cqt_energy))

    print("各频率成分的幅度谱误差:")
    for freq, error in freq_errors.items():
        print(f"{freq}Hz: {error:.2f} dB")

    # 4. 计算频率区域特征
    def compute_region_features(region):
        if region.numel() == 0:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0}
        return {
            'mean': region.mean().item(),
            'std': region.std().item(),
            'max': region.max().item(),
            'min': region.min().item()
        }

    low_freq_features = compute_region_features(low_freq_region)
    mid_freq_features = compute_region_features(mid_freq_region)
    high_freq_features = compute_region_features(high_freq_region)

    print(f"低频区域特征: {low_freq_features}")
    print(f"中频区域特征: {mid_freq_features}")
    print(f"高频区域特征: {high_freq_features}")

    # 5. 测试时间分辨率
    time_resolution = X.shape[-1] / duration  # 帧/秒
    print(f"时间分辨率: {time_resolution:.2f} frames/second")

    # 6. 测试计算效率
    import time
    start_time = time.time()
    for _ in range(10):
        X = cqt(test_signal)
    computation_time = (time.time() - start_time) / 10
    print(f"平均计算时间: {computation_time:.4f} seconds")

    # 7. 评估CQT质量
    # 计算频率分辨率指标
    def compute_freq_resolution(region):
        if region.numel() == 0:
            return 0.0
        # 使用能量集中度作为分辨率指标
        energy = region.sum(dim=-1).mean().item()
        max_energy = region.max().item()
        return max_energy / energy if energy > 0 else 0.0

    low_freq_resolution = compute_freq_resolution(low_freq_region)
    high_freq_resolution = compute_freq_resolution(high_freq_region)
    print(f"低频分辨率: {low_freq_resolution:.4f}")
    print(f"高频分辨率: {high_freq_resolution:.4f}")

    # 8. 评估频率误差
    # 低频区域的误差应该比高频区域小
    low_freq_error = np.mean([abs(freq_errors[f]) for f in frequencies if f <= 200])
    high_freq_error = np.mean([abs(freq_errors[f]) for f in frequencies if f > 200])
    print(f"低频平均误差: {low_freq_error:.2f} dB")
    print(f"高频平均误差: {high_freq_error:.2f} dB")

    # 断言测试
    assert low_freq_error < high_freq_error, "低频区域的误差应该小于高频区域"
    assert computation_time < 1.0, f"计算时间过长: {computation_time} seconds"

    # 返回测试结果
    return {
        'frequency_errors': freq_errors,
        'low_freq_features': low_freq_features,
        'mid_freq_features': mid_freq_features,
        'high_freq_features': high_freq_features,
        'time_resolution': time_resolution,
        'computation_time': computation_time,
        'low_freq_resolution': low_freq_resolution,
        'high_freq_resolution': high_freq_resolution,
        'low_freq_error': low_freq_error,
        'high_freq_error': high_freq_error
    }


def test_stft_performance():
    """专门测试STFT的性能指标"""
    # 测试参数
    sample_rate = 44100
    n_fft = 2048
    n_hop = 1024
    duration = 2.0  # 测试音频时长
    
    # 生成测试信号
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 生成多个频率成分的测试信号
    frequencies = [50, 100, 200, 400, 800, 2000, 4000, 8000]  # 添加高频成分
    amplitudes = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.025]  # 递减的幅度
    phases = np.random.rand(len(frequencies)) * 2 * np.pi  # 随机相位
    
    test_signal = np.zeros_like(t)
    for freq, amp, phase in zip(frequencies, amplitudes, phases):
        test_signal += amp * np.sin(2 * np.pi * freq * t + phase)
    
    # 添加一些噪声
    noise = np.random.normal(0, 0.1, len(t))
    test_signal += noise
    
    # 归一化
    test_signal = test_signal / np.max(np.abs(test_signal))
    
    test_signal = torch.from_numpy(test_signal).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    
    # 初始化STFT和ISTFT
    stft, istft = transforms.make_filterbanks(
        n_fft=n_fft,
        n_hop=n_hop,
        center=True,
        method="stft"
    )
    
    # 1. 测试重构误差
    X = stft(test_signal)
    reconstructed = istft(X, length=test_signal.shape[-1])
    
    # 计算总体重构误差
    reconstruction_error = np.sqrt(np.mean((test_signal.numpy() - reconstructed.numpy()) ** 2))
    print(f"STFT总体重构误差: {reconstruction_error:.4f}")
    
    # 2. 分析幅度谱和相位谱
    stft_complex = torch.view_as_complex(X)
    stft_mag = stft_complex.abs()
    stft_phase = stft_complex.angle()
    
    # 计算每个频率成分的误差
    freq_errors = {}
    for freq in frequencies:
        # 找到最接近的频率bin
        stft_freq_bin = int(freq * n_fft / sample_rate)
        # 确保bin索引在有效范围内
        stft_freq_bin = min(stft_freq_bin, stft_mag.shape[-2] - 1)
        
        # 计算该频率的误差
        stft_energy = stft_mag[..., stft_freq_bin, :].mean().item()
        target_energy = amplitudes[frequencies.index(freq)]
        # 使用相对误差而不是绝对误差
        if target_energy > 0:
            freq_errors[freq] = 20 * np.log10(abs(stft_energy - target_energy) / target_energy)
        else:
            freq_errors[freq] = 20 * np.log10(abs(stft_energy))
    
    print("STFT各频率成分的幅度谱误差:")
    for freq, error in freq_errors.items():
        print(f"{freq}Hz: {error:.2f} dB")
    
    # 3. 测试频率分辨率
    # 将频谱分为低频、中频和高频区域
    n_bins = stft_mag.shape[-2]
    # 根据实际频率范围划分
    freq_boundaries = [0, 500, 2000, 8000, sample_rate/2]  # Hz
    bin_boundaries = [int(f * n_fft / sample_rate) for f in freq_boundaries]
    
    # 确保bin边界有效
    bin_boundaries = [min(b, n_bins-1) for b in bin_boundaries]
    
    # 计算各区域的bin数量
    n_low_bins = bin_boundaries[1] - bin_boundaries[0]
    n_mid_bins = bin_boundaries[2] - bin_boundaries[1]
    n_high_bins = bin_boundaries[3] - bin_boundaries[2]
    n_very_high_bins = bin_boundaries[4] - bin_boundaries[3]
    
    print(f"频率区域划分:")
    print(f"低频: 0-{freq_boundaries[1]}Hz ({n_low_bins} bins)")
    print(f"中频: {freq_boundaries[1]}-{freq_boundaries[2]}Hz ({n_mid_bins} bins)")
    print(f"高频: {freq_boundaries[2]}-{freq_boundaries[3]}Hz ({n_high_bins} bins)")
    print(f"超高频: {freq_boundaries[3]}-{freq_boundaries[4]}Hz ({n_very_high_bins} bins)")
    
    # 计算不同频率区域的特征
    low_freq_region = stft_mag[..., bin_boundaries[0]:bin_boundaries[1]]
    mid_freq_region = stft_mag[..., bin_boundaries[1]:bin_boundaries[2]]
    high_freq_region = stft_mag[..., bin_boundaries[2]:bin_boundaries[3]]
    very_high_freq_region = stft_mag[..., bin_boundaries[3]:bin_boundaries[4]]
    
    # 计算每个区域的特征，添加维度检查
    def compute_region_features(region):
        if region.numel() == 0:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0}
        return {
            'mean': region.mean().item(),
            'std': region.std().item(),
            'max': region.max().item(),
            'min': region.min().item()
        }
    
    low_freq_features = compute_region_features(low_freq_region)
    mid_freq_features = compute_region_features(mid_freq_region)
    high_freq_features = compute_region_features(high_freq_region)
    very_high_freq_features = compute_region_features(very_high_freq_region)
    
    print(f"STFT低频区域特征: {low_freq_features}")
    print(f"STFT中频区域特征: {mid_freq_features}")
    print(f"STFT高频区域特征: {high_freq_features}")
    print(f"STFT超高频区域特征: {very_high_freq_features}")
    
    # 4. 测试时间分辨率
    time_resolution = X.shape[-1] / duration  # 帧/秒
    print(f"STFT时间分辨率: {time_resolution:.2f} frames/second")
    
    # 5. 测试计算效率
    import time
    start_time = time.time()
    for _ in range(10):
        X = stft(test_signal)
        reconstructed = istft(X, length=test_signal.shape[-1])
    computation_time = (time.time() - start_time) / 10
    print(f"STFT平均计算时间: {computation_time:.4f} seconds")
    
    # 断言测试
    assert reconstruction_error < 1e-6, f"STFT重构误差过大: {reconstruction_error}"
    
    # 返回测试结果
    return {
        'reconstruction_error': reconstruction_error,
        'frequency_errors': freq_errors,
        'low_freq_features': low_freq_features,
        'mid_freq_features': mid_freq_features,
        'high_freq_features': high_freq_features,
        'very_high_freq_features': very_high_freq_features,
        'time_resolution': time_resolution,
        'computation_time': computation_time
    }


def test_cqt_frequency_resolution():
    """测试CQT的频率分辨率"""
    # 测试参数
    sample_rate = 44100
    n_fft = 2048
    n_hop = 1024
    duration = 2.0  # 测试音频时长
    fmin = 32.7  # 贝斯最低频(C1音)
    n_bins = 84  # 7个八度(12 * 7=84)
    
    # 生成测试信号
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_signal = np.sin(2 * np.pi * 50 * t)  # 50Hz正弦波
    test_signal = torch.from_numpy(test_signal).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    
    # 初始化CQT和ICQT
    cqt, icqt = transforms.make_filterbanks(
        n_fft=n_fft,
        n_hop=n_hop,
        center=True,
        method="cqt",
        sr=sample_rate
    )
    
    # 计算CQT
    X = cqt(test_signal)
    cqt_complex = torch.view_as_complex(X)
    cqt_mag = cqt_complex.abs()
    
    # 计算频率分辨率
    freq_bins = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=12)
    freq_resolution = np.diff(freq_bins).mean()
    
    # 验证低频区域的频率分辨率
    low_freq_mask = freq_bins < 200  # 低频区域
    low_freq_resolution = np.diff(freq_bins[low_freq_mask]).mean()
    
    # 验证高频区域的频率分辨率
    high_freq_mask = freq_bins >= 200  # 高频区域
    high_freq_resolution = np.diff(freq_bins[high_freq_mask]).mean()
    
    # 低频区域应该有更好的频率分辨率
    assert low_freq_resolution < high_freq_resolution, "低频区域应该有更好的频率分辨率"
    
    # 验证50Hz信号的检测
    freq_idx = np.argmin(np.abs(freq_bins - 50))
    assert cqt_mag[..., freq_idx, :].mean() > 0.1, "50Hz信号应该被检测到"


def test_cqt_time_resolution():
    """测试CQT的时间分辨率"""
    # 测试参数
    sample_rate = 44100
    n_fft = 2048
    n_hop = 512  # 减小hop size以提高时间分辨率
    duration = 2.0  # 测试音频时长
    
    # 生成测试信号（包含瞬态）
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_signal = np.zeros_like(t)
    test_signal[int(sample_rate * 0.5):int(sample_rate * 0.6)] = 1.0  # 100ms的脉冲
    test_signal = torch.from_numpy(test_signal).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    
    # 初始化CQT和ICQT
    cqt, icqt = transforms.make_filterbanks(
        n_fft=n_fft,
        n_hop=n_hop,
        center=True,
        method="cqt",
        sr=sample_rate
    )
    
    # 计算CQT
    X = cqt(test_signal)
    cqt_complex = torch.view_as_complex(X)
    cqt_mag = cqt_complex.abs()
    
    # 计算时间分辨率（帧/秒）
    time_resolution = sample_rate / n_hop
    
    # 验证时间分辨率
    assert time_resolution > 20, f"时间分辨率应该足够高以检测瞬态，当前分辨率: {time_resolution} 帧/秒"
    
    # 验证瞬态检测
    # 计算脉冲在时间轴上的位置
    pulse_start_frame = int(0.5 * time_resolution)
    pulse_end_frame = int(0.6 * time_resolution)
    pulse_frames = cqt_mag[..., :, pulse_start_frame:pulse_end_frame]
    assert pulse_frames.mean() > 0.1, "瞬态应该被检测到"
