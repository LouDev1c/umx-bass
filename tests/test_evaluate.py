import pytest
import torch
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock, mock_open
import musdb
import museval
from openunmix import evaluate
import json
import sys


@pytest.fixture
def mock_track():
    track = MagicMock()
    track.audio = np.random.rand(44100 * 5, 2)  # 5秒立体声音频
    track.rate = 44100
    track.name = "test_track"
    
    # 添加bass目标
    bass_target = MagicMock()
    bass_target.audio = np.random.rand(44100 * 5, 2)  # 5秒立体声音频
    track.targets = {"bass": bass_target}
    
    return track


@pytest.fixture
def mock_musdb(mock_track):
    mus = MagicMock()
    mus.tracks = [mock_track]
    mus.save_estimates = MagicMock()
    return mus


@pytest.fixture
def mock_separator():
    separator = MagicMock()
    separator.sample_rate = 44100
    separator.freeze.return_value = None
    separator.to.return_value = None

    # 设置分离器返回值
    mock_estimates = {
        "vocals": torch.rand(1, 2, 44100 * 5),
        "drums": torch.rand(1, 2, 44100 * 5),
        "bass": torch.rand(1, 2, 44100 * 5),
        "other": torch.rand(1, 2, 44100 * 5),
    }
    separator.return_value = mock_estimates

    # 正确设置to_dict方法
    def to_dict(estimates, aggregate_dict=None):
        if aggregate_dict:
            return {
                "vocals": estimates["vocals"],
                "drums": estimates["drums"],
                "bass": estimates["bass"],
                "other": estimates["other"]
            }
        return estimates

    separator.to_dict.side_effect = to_dict
    return separator


@pytest.fixture
def mock_museval():
    scores = MagicMock()
    return scores


def test_separate_and_evaluate(mock_track, mock_separator, mock_museval, mock_musdb):
    with patch('openunmix.evaluate.utils.load_separator', return_value=mock_separator), \
            patch('museval.eval_mus_track', return_value=mock_museval):
        with tempfile.TemporaryDirectory() as output_dir, \
                tempfile.TemporaryDirectory() as eval_dir:
            scores = evaluate.separate_and_evaluate(
                track=mock_track,
                targets=["vocals", "drums", "bass", "other"],
                model_str_or_path="umxl",
                niter=1,
                output_dir=output_dir,
                eval_dir=eval_dir,
                residual=False,
                mus=mock_musdb,
                aggregate_dict=None,
                device="cpu",
                wiener_win_len=300,
                filterbank="stft"
            )

            assert scores is not None
            mock_musdb.save_estimates.assert_called_once()


def test_aggregate_dict_handling(mock_track, mock_separator, mock_musdb):
    with patch('openunmix.evaluate.utils.load_separator', return_value=mock_separator), \
            patch('museval.eval_mus_track', return_value=MagicMock()):
        aggregate_dict = {
            "vocals": ["vocals"],
            "drums": ["drums"],
            "bass": ["bass"],
            "other": ["other"]
        }
        with tempfile.TemporaryDirectory() as output_dir, \
                tempfile.TemporaryDirectory() as eval_dir:
            scores = evaluate.separate_and_evaluate(
                track=mock_track,
                targets=["vocals", "drums", "bass", "other"],
                model_str_or_path="umxl",
                niter=1,
                output_dir=output_dir,
                eval_dir=eval_dir,
                residual=False,
                mus=mock_musdb,
                aggregate_dict=aggregate_dict,
                device="cpu",
                wiener_win_len=300,
                filterbank="stft"
            )

            assert scores is not None
            mock_musdb.save_estimates.assert_called_once()


def test_detect_notes():
    # 创建一个简单的测试信号
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # 生成一个包含两个明显音符的信号
    signal = np.zeros_like(t)
    # 使用更大的幅度和更长的持续时间
    signal[1000:3000] = 0.5 * np.sin(2 * np.pi * 440 * t[1000:3000])  # 第一个音符
    signal[4000:6000] = 0.5 * np.sin(2 * np.pi * 880 * t[4000:6000])  # 第二个音符
    
    # 检测音符
    notes = evaluate.detect_notes(signal, sr)
    
    # 验证检测结果
    assert len(notes) > 0  # 应该检测到至少一个音符
    assert all(0 <= note < len(signal) for note in notes)  # 音符位置应该在信号长度范围内


def test_calculate_f1_score():
    # 创建测试信号
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # 创建原始信号（包含两个明显音符）
    original_signal = np.zeros_like(t)
    # 使用更低的频率和更大的幅度
    freq1 = 100  # 低音频率
    freq2 = 200  # 低音频率
    # 添加谐波成分
    original_signal[1000:3000] = (
        0.8 * np.sin(2 * np.pi * freq1 * t[1000:3000]) +  # 基频
        0.4 * np.sin(2 * np.pi * 2 * freq1 * t[1000:3000]) +  # 二次谐波
        0.2 * np.sin(2 * np.pi * 3 * freq1 * t[1000:3000])  # 三次谐波
    )
    original_signal[4000:6000] = (
        0.8 * np.sin(2 * np.pi * freq2 * t[4000:6000]) +  # 基频
        0.4 * np.sin(2 * np.pi * 2 * freq2 * t[4000:6000]) +  # 二次谐波
        0.2 * np.sin(2 * np.pi * 3 * freq2 * t[4000:6000])  # 三次谐波
    )
    
    # 创建估计信号（包含一个正确的音符和一个错误的音符）
    estimated_signal = np.zeros_like(t)
    # 正确的音符（与原始信号中的第一个音符相同）
    estimated_signal[1000:3000] = (
        0.8 * np.sin(2 * np.pi * freq1 * t[1000:3000]) +  # 基频
        0.4 * np.sin(2 * np.pi * 2 * freq1 * t[1000:3000]) +  # 二次谐波
        0.2 * np.sin(2 * np.pi * 3 * freq1 * t[1000:3000])  # 三次谐波
    )
    # 错误的音符（使用不同的频率）
    estimated_signal[6000:8000] = (
        0.8 * np.sin(2 * np.pi * 150 * t[6000:8000]) +  # 基频
        0.4 * np.sin(2 * np.pi * 2 * 150 * t[6000:8000]) +  # 二次谐波
        0.2 * np.sin(2 * np.pi * 3 * 150 * t[6000:8000])  # 三次谐波
    )
    
    # 计算F1分数
    f1 = evaluate.calculate_f1_score(original_signal, estimated_signal, sr)
    
    # 验证F1分数
    assert 0 <= f1 <= 1  # F1分数应该在0到1之间
    assert f1 > 0  # 由于有一个正确的音符，F1分数应该大于0
    assert f1 < 1  # 由于有一个错误的音符，F1分数应该小于1


def test_f1_score_integration(mock_track, mock_separator, mock_museval, mock_musdb):
    with patch('openunmix.evaluate.utils.load_separator', return_value=mock_separator), \
            patch('museval.eval_mus_track', return_value=mock_museval):
        with tempfile.TemporaryDirectory() as output_dir, \
                tempfile.TemporaryDirectory() as eval_dir:
            # 设置mock_track的bass音频
            bass_audio = np.random.rand(44100 * 5, 2)
            mock_track.targets = {
                "bass": MagicMock(audio=bass_audio)
            }
            
            scores = evaluate.separate_and_evaluate(
                track=mock_track,
                targets=["vocals", "drums", "bass", "other"],
                model_str_or_path="umxl",
                niter=1,
                output_dir=output_dir,
                eval_dir=eval_dir,
                residual=False,
                mus=mock_musdb,
                aggregate_dict=None,
                device="cpu",
                wiener_win_len=300,
                filterbank="stft"
            )
            
            # 验证F1分数是否被添加到scores中
            if isinstance(scores, str):
                scores_dict = json.loads(scores)
                assert "bass_f1" in scores_dict
                assert 0 <= scores_dict["bass_f1"] <= 1
            else:
                assert hasattr(scores, "bass_f1")
                assert 0 <= scores.bass_f1 <= 1
