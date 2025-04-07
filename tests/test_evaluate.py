import pytest
import torch
import numpy as np
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from openunmix import evaluate


@pytest.fixture
def mock_track():
    track = MagicMock()
    track.audio = np.random.rand(44100 * 5, 2)  # 5秒立体声音频
    track.rate = 44100
    track.name = "test_track"
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
                filterbank="torch"
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

        scores = evaluate.separate_and_evaluate(
            track=mock_track,
            targets=["vocals", "drums", "bass", "other"],
            model_str_or_path="umxl",
            niter=1,
            output_dir=None,
            eval_dir=None,
            residual=None,
            mus=mock_musdb,
            aggregate_dict=aggregate_dict,
            device="cpu",
            wiener_win_len=None,
            filterbank="torch"
        )

        assert scores is not None


def test_residual_handling(mock_track, mock_musdb):
    # 创建专门处理残差的模拟分离器
    separator = MagicMock()
    separator.sample_rate = 44100

    def mock_separate(audio):
        return {
            "vocals": torch.rand(1, 2, audio.shape[-1]),
            "drums": torch.rand(1, 2, audio.shape[-1]),
            "bass": torch.rand(1, 2, audio.shape[-1]),
            "other": torch.rand(1, 2, audio.shape[-1]),
            "residual": torch.rand(1, 2, audio.shape[-1])
        }

    separator.side_effect = mock_separate

    with patch('openunmix.evaluate.utils.load_separator', return_value=separator), \
            patch('museval.eval_mus_track', return_value=MagicMock()):
        scores = evaluate.separate_and_evaluate(
            track=mock_track,
            targets=["vocals", "drums", "bass", "other"],
            model_str_or_path="umxl",
            niter=1,
            output_dir=None,
            eval_dir=None,
            residual="residual",
            mus=mock_musdb,
            aggregate_dict=None,
            device="cpu",
            wiener_win_len=None,
            filterbank="torch"
        )

        assert scores is not None


def test_f1_score_calculation(mock_track, mock_separator, mock_musdb):
    # 创建模拟的F1 score数据
    mock_track.audio = np.random.rand(44100 * 5, 2)  # 5秒立体声音频
    mock_track.rate = 44100

    # 设置分离器返回值
    mock_estimates = {
        "vocals": torch.rand(1, 2, 44100 * 5),
        "drums": torch.rand(1, 2, 44100 * 5),
        "bass": torch.rand(1, 2, 44100 * 5),
        "other": torch.rand(1, 2, 44100 * 5),
    }
    mock_separator.return_value = mock_estimates

    # 设置museval返回值，包含F1 score
    mock_scores = MagicMock()
    mock_scores.targets = [
        {
            "name": "bass",
            "frames": [
                {
                    "metrics": {
                        "F1 score: 0.85": 0.85
                    }
                }
            ]
        }
    ]

    with patch('openunmix.evaluate.utils.load_separator', return_value=mock_separator), \
            patch('museval.eval_mus_track', return_value=mock_scores):
        scores = evaluate.separate_and_evaluate(
            track=mock_track,
            targets=["vocals", "drums", "bass", "other"],
            model_str_or_path="umxl",
            niter=1,
            output_dir=None,
            eval_dir=None,
            residual=None,
            mus=mock_musdb,
            aggregate_dict=None,
            device="cpu",
            wiener_win_len=None,
            filterbank="torch"
        )

        # 验证F1 score是否正确提取
        assert scores is not None
        assert hasattr(scores, 'targets')
        assert len(scores.targets) > 0
        bass_target = next((t for t in scores.targets if t['name'] == 'bass'), None)
        assert bass_target is not None
        assert 'frames' in bass_target
        assert len(bass_target['frames']) > 0
        assert 'metrics' in bass_target['frames'][0]
        assert any('F1 score:' in k for k in bass_target['frames'][0]['metrics'].keys())
