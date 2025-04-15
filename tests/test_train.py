import pytest
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置后端为Agg，避免Tkinter错误
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from unittest.mock import patch, MagicMock

from openunmix import model
from openunmix import transforms
from scripts.train import plot_loss_history, train, valid

@pytest.fixture
def mock_model():
    """创建模拟的模型和编码器"""
    stft = MagicMock()
    encoder = MagicMock()
    unmix = MagicMock()
    
    # 设置模拟模型的返回值，确保有正确的维度
    mock_tensor = torch.tensor([[[0.5]]], requires_grad=True)  # [batch, channels, time]
    unmix.return_value = mock_tensor
    encoder.return_value = mock_tensor
    
    return unmix, encoder, stft

@pytest.fixture
def mock_data():
    """创建模拟的训练数据"""
    batch_size = 4
    channels = 2
    length = 44100
    x = torch.randn(batch_size, channels, length)
    y = torch.randn(batch_size, channels, length)
    return x, y

def test_plot_loss_history():
    """测试loss历史记录绘图功能"""
    # 创建模拟的loss数据
    batches_per_epoch = 10
    epochs = 1000
    total_batches = epochs * batches_per_epoch
    
    # 创建训练loss数据（每个batch一个值）
    train_losses = [1.0 - 0.0008 * (i // batches_per_epoch) + np.random.normal(0, 0.02) 
                   for i in range(total_batches)]
    train_losses = np.maximum(train_losses, 0)  # 确保loss不为负
    
    # 创建验证loss数据（每个epoch一个值）
    valid_losses = [1.2 - 0.0009 * i + np.random.normal(0, 0.03) 
                   for i in range(epochs)]
    valid_losses = np.maximum(valid_losses, 0)  # 确保loss不为负
    
    # 生成图表
    output_path = Path("tests/loss_history.png")
    plot_loss_history(train_losses, valid_losses, output_path, batches_per_epoch)
    
    # 验证文件是否创建
    assert output_path.exists()
    
    # 验证图像内容
    img = plt.imread(output_path)
    assert img.shape[0] > 0 and img.shape[1] > 0

def test_train_function(mock_model, mock_data):
    """测试训练函数"""
    unmix, encoder, stft = mock_model
    x, y = mock_data
    
    # 模拟设备
    device = torch.device("cpu")
    
    # 模拟优化器
    optimizer = MagicMock()
    
    # 模拟数据加载器
    train_sampler = [(x, y)]
    
    # 运行训练
    with patch('torch.nn.functional.mse_loss', return_value=torch.tensor(0.5, requires_grad=True)):
        with patch('openunmix.utils.AverageMeter') as mock_avg:
            mock_avg.return_value.avg = 0.5
            avg_loss, batch_losses = train(MagicMock(), unmix, encoder, device, train_sampler, optimizer)
    
    # 验证loss值
    assert isinstance(avg_loss, float)
    assert avg_loss >= 0
    assert isinstance(batch_losses, list)
    assert len(batch_losses) == len(train_sampler)

def test_valid_function(mock_model, mock_data):
    """测试验证函数"""
    unmix, encoder, stft = mock_model
    x, y = mock_data
    
    # 模拟设备
    device = torch.device("cpu")
    
    # 模拟数据加载器
    valid_sampler = [(x, y)]
    
    # 运行验证
    with patch('torch.nn.functional.mse_loss', return_value=torch.tensor(0.5)):
        with patch('openunmix.utils.AverageMeter') as mock_avg:
            mock_avg.return_value.avg = 0.5
            loss = valid(MagicMock(), unmix, encoder, device, valid_sampler)
    
    # 验证loss值
    assert isinstance(loss, float)
    assert loss >= 0

def test_loss_history_saving():
    """测试loss历史记录保存功能"""
    # 创建模拟的loss历史数据
    batches_per_epoch = 10
    epochs = 1000
    total_batches = epochs * batches_per_epoch
    
    # 创建训练loss数据（每个batch一个值）
    train_losses = [1.0 - 0.0008 * (i // batches_per_epoch) + np.random.normal(0, 0.02) 
                   for i in range(total_batches)]
    train_losses = np.maximum(train_losses, 0)
    
    # 创建验证loss数据（每个epoch一个值）
    valid_losses = [1.2 - 0.0009 * i + np.random.normal(0, 0.03) 
                   for i in range(epochs)]
    valid_losses = np.maximum(valid_losses, 0)
    
    # 保存数据
    np.save(Path("tests/train_loss_history.npy"), np.array(train_losses))
    np.save(Path("tests/valid_loss_history.npy"), np.array(valid_losses))
    
    # 验证文件是否创建
    assert Path("tests/train_loss_history.npy").exists()
    assert Path("tests/valid_loss_history.npy").exists()
    
    # 验证数据是否正确保存
    loaded_train = np.load(Path("tests/train_loss_history.npy"))
    loaded_valid = np.load(Path("tests/valid_loss_history.npy"))
    
    assert np.allclose(loaded_train, train_losses)
    assert np.allclose(loaded_valid, valid_losses)

def test_1000_epochs_loss_plot():
    """测试1000个epoch的loss绘图功能"""
    # 创建更真实的loss数据
    batches_per_epoch = 10
    epochs = 1000
    total_batches = epochs * batches_per_epoch
    
    # 创建训练loss数据（每个batch一个值）
    base_train = np.linspace(1.0, 0.2, total_batches)
    train_noise = np.random.normal(0, 0.02, total_batches)
    train_losses = base_train + train_noise
    train_losses = np.maximum(train_losses, 0)
    
    # 创建验证loss数据（每个epoch一个值）
    base_valid = np.linspace(1.2, 0.3, epochs)
    valid_noise = np.random.normal(0, 0.03, epochs)
    valid_losses = base_valid + valid_noise
    valid_losses = np.maximum(valid_losses, 0)
    
    # 生成图表
    output_path = Path("tests/1000_epochs_loss_history.png")
    plot_loss_history(train_losses, valid_losses, output_path, batches_per_epoch)
    
    # 验证文件是否创建
    assert output_path.exists()
    
    # 验证图像内容
    img = plt.imread(output_path)
    assert img.shape[0] > 0 and img.shape[1] > 0
    
    # 保存数据
    np.save(Path("tests/1000_epochs_train_loss.npy"), train_losses)
    np.save(Path("tests/1000_epochs_valid_loss.npy"), valid_losses)
    
    # 验证数据是否正确保存
    loaded_train = np.load(Path("tests/1000_epochs_train_loss.npy"))
    loaded_valid = np.load(Path("tests/1000_epochs_valid_loss.npy"))
    
    assert np.allclose(loaded_train, train_losses)
    assert np.allclose(loaded_valid, valid_losses)
    