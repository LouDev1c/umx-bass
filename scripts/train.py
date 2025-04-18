import argparse
import torch
import time
from pathlib import Path
import tqdm
import json
import sklearn.preprocessing
import numpy as np
import random
from git import Repo
import os
import copy
import torchaudio
import matplotlib.pyplot as plt

from openunmix import data
from openunmix import model
from openunmix import utils
from openunmix import transforms

tqdm.monitor_interval = 0


def train(args, unmix, encoder, device, train_sampler, optimizer):
    losses = utils.AverageMeter()
    batch_losses = []  # 记录每个batch的loss
    unmix.train()
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)
    for x, y in pbar:
        pbar.set_description("Training batch")
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        X = encoder(x)
        Y_hat = unmix(X)
        Y = encoder(y)
        loss = torch.nn.functional.mse_loss(Y_hat, Y)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), Y.size(1))
        batch_losses.append(loss.item())  # 记录当前batch的loss
        pbar.set_postfix(loss="{:.3f}".format(losses.avg))
    return losses.avg, batch_losses  # 返回平均loss和每个batch的loss


def valid(args, unmix, encoder, device, valid_sampler):
    losses = utils.AverageMeter()
    unmix.eval()
    with torch.no_grad():
        for x, y in valid_sampler:
            x, y = x.to(device), y.to(device)
            X = encoder(x)
            Y_hat = unmix(X)
            Y = encoder(y)
            loss = torch.nn.functional.mse_loss(Y_hat, Y)
            losses.update(loss.item(), Y.size(1))
        return losses.avg


def get_statistics(args, encoder, dataset):
    encoder = copy.deepcopy(encoder).to("cpu")
    scaler = sklearn.preprocessing.StandardScaler()

    dataset_scaler = copy.deepcopy(dataset)
    if isinstance(dataset_scaler, data.SourceFolderDataset):
        dataset_scaler.random_chunks = False
    else:
        dataset_scaler.random_chunks = False
        dataset_scaler.seq_duration = None

    dataset_scaler.samples_per_track = 1
    dataset_scaler.augmentations = None
    dataset_scaler.random_track_mix = False
    dataset_scaler.random_interferer_mix = False

    pbar = tqdm.tqdm(range(len(dataset_scaler)), disable=args.quiet)
    for ind in pbar:
        x, y = dataset_scaler[ind]
        pbar.set_description("Compute dataset statistics")
        # 降为单通道
        X = encoder(x[None, ...]).mean(1, keepdim=False).permute(0, 2, 1)
        scaler.partial_fit(np.squeeze(X))

    # 设置初始输入标量值
    std = np.maximum(scaler.scale_, 1e-4 * np.max(scaler.scale_))
    return scaler.mean_, std


def plot_loss_history(train_losses, valid_losses, output_path, batch_size=None):
    """绘制训练和验证loss曲线
    
    Args:
        train_losses: 每个batch的训练loss列表
        valid_losses: 每个epoch的验证loss列表
        output_path: 输出文件路径
        batch_size: 每个epoch的batch数量，用于计算x轴刻度
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制训练loss
    if batch_size is not None:
        # 计算每个batch的x轴位置
        x_train = np.arange(len(train_losses)) / batch_size
        plt.plot(x_train, train_losses, label='Training Loss (per batch)', alpha=0.5)
    else:
        plt.plot(train_losses, label='Training Loss (per batch)', alpha=0.5)
    
    # 绘制验证loss
    x_valid = np.arange(len(valid_losses))
    plt.plot(x_valid, valid_losses, label='Validation Loss (per epoch)', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss History')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Umx-Bass Trainer")

    # 训练目标
    parser.add_argument("--target", type=str, default="bass", help="target source (will be passed to the dataset)")

    # 数据集参数
    parser.add_argument("--dataset", type=str, default=r"\umx-bass\musdb", help="Name of the dataset.")
    parser.add_argument("--root", type=str, help="root path of dataset")
    parser.add_argument("--output", type=str, default="umx-bass-2", help="provide output path base folder name")
    parser.add_argument("--model", type=str, help="Name or path of pretrained model to fine-tune")
    parser.add_argument("--checkpoint", type=str, help="Path of checkpoint to resume training")
    parser.add_argument("--audio-backend", type=str, default="soundfile", help="Set torchaudio backend (`sox_io` or `soundfile`")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate, defaults to 1e-3")
    parser.add_argument("--patience", type=int, default=140, help="maximum number of train epochs (default: 140)")
    parser.add_argument("--lr-decay-patience", type=int, default=80, help="lr decay patience for plateau scheduler")
    parser.add_argument("--lr-decay-gamma", type=float, default=0.3, help="gamma of learning rate scheduler decay",)
    parser.add_argument("--weight-decay", type=float, default=0.00001, help="weight decay")
    parser.add_argument("--seed", type=int, default=42, metavar="S", help="random seed (default: 42)")

    # 模型参数
    parser.add_argument("--method", type=str, default="stft", help="Method for time/frequency domain transmission")
    parser.add_argument("--use-cqt", type=bool, default=False, help="Whether use cqt to replace STFT")
    parser.add_argument("--seq-dur", type=float, default=6.0, help="Sequence duration in seconds" "value of <=0.0 will use full/variable length")
    parser.add_argument("--unidirectional", action="store_true", default=False, help="Use unidirectional LSTM")
    parser.add_argument("--nfft", type=int, default=4096, help="STFT fft size and window size")
    parser.add_argument("--nhop", type=int, default=1024, help="STFT hop size")
    parser.add_argument("--hidden-size", type=int, default=512, help="hidden size parameter of bottleneck layers")
    parser.add_argument("--bandwidth", type=int, default=16000, help="maximum model bandwidth in herz")
    parser.add_argument("--nb-channels", type=int, default=2, help="set number of channels for model (1, 2)")
    parser.add_argument("--nb-workers", type=int, default=0, help="Number of workers for dataloader.")
    parser.add_argument("--debug", action="store_true", default=False, help="Speed up training init for dev purposes")

    # 混合参数
    parser.add_argument("--quiet", action="store_true", default=False, help="less verbose during training")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")

    args, _ = parser.parse_known_args()

    torchaudio.set_audio_backend(args.audio_backend)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Using GPU:", use_cuda)
    dataloader_kwargs = {"num_workers": args.nb_workers, "pin_memory": True} if use_cuda else {}

    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    repo = Repo(repo_dir)
    commit = repo.head.commit.hexsha[:7]

    # use jpg or npy
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset, valid_dataset, args = data.load_datasets(parser, args)

    # 如果输出目录不存在则新建一个
    target_path = Path(args.output)
    target_path.mkdir(parents=True, exist_ok=True)

    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **dataloader_kwargs
    )
    valid_sampler = torch.utils.data.DataLoader(valid_dataset, batch_size=1, **dataloader_kwargs)

    processor, _ = transforms.make_filterbanks(
        n_fft=args.nfft,
        n_hop=args.nhop,
        sample_rate=train_dataset.sample_rate,
        method=args.method,
        use_cqt=args.use_cqt,
    )
    encoder = torch.nn.Sequential(processor, model.ComplexNorm(mono=args.nb_channels == 1)).to(device)

    separator_conf = {
        "nfft": args.nfft,
        "nhop": args.nhop,
        "sample_rate": train_dataset.sample_rate,
        "nb_channels": args.nb_channels,
    }

    with open(Path(target_path, "separator.json"), "w") as outfile:
        outfile.write(json.dumps(separator_conf, indent=4, sort_keys=True))

    if args.checkpoint or args.model or args.debug:
        scaler_mean = None
        scaler_std = None
    else:
        scaler_mean, scaler_std = get_statistics(args, encoder, train_dataset)

    max_bin = utils.bandwidth_to_max_bin(train_dataset.sample_rate, args.nfft, args.bandwidth)

    if args.model:
        # fine tune model
        print(f"Fine-tuning model from {args.model}")
        unmix = utils.load_target_models(
            args.target, model_str_or_path=args.model, device=device, pretrained=True
        )[args.target]
        unmix = unmix.to(device)
    else:
        unmix = model.OpenUnmix(
            input_mean=scaler_mean,
            input_scale=scaler_std,
            nb_bins=args.nfft // 2 + 1,
            nb_channels=args.nb_channels,
            hidden_size=args.hidden_size,
            max_bin=max_bin,
            unidirectional=args.unidirectional,
        ).to(device)

    optimizer = torch.optim.Adam(unmix.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10,
    )

    es = utils.EarlyStopping(patience=args.patience)

    # 如果检查点已生成，则继续训练
    if args.checkpoint:
        model_path = Path(args.checkpoint).expanduser()
        with open(Path(model_path, args.target + ".json"), "r") as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, args.target + ".chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        unmix.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        # train for another epochs_trained
        t = tqdm.trange(
            results["epochs_trained"],
            results["epochs_trained"] + args.epochs + 1,
            disable=args.quiet,
        )
        train_losses = results["train_loss_history"]
        valid_losses = results["valid_loss_history"]
        train_times = results["train_time_history"]
        best_epoch = results["best_epoch"]
        es.best = results["best_loss"]
        es.num_bad_epochs = results["num_bad_epochs"]
    # 否则从头开启优化器
    else:
        t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
        train_losses = []
        valid_losses = []
        train_times = []
        best_epoch = 0

    # 添加loss历史记录
    train_loss_history = []  # 记录每个batch的loss
    valid_loss_history = []  # 记录每个epoch的loss
    batch_size = len(train_sampler)  # 获取每个epoch的batch数量

    for epoch in t:
        t.set_description("Training epoch")
        end = time.time()
        train_loss, batch_losses = train(args, unmix, encoder, device, train_sampler, optimizer)
        valid_loss = valid(args, unmix, encoder, device, valid_sampler)
        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        # 记录loss历史
        train_loss_history.extend(batch_losses)  # 添加所有batch的loss
        valid_loss_history.append(valid_loss)  # 添加epoch的验证loss
        
        t.set_postfix(train_loss=train_loss, val_loss=valid_loss)

        stop = es.step(valid_loss)
        
        # 每10个epoch保存一次loss图
        if epoch % 10 == 0:
            plot_loss_history(
                train_loss_history, 
                valid_loss_history,
                Path(target_path, f"{args.output}_loss_history_epoch_{epoch}.png"),
                batch_size
            )
        
        stop = es.step(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch

        utils.save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": unmix.state_dict(),
                "best_loss": es.best,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best=valid_loss == es.best,
            path=target_path,
            target=args.target,
        )

        # save params
        params = {
            "epochs_trained": epoch,
            "args": vars(args),
            "best_loss": es.best,
            "best_epoch": best_epoch,
            "train_loss_history": train_losses,
            "valid_loss_history": valid_losses,
            "train_time_history": train_times,
            "num_bad_epochs": es.num_bad_epochs,
            "commit": commit,
        }

        with open(Path(target_path, args.target + ".json"), "w") as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        train_times.append(time.time() - end)

        if stop:
            print("Apply Early Stopping")
            break

    # 训练结束后保存最终的loss图
    plot_loss_history(
        train_loss_history,
        valid_loss_history,
        Path(target_path, f"{args.output}_final_loss_history.png"),
        batch_size
    )
    
    # 保存loss历史数据
    np.save(Path(target_path, "train_loss_history.npy"), np.array(train_loss_history))
    np.save(Path(target_path, "valid_loss_history.npy"), np.array(valid_loss_history))


if __name__ == "__main__":
    main()
