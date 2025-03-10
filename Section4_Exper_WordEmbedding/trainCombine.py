import argparse

import numpy as np
import yaml
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.dataloader import get_dataloader, load_vocab
from utils.trainer import Trainer
from utils.helper import (
    get_model_class,
    get_optimizer_class,
    get_lr_scheduler,
    save_config,
    save_vocab,
)


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    加载模型和优化器的检查点

    Args:
        model (nn.Module): 待加载的模型
        optimizer (torch.optim.Optimizer): 待加载的优化器
        checkpoint_path (str): 检查点文件路径

    Returns:
        model, optimizer, epoch (tuple): 返回加载后的模型、优化器和 epoch
    """
    if os.path.exists(checkpoint_path):
        print(f"Loading existing model checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)

        # 加载模型和优化器的状态
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded. Continuing from epoch {epoch}.")
    else:
        epoch = 0
        print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")

    return model, optimizer, epoch


def train(config):
    # os.makedirs(config["model_dir"])

    # 阶段1：使用Skip-Gram训练非叶节点
    model_name = "skipgram"
    vocab = load_vocab("/root/ARPN4ITS/vocab/vocab_Exper.json")  # 加载词汇表

    train_dataloader = get_dataloader(
        model_name=model_name,
        vocab=vocab,
        data_dir=config["train_data_dir"],
        # data_dir=config["data_dir_non_leaf"],
        batch_size=config["train_batch_size"],
        shuffle=config["shuffle"],
        filter=config['filter'],
    )

    val_dataloader = get_dataloader(
        model_name=model_name,
        vocab=vocab,
        # ds_name=config["dataset"],
        # ds_type="valid",
        data_dir=config["val_data_dir"],
        batch_size=config["val_batch_size"],
        shuffle=config["shuffle"],
        filter=config['filter'],
    )

    model_class = get_model_class(model_name)
    model = model_class(vocab_size=len(vocab))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    lr_scheduler = get_lr_scheduler(optimizer, config["epochs"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # 加载模型检查点（如果有）
    checkpoint_path = os.path.join(config["model_dir"], "model.pt")
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)


    trainer = Trainer(
        model=model,
        epochs=config["epochs"],
        train_dataloader=train_dataloader,
        train_steps=config["train_steps"],
        val_dataloader=val_dataloader,
        val_steps=config["val_steps"],
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_frequency=config["checkpoint_frequency"],
        lr_scheduler=lr_scheduler,
        device=device,
        model_dir=config["model_dir"],
        model_name=model_name
    )
    trainer.train()

    print("Phase1: Skip-Gram Training Finished.")

    trainer.save_model()
    trainer.save_loss()
    save_vocab(vocab, config["model_dir"])
    save_config(config, config["model_dir"])

    # 保存词向量（嵌入）
    embeddings = model.embeddings.weight.detach().cpu().numpy()  # 假设嵌入在 `model.embeddings` 中
    np.save(os.path.join(config["model_dir"], "skipgram_word_embeddings.npy"), embeddings)
    print("Skip-Gram Word embeddings saved.")

    print("Model artifacts saved to folder:", config["model_dir"])

    print("--------------------------------------")
    print()

    # 阶段2：使用CBOW训练叶节点

    model_name = "cbow"

    train_dataloader = get_dataloader(
        model_name=model_name,
        vocab=vocab,
        data_dir=config["train_data_dir"],
        # data_dir=config["data_dir_non_leaf"],
        batch_size=config["train_batch_size"],
        shuffle=config["shuffle"],
        filter=config['filter'],
    )

    val_dataloader = get_dataloader(
        model_name=model_name,
        vocab=vocab,
        # ds_name=config["dataset"],
        # ds_type="valid",
        data_dir=config["val_data_dir"],
        batch_size=config["val_batch_size"],
        shuffle=config["shuffle"],
        filter=config['filter'],
    )

    # 加载 Skip-Gram 训练完成后的 embedding 参数
    embedding_path = os.path.join(config["model_dir"], "skipgram_word_embeddings.npy")
    skipgram_embeddings = np.load(embedding_path)
    skipgram_embeddings = torch.tensor(skipgram_embeddings, dtype=torch.float32)
    # freeze=True 冻结Skip-Gram的参数，不进行更新
    embedding_layer = nn.Embedding.from_pretrained(skipgram_embeddings, freeze=False)

    model_class = get_model_class(model_name)
    model = model_class(vocab_size=len(vocab), skipgram_embeddings=embedding_layer)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    lr_scheduler = get_lr_scheduler(optimizer, config["epochs"])

    # 加载模型检查点（如果有）
    checkpoint_path = os.path.join(config["model_dir"], "model.pt")
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    trainer = Trainer(
        model=model,
        epochs=config["epochs"],
        train_dataloader=train_dataloader,
        train_steps=config["train_steps"],
        val_dataloader=val_dataloader,
        val_steps=config["val_steps"],
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_frequency=config["checkpoint_frequency"],
        lr_scheduler=lr_scheduler,
        device=device,
        model_dir=config["model_dir"],
        model_name=model_name
    )
    trainer.train()

    print("Phase2: CBOW Training Finished.")

    trainer.save_model()
    trainer.save_loss()
    save_vocab(vocab, config["model_dir"])
    save_config(config, config["model_dir"])

    # 保存词向量（嵌入）
    embeddings = model.embeddings.weight.detach().cpu().numpy()  # 假设嵌入在 `model.embeddings` 中
    np.save(os.path.join(config["model_dir"], "word_embeddings.npy"), embeddings)
    print("CBOW Word embeddings saved.")

    print("Model artifacts saved to folder:", config["model_dir"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to yaml config')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    train(config)
