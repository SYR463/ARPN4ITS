import argparse

import numpy as np
import yaml
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.dataloader import get_dataloader_and_vocab
from utils.trainer import Trainer
from utils.helper import (
    get_model_class,
    get_optimizer_class,
    get_lr_scheduler,
    save_config,
    save_vocab,
)


def train(config):
    # 阶段1：使用Skip-Gram训练非叶节点
    train_dataloader, vocab = get_dataloader_and_vocab(
        model_name="skipgram",
        data_dir=config["data_dir"],
        # data_dir=config["data_dir_non_leaf"],
        batch_size=config["train_batch_size"],
        shuffle=config["shuffle"],
        # node_type="non_leaf"
    )

    val_dataloader, _ = get_dataloader_and_vocab(
        model_name=config["model_name"],
        # ds_name=config["dataset"],
        # ds_type="valid",
        data_dir=config["data_dir"],
        batch_size=config["val_batch_size"],
        shuffle=config["shuffle"],
        # vocab=vocab,
    )

    model_class = get_model_class("skipgram")
    model = model_class(vocab_size=len(vocab))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    lr_scheduler = get_lr_scheduler(optimizer, config["epochs"])

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

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
        model_name="skipgram"
    )
    trainer.train_phase(phase=1)

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

    train_dataloader, vocab = get_dataloader_and_vocab(
        model_name="cbow",
        data_dir=config["data_dir"],
        # data_dir=config["data_dir_non_leaf"],
        batch_size=config["train_batch_size"],
        shuffle=config["shuffle"],
        # node_type="non_leaf"
    )

    val_dataloader, _ = get_dataloader_and_vocab(
        model_name="cbow",
        # ds_name=config["dataset"],
        # ds_type="valid",
        data_dir=config["data_dir"],
        batch_size=config["val_batch_size"],
        shuffle=config["shuffle"],
        # vocab=vocab,
    )

    # 加载 Skip-Gram 训练完成后的 embedding 参数
    embedding_path = os.path.join(config["model_dir"], "skipgram_word_embeddings.npy")
    skipgram_embeddings = np.load(embedding_path)
    skipgram_embeddings = torch.tensor(skipgram_embeddings, dtype=torch.float32)
    embedding_layer = nn.Embedding.from_pretrained(skipgram_embeddings, freeze=True)

    model_class = get_model_class("cbow")
    model = model_class(vocab_size=len(vocab), skipgram_embeddings=embedding_layer)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    lr_scheduler = get_lr_scheduler(optimizer, config["epochs"])

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
        model_name="cbow"
    )
    trainer.train_phase(phase=2)

    print("Phase2: CBOW Training Finished.")

    trainer.save_model()
    trainer.save_loss()
    save_vocab(vocab, config["model_dir"])
    save_config(config, config["model_dir"])

    # 保存词向量（嵌入）
    embeddings = model.embeddings.weight.detach().cpu().numpy()  # 假设嵌入在 `model.embeddings` 中
    np.save(os.path.join(config["model_dir"], "cbow_word_embeddings.npy"), embeddings)
    print("CBOW Word embeddings saved.")

    print("Model artifacts saved to folder:", config["model_dir"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to yaml config')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    train(config)
