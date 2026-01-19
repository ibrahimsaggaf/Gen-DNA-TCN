import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from utils import reader_autoregressive


def fit(file_train_seq, file_val_seq, gen_tcn, clf, batch_size, lr, epochs, hist_len, clip, device,
        model, seed, res_path):
    torch.manual_seed(seed)

    train_data = reader_autoregressive(file_train_seq)
    val_data = reader_autoregressive(file_val_seq)
    train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, drop_last=True, shuffle=True)
    gen_tcn = gen_tcn.to(device)
    clf = clf.to(device)

    opt = optim.Adam(list(gen_tcn.parameters()) + list(clf.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Model training
    train_loss_list = []
    val_loss_list = []
    best_val_loss = np.inf
    best_val_epoch = 0

    for epoch in range(epochs + 1):
        gen_tcn.train()
        clf.train()
        train_loss, count = 0, 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()

            out = clf(gen_tcn(x))
            out = out[:, hist_len:].contiguous().view(-1, config.NUM_CHAR)
            y = y[:, hist_len:].contiguous().view(-1)

            loss = criterion(out, y)
            loss.backward()

            nn.utils.clip_grad_norm_(gen_tcn.parameters(), clip)
            nn.utils.clip_grad_norm_(clf.parameters(), clip)

            opt.step()
            train_loss += loss.item()
            count += 1

        train_loss_list.append(train_loss / count)

        # Model Validation
        gen_tcn.eval()
        clf.eval()
        val_loss, count = 0, 0

        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            out = clf(gen_tcn(x))
            out = out[:, hist_len:].contiguous().view(-1, config.NUM_CHAR)
            y = y[:, hist_len:].contiguous().view(-1)

            loss = criterion(out, y)
            val_loss += loss.item()
            count += 1

        val_loss_list.append(val_loss / count)

        if val_loss_list[-1] <= best_val_loss:
            best_val_loss = val_loss_list[-1]
            best_val_epoch = epoch
    
            # Save the model
            torch.save(gen_tcn.state_dict(), Path(res_path, f'Model_{model}_Gen-DNA-TCN.pt'))
            torch.save(clf.state_dict(), Path(res_path, f'Model_{model}_clf.pt'))

        print(f'Epoch: {epoch}, Train CE: {train_loss_list[-1]:.6f}, Best Val CE: {best_val_loss:.6f}')

    torch.save(
        {
        'train_loss': train_loss_list,
        'val_loss': val_loss_list,
        'best_val_epoch': best_val_epoch
        }, Path(res_path, 'res.pt')
    )




