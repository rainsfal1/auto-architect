# src/nas/enas_controller.py

import logging
import torch
import torch.optim as optim
from ..config import DEVICE

logger = logging.getLogger('nni_enas_ner')

def train_eval(model, train_loader, val_loader, max_epochs=10):
    model.to(DEVICE)

    optimizer_name = model.config['optimizer']
    lr = model.config['lr']
    weight_decay = model.config['weight_decay']

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'radam':
        optimizer = optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'nadam':
        optimizer = optim.NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    scheduler_name = model.config['lr_scheduler']
    if scheduler_name == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    elif scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    elif scheduler_name == 'warmup':
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    elif scheduler_name == 'linear':
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=max_epochs)
    elif scheduler_name == 'constant':
        scheduler = optim.lr_scheduler.ConstantLR(optimizer)
    else:
        scheduler = None

    best_val_loss = float('inf')
    early_stopping_patience = 5
    no_improve_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            word_ids, char_ids, label_ids, attention_mask = [t.to(DEVICE) for t in batch.values()]
            optimizer.zero_grad()
            loss = model.loss(word_ids, char_ids, label_ids, attention_mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                word_ids, char_ids, label_ids, attention_mask = [t.to(DEVICE) for t in batch.values()]
                loss = model.loss(word_ids, char_ids, label_ids, attention_mask)
                val_loss += loss.item()

        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        logger.info(f'Epoch {epoch+1}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= early_stopping_patience:
            logger.info(f'Early stopping triggered after {epoch+1} epochs')
            break

    return best_val_loss