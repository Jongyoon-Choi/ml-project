#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from models.MF.model import MF
from tqdm import tqdm


class MyTrainer:
    def __init__(self, device, num_users, num_items):
        self.device = device
        self.num_users = num_users
        self.num_items = num_items

    def train_with_hyper_param(self, train_data, hyper_param, verbose=False):
        # Hyperparameters
        epochs = hyper_param['epochs']
        batch_size = hyper_param['batch_size']
        learning_rate = hyper_param['learning_rate']
        k = hyper_param['k']
        
        # rating mean
        rating_mean = train_data.rating_mean
        
        # Loaders
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

        # MF 모델 초기화
        model = MF(self.num_users, self.num_items, k, rating_mean = rating_mean).to(self.device)
        
        # optimizer 설정
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        pbar = tqdm(range(epochs), leave=False, colour='green', desc='epoch')
        #pbar = range(epochs)
        for epoch in pbar:
            model.train()
            
            for user, item, rating in tqdm(train_loader, leave=False, colour='red', desc='batch'):
            #for user, item, rating in train_loader:
                # GPU 또는 CPU로 데이터 이동
                user, item, rating = user.to(self.device), item.to(self.device), rating.to(self.device)

                optimizer.zero_grad()
                
                loss = model(user, item, rating)
                rmse_train = torch.sqrt(loss)
                
                loss.backward()
                optimizer.step()
            
            if verbose:
                pbar.write('Epoch {:02}: {:.4} training RMSE'.format(epoch, rmse_train))
                print('Epoch {:02}: {:.4} training RMSE'.format(epoch, rmse_train))

        pbar.close()

        return model