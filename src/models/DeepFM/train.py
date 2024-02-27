#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from models.DeepFM.model import DeepFM
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import tqdm

class MyTrainer:
    def __init__(self, device, num_users, num_items, feature_sizes):
        self.device = device
        self.num_users = num_users
        self.num_items = num_items
        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes

    def train_and_test(self, train_data, test_data, hyper_param, verbose=False):
        # Hyperparameters
        epochs = hyper_param['epochs']
        batch_size = hyper_param['batch_size']
        embedding_dim = hyper_param['embedding_dim']
        hidden_dim = hyper_param['hidden_dim']
        learning_rate = hyper_param['learning_rate']

        # Loaders
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

        total_batches = len(train_loader)
        model = DeepFM(self.device, self.num_users, self.num_items, self.feature_sizes, embedding_dim, hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        pbar = tqdm(range(epochs), leave=False, colour='green', desc='epoch')

        for epoch in pbar:
            avg_loss = 0
            model.train()
            
            for user, item, features, rating in tqdm(train_loader, leave=False, colour='red', desc='batch'):
                # send data to a running device (GPU or CPU)
                user, item, features, rating = user.to(self.device), item.to(self.device), features.to(self.device), rating.to(self.device)

                optimizer.zero_grad()
                
                predictions = model(user, item, features).squeeze()
                loss = F.mse_loss(predictions, rating)
                rmse_train = torch.sqrt(loss)
                
                loss.backward()
                optimizer.step()

                avg_loss += loss / total_batches
        
            model.eval()
        
            all_ratings = []
            all_predictions = []

            with torch.no_grad():
                for user, item, features, rating in tqdm(test_loader, desc='평가 중', leave=False):
                    # GPU 또는 CPU로 데이터 이동
                    user, item, features, rating = user.to(self.device), item.to(self.device), features.to(self.device), rating.to(self.device)

                    # 모델 예측
                    output = model(user, item, features).squeeze()

                    # 결과 기록
                    all_ratings.extend(rating.cpu().numpy())
                    all_predictions.extend(output.cpu().numpy())

            # 평균 제곱근 오차 (RMSE) 계산
            rmse_test = np.sqrt(mean_squared_error(all_ratings, all_predictions))
            
            if verbose:
                pbar.write('Epoch {:02}: {:.4} training RMSE, {:.4} testing RMSE'.format(epoch, rmse_train, rmse_test))

        pbar.close()

        return model
    
    def predict(self, model, evaluation_data):
        model.eval()
        evaluation_loader = torch.utils.data.DataLoader(evaluation_data, batch_size=64, shuffle=False)
        all_predictions = []
        # 훈련된 모델을 사용하여 예측
        with torch.no_grad():
            for user, item, features, _ in tqdm(evaluation_loader, desc='예측 중', leave=False):
                # GPU 또는 CPU로 데이터 이동
                user, item, features= user.to(self.device), item.to(self.device), features.to(self.device)
                # 모델 예측
                output = model(user, item, features).squeeze()

                # 결과 기록
                all_predictions.extend(output.cpu().numpy())

        return all_predictions
