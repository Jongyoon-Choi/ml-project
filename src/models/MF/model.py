#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn.init as init

class MF(torch.nn.Module):
    def __init__(self, n_users, n_items, k, rating_mean = None):
        super(MF, self).__init__()
        self.Q = torch.nn.Parameter(torch.zeros(n_users, k))
        self.P = torch.nn.Parameter(torch.zeros(n_items, k))
        self.rating_mean = rating_mean # mu
        self.bias_users = torch.nn.Parameter(torch.zeros(n_users))
        self.bias_items = torch.nn.Parameter(torch.zeros(n_items))
        
        # 초기화
        init.uniform_(self.Q, 0, 0.1)
        init.uniform_(self.P, 0, 0.1)
        init.uniform_(self.bias_users, 0, 0.1)
        init.uniform_(self.bias_items, 0, 0.1)
        
        # 손실 함수로 평균 제곱 오차(Mean Squared Error) 사용
        self.criterion = torch.nn.functional.mse_loss                                                             

    def forward(self, user, item, rating):
        
        prediction = self.predict(user, item)
        
        # 예측과 실제 평점 간의 평균 제곱 오차 계산
        loss = self.criterion(prediction, rating)
        
        return loss

    def predict(self, user, item):
        
        user-=1
        item-=1
        
         # 사용자 및 아이템에 대한 편향
        user_bias = self.bias_users[user]
        item_bias = self.bias_items[item]

        # 예측 평점 계산
        prediction = torch.sum(self.Q[user] * self.P[item], dim=1) + user_bias + item_bias
        
        # 만약 rating_mean이 초기화되었다면 평균을 더해줌
        if self.rating_mean is not None:
            prediction += self.rating_mean
        
        return prediction