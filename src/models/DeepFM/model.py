#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepFM(nn.Module):
    def __init__(self, device, num_users, num_items, feature_sizes, embedding_dim, hidden_dim):
        super(DeepFM, self).__init__()
        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes
        self.device = device

        # 임베딩 레이어
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.feature_embeddings = nn.ModuleList([nn.Embedding(feature_size, embedding_dim).to(self.device) for feature_size in self.feature_sizes])

        # Deep component에 대한 완전 연결 레이어
        self.fc1 = nn.Linear(embedding_dim * (2 + self.field_size), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, user, item, feature):
        # 임베딩
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        feature_emb_list = [emb(feature[:, i]) for i, emb in enumerate(self.feature_embeddings)]

        # FM component
        fm_terms = torch.cat([user_emb, item_emb, *feature_emb_list], dim=1)
        fm_interactions = torch.sum(fm_terms, dim=1).unsqueeze(1)

        # Deep component
        deep_input = torch.cat([user_emb, item_emb, *feature_emb_list], dim=1)
        deep_out = F.relu(self.fc1(deep_input))
        deep_out = self.fc2(deep_out)

        # 최종 예측
        prediction = torch.sigmoid(fm_interactions + deep_out)
        prediction = prediction * 10.0

        return prediction
    
    
