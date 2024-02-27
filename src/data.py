#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.iloc[:, 1:]
        self.num_users = len(self.data['User-ID'].unique())
        self.num_items = len(self.data['Book-ID'].unique())
        features = ['Book-Author', 'Year-Of-Publication', 'Publisher', 'Main_Title', 'Sub_Title', 'City', 'State', 'Country', 'Age_gb']
        self.feature_sizes = [len(self.data[col].unique()) for col in features]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # PyTorch 모델에 입력할 수 있도록 데이터 반환
        user = torch.tensor(self.data.iloc[idx]['User-ID'], dtype=torch.long)
        item = torch.tensor(self.data.iloc[idx]['Book-ID'], dtype=torch.long)
        features = torch.tensor(self.data.iloc[idx, ~self.data.columns.isin(['User-ID', 'Book-ID', 'Book-Rating'])].values, dtype=torch.long)
        rating = torch.tensor(self.data.iloc[idx]['Book-Rating'] if 'Book-Rating' in self.data.columns else 0, dtype=torch.float)

        return user, item, features, rating
    
class MovieLensDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.num_users = max(self.data['user id'])
        self.num_items = max(self.data['movie id'])
        self.rating_mean = self.data['rating'].mean() if 'rating' in self.data.columns else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # PyTorch 모델에 입력할 수 있도록 데이터 반환
        user = torch.tensor(self.data.iloc[idx]['user id'], dtype=torch.long)
        item = torch.tensor(self.data.iloc[idx]['movie id'], dtype=torch.long)
        rating = torch.tensor(self.data.iloc[idx]['rating'] if 'rating' in self.data.columns else 0, dtype=torch.float)

        return user, item, rating
    
    def get_users(self):
        return torch.tensor(self.data['user id'].values, dtype=torch.long)

    def get_items(self):
        return torch.tensor(self.data['movie id'].values, dtype=torch.long)

    def get_ratings(self):
        return torch.tensor(self.data['rating'].values, dtype=torch.float32)