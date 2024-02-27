#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch

class MyEvaluator:
    def __init__(self, device):
        self.device = device
        
    def evaluate(self, model, val_data):
        with torch.no_grad():
            model.eval()
            users = val_data.get_users().to(self.device)
            items = val_data.get_items().to(self.device)
            ratings = val_data.get_ratings().to(self.device)

            predictions = model.predict(users, items)
            rmse = torch.sqrt(torch.nn.functional.mse_loss(predictions, ratings.float()))

        return rmse.item()
