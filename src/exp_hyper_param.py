#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import pandas as pd
from pathlib import Path
from utils import set_random_seed
from loguru import logger
from models.MF.train import MyTrainer
from models.MF.eval import MyEvaluator
from data import MovieLensDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# check the effect of the learning rate
def main():
    # Step 0. Initialization
    logger.info("Start the experiment for checking the effect of the learning rate.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42
    set_random_seed(seed=seed, device=device)

    output_dir = Path(__file__).parents[1].absolute().joinpath("plots")
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir.joinpath("exp_hyper_param.tsv")

    param = dict()
    param['seed'] = seed
    param['device'] = device
    param['output_path'] = output_path
    
    logger.info("The main procedure has started with the following parameters:")
    logger.info(f"Model: {'MF'}")
    logger.info(f"Seed: {param['seed']}")
    logger.info(f"Device: {param['device']}")

    # Step 1. Load datasets
    folder_path = Path(__file__).parents[1].absolute().joinpath("datasets", "MovieLens")
    train_path = folder_path.joinpath("train.csv")
    train_data = pd.read_csv(train_path, encoding='utf-8')
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=seed)
    train_dataset = MovieLensDataset(train_data)
    val_dataset = MovieLensDataset(val_data)
    
    logger.info("The datasets are loaded where their statistics are as follows:")
    logger.info(f"training instances: {len(train_dataset)}")
    logger.info(f"validation instances: {len(val_dataset)}")

    hyper_param = dict()
    hyper_param['batch_size'] = 64
    hyper_param['epochs'] = 10
    hyper_param['k'] = 10
    
    logger.info("Training the model has begun with the following hyperparameters:")
    logger.info(f"Batch Size: {hyper_param['batch_size']}")
    logger.info(f"Epochs: {hyper_param['epochs']}")
    logger.info(f"K: {hyper_param['k']}")

    # Step 2. Do experiment
    trainer = MyTrainer(device=device,
                        num_users=max(train_dataset.num_users, val_dataset.num_users),
                        num_items=max(train_dataset.num_items, val_dataset.num_items))
    evaluator = MyEvaluator(device=device)

    learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    with open(output_path, "w") as output_file:
        pbar = tqdm(learning_rates, leave=False, colour='blue', desc='lrate')
        for learning_rate in pbar:
            hyper_param['learning_rate'] = learning_rate
            model = trainer.train_with_hyper_param(train_data=train_dataset,
                                          hyper_param=hyper_param,
                                          verbose=False)
            rmse = evaluator.evaluate(model, val_dataset)
            pbar.write("learning_rate: {:.4f}\tval_rmse: {:.4f}".format(learning_rate, rmse))
            output_file.write("{}\t{}\n".format(learning_rate, rmse))
        pbar.close()

if __name__ == "__main__":
    main()
