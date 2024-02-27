#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import fire
import torch
import pandas as pd
from pathlib import Path
from data import MovieLensDataset
from models.MF.train import MyTrainer
from models.MF.eval import MyEvaluator
from loguru import logger
from utils import set_random_seed
from sklearn.model_selection import train_test_split


def run_mymodel(device, train_dataset, val_dataset, test_dataset, hyper_param, save_predictions):
    # 모델 훈련
    trainer = MyTrainer(device=device,
                        num_users=max(train_dataset.num_users, val_dataset.num_users),
                        num_items=max(train_dataset.num_items, val_dataset.num_items))
    model = trainer.train_with_hyper_param(train_data=train_dataset,
                                          hyper_param=hyper_param,
                                          verbose=True)
    # 모델 평가
    evaluator = MyEvaluator(device=device)
    rmse = evaluator.evaluate(model, val_dataset)
    
    if save_predictions:
        # 예측 결과 저장
        users = test_dataset.get_users().to(device)
        items = test_dataset.get_items().to(device)
        predictions = model.predict(users, items).detach().cpu().numpy()

        # 데이터프레임 생성
        result_df = pd.DataFrame({'id': range(1, len(predictions)+1), 'rating': predictions})

        # CSV 파일로 저장
        output_filename = f'/home/chg9535/ml-project/datasets/MovieLens/MF_epoch{hyper_param["epochs"]}_k{hyper_param["k"]}_lr{hyper_param["learning_rate"]}.csv'
        result_df.to_csv(output_filename, index=False)
    
    return rmse

def main(model='MF',
         seed=42,
         batch_size=64,
         epochs=9,
         k=11, 
         learning_rate=0.001, 
         save_predictions=True):
    """
    :param model: 훈련 및 테스트할 모델의 이름
    :param seed: 랜덤 시드 (만약 -1이면 기본 시드 사용)
    :param batch_size: 배치 크기
    :param epochs: 훈련 에포크 수
    :param k: 행렬 분해 모델의 잠재 요인 수
    :param learning_rate: 학습률
    :param save_predictions: 예측 결과를 저장할지 여부
    """
    
    # Step 0. 초기화
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_random_seed(seed=seed, device=device)
    param = dict()
    param['model'] = model
    param['seed'] = seed
    param['device'] = device
    
    logger.info("The main procedure has started with the following parameters:")
    logger.info(f"Model: {param['model']}")
    logger.info(f"Seed: {param['seed']}")
    logger.info(f"Device: {param['device']}")

    # Step 1. 데이터셋 로드
    folder_path = Path(__file__).parents[1].absolute().joinpath("datasets", "MovieLens")
    train_path = folder_path.joinpath("train.csv")
    train_data = pd.read_csv(train_path, encoding='utf-8')
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=seed)
    train_dataset = MovieLensDataset(train_data)
    val_dataset = MovieLensDataset(val_data)

    logger.info("The datasets are loaded where their statistics are as follows:")
    logger.info(f"training instances: {len(train_dataset)}")
    logger.info(f"validation instances: {len(val_dataset)}")
    
    if save_predictions:
        test_path = folder_path.joinpath("test.csv")
        test_data = pd.read_csv(test_path, encoding='utf-8')
        test_dataset = MovieLensDataset(test_data)
        logger.info(f"test instances: {len(test_dataset)}")
    else:
        test_dataset = None

    # 하이퍼 파라미터
    hyper_param = dict()
    hyper_param['batch_size'] = batch_size
    hyper_param['epochs'] = epochs
    hyper_param['k'] = k
    hyper_param['learning_rate'] = learning_rate
    
    logger.info("Training the model has begun with the following hyperparameters:")
    logger.info(f"Batch Size: {hyper_param['batch_size']}")
    logger.info(f"Epochs: {hyper_param['epochs']}")
    logger.info(f"K: {hyper_param['k']}")
    logger.info(f"Learning Rate: {hyper_param['learning_rate']}")
    
    # Step 2. 지정된 모델 실행 (train and evaluate)
    if model == 'MF':
        rmse = run_mymodel(device=device,
                            train_dataset = train_dataset,
                            val_dataset = val_dataset, 
                            test_dataset = test_dataset, 
                            hyper_param = hyper_param, 
                            save_predictions=save_predictions)
        
    else:
        logger.error("The given \"{}\" is not supported...".format(model))
        return

    # Step 3. 최종 결과 report
    logger.info("The model has been trained. The test rmse is {:.4}.".format(rmse))


if __name__ == "__main__":
    sys.exit(fire.Fire(main))
