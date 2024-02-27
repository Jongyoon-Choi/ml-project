#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import re 
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
import torch


def set_random_seed(seed, device):

    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def preprocessing_data(df):
    
    # 데이터 정리
    df.loc[:, 'Book-Title'] = df['Book-Title'].apply(lambda x: re.sub(r'[^0-9a-zA-Z:,]',  ' ', str(x)))
    df.loc[:, 'Main_Title'] = df['Book-Title'].apply(lambda x: x.split('  ')[0])
    df.loc[:, 'Sub_Title'] = df['Book-Title'].apply(lambda x: ''.join(x.split('  ')[1:]) if x.split('  ')[1:] else 'No_SUB')
    
    df.loc[:, 'Location'] = df['Location'].apply(lambda x: re.sub(r'[^0-9a-zA-Z:,]',  ' ', str(x)))
    
    df.loc[:, 'City'] = df['Location'].apply(lambda x: (x.split(',')[0]).lstrip().lower())
    df.loc[:, 'State'] = df['Location'].apply(lambda x: (x.split(',')[1]).lstrip().lower())
    df.loc[:, 'Country'] = df['Location'].apply(lambda x: (x.split(',')[2]).lstrip().lower())

    # preprocessing Location
    # NaN, N/A, etc.. Change 'unknown'
    # Only using Train Data	#
    
    # 최빈값을 사용하기 위해 새로운 데이터 프레임 생성(pd.Series.mode를 이용하면 같은 count수의 값을 list로 묶어서 정확하지 않음)
    new_state = df.groupby(['City'])['State'].value_counts().to_frame().rename(columns = {'State' : 'count'}).reset_index()
    new_state = new_state[(~new_state['City'].isna())&(~new_state['State'].isna())&(new_state['count']!=1)]
    new_state = new_state.sort_values(by=['City', 'count'], ascending=[True, False]).drop_duplicates(subset='City', keep='first')
    new_state = new_state.rename(columns = {'State' : 'N_State'}) 
    new_state = new_state.drop(columns = ['count'])
    
    new_country = df.groupby(['State'])['Country'].value_counts().to_frame().rename(columns = {'Country' : 'count'}).reset_index()
    new_country = new_country[(~new_country['State'].isna())&(~new_country['Country'].isna())&(new_country['count']!=1)]
    new_country = new_country.sort_values(by=['State', 'count'], ascending=[True, False]).drop_duplicates(subset='State', keep='first')
    new_country = new_country.rename(columns = {'Country' : 'N_Country'}) 
    new_country = new_country.drop(columns = ['count'])
    
    df = df.merge(new_country, on='State', how='left')
    df = df.merge(new_state, on='City', how='left')
    
    df.loc[:, 'Country'] = np.where((df['Country'] == '')|(df['Country'].astype(str) == 'nan'), df['N_Country'], df['Country'])
    df.loc[:, 'State'] = np.where((df['State'] == '')|(df['State'].astype(str) == 'nan'), df['N_State'], df['State'])
    
    # 채워지지 않은 값은 Unknown 처리
    df[['Country', 'State', 'City']] = df[['Country', 'State', 'City']].fillna(value= 'Unknown')
    df = df.drop(columns = ['N_Country', 'N_State'])
    df = df.drop(columns = ['Book-Title', 'Location'])

    return df

def feature_engineering(df):
    labels = ['0-3','3-6','6-8','8-12','12-18','18-25','25-34','35-44','45-54','55-64','65-74','75+']
    bins = [0, 3, 6, 8, 12, 18, 25, 34, 44, 54, 64, 74, 250]
    
    # Age 이상치 처리
    df.loc[(df['Age'] > 90) | (df['Age'] < 3), 'Age'] = np.nan
    
    # 평균값으로 대체
    df.loc[:, 'Age'] = df['Age'].fillna(df['Age'].mean())
    df.loc[:, 'Age'] = df['Age'].astype(np.int32)

    
    df.loc[:, 'Age_gb'] = pd.cut(df.Age, bins, labels = labels,include_lowest = True)
    df = df.drop(columns = ['Age'])
    
    return df
def extract_numbers(df):
    # 문자열에서 숫자 부분 추출
    df.loc[:, 'User-ID'] = [int(re.sub(r'[^0-9]', '',i)) for i in df['User-ID']]
    df.loc[:, 'Book-ID'] = [int(re.sub(r'[^0-9]', '',i)) for i in df['Book-ID']]
    #df['ID'] = [int(re.sub(r'[^0-9]', '',i)) for i in df['ID']]
    return df

def label_encoding(train_data, test_data, evaluation_data):
    columns = ['Book-Author', 'Year-Of-Publication', 'Publisher', 'Main_Title', 'Sub_Title', 'City', 'State', 'Country', 'Age_gb']
    for col in columns:
        # LabelEncoder 객체 생성
        le = LabelEncoder()
        
        # train_data, test_data, evaluation_data의 해당 열을 합친 후 LabelEncoder로 변환
        all_data = np.concatenate([train_data[col], test_data[col], evaluation_data[col]])
        le.fit(all_data)
        
        # 변환된 값을 각 데이터프레임에 매핑
        train_data[col] = le.transform(train_data[col])
        test_data[col] = le.transform(test_data[col])
        evaluation_data[col] = le.transform(evaluation_data[col])

    return train_data, test_data, evaluation_data
