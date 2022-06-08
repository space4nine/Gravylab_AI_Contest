import pandas as pd
import numpy as np


class Preprocessing:

    def __init__(self) -> None:
        self.data_path = './data/'

        self.train_df = pd.read_json(self.data_path+'train.json')
        self.test_df = pd.read_json(self.data_path+'test.json')

    def json2df(self, df, mode='train'):
        content_lv = df['data'].apply(lambda x: pd.DataFrame(x))
        df_ = pd.DataFrame()
        for row  in content_lv:
            df_ = pd.concat((df_, row), axis=0)
        df_ = df_.reset_index(drop=True)

        def get_answer_text(x):
            if len(x) > 0:
                return x[0]['text']
            else:
                return ''

        def get_answer_start(x):
            if len(x) > 0:
                return x[0]['answer_start']
            else:
                return np.nan

        df_['paragraph_id'] = df_['paragraphs'].apply(lambda x: x['paragraph_id'])
        df_['context'] = df_['paragraphs'].apply(lambda x: x['context'])
        df_['question_id'] = df_['paragraphs'].apply(lambda x: x['qas'][0]['question_id'])
        df_['question'] = df_['paragraphs'].apply(lambda x: x['qas'][0]['question'])
        if mode == 'train':
            df_['answers'] = df_['paragraphs'].apply(lambda x: x['qas'][0]['answers'])
            df_['is_impossible'] = df_['paragraphs'].apply(lambda x: x['qas'][0]['is_impossible'])
            df_['answer_text'] = df_['answers'].apply(get_answer_text)
            df_['answer_start'] = df_['answers'].apply(get_answer_start)
            df_.drop(['paragraphs', 'answers'], axis=1, inplace=True)
        else:
            df_.drop(['paragraphs'], axis=1, inplace=True)
        return df_
    
    def get_rdb(self):
        train = self.json2df(self.train_df)
        test = self.json2df(self.test_df, mode='test')
        return train, test
    

 