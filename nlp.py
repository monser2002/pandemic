import re
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
import pymorphy2


class TextModel:
    def __init__(self, df):
        self.corpus = self.parallelize_df(df, self.transform_df)
        self.df_idf = self.parallelize_df(self.corpus, self.transform_df_idf)

    @staticmethod
    def get_stopwords():
        stopwords = ['и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так',
                     'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было',
                     'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг',
                     'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж',
                     'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть',
                     'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего',
                     'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого',
                     'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее',
                     'сейчас',
                     'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой',
                     'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая',
                     'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда',
                     'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между',
                     'кем', 'кому']
        return stopwords

    @staticmethod
    def word_tokenize(text):
        pattern = r'[a-zA-Zа-яА-Я]+'
        return re.findall(pattern, text)

    @staticmethod
    def parallelize_df(df, func):
        n_cores = cpu_count() - 1
        df_split = np.array_split(df, n_cores)
        pool = Pool(n_cores)
        df = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()
        return df

    def transform_df(self, df):
        if df.columns.shape[0] == 2:
            for ind, row in df.iterrows():
                df.at[ind, 'position'] = self.lemmatize_sentence(row.position)
        else:
            for ind, row in df.iterrows():
                df.at[ind, 'position'] = self.lemmatize_sentence(row.position)
                df.at[ind, 'responsibilities'] = self.lemmatize_sentence(row.responsibilities)
        return df

    def lemmatize_sentence(self, sentence):
        tokens = self.word_tokenize(sentence)
        without_stopwords = set([word.lower() for word in tokens if word.lower() not in self.get_stopwords()])
        morph = pymorphy2.MorphAnalyzer()
        lemmas = set()
        for token in without_stopwords:
            lemma = morph.parse(token)[0].normal_form
            if morph.parse(lemma)[0].tag.POS == 'NOUN' and len(lemma) > 2:
                lemmas.add(lemma)
        return list(lemmas)

    def transform_df_idf(self, df):
        for ind, row in df.iterrows():
            df.at[ind, 'responsibilities'] = self.get_idf(row.responsibilities)
        return df

    def get_idf(self, corpus):
        idf_over_words = []
        for word in corpus:
            count = 0
            for doc_inner in self.corpus.responsibilities:
                if word in doc_inner:
                    count += 1.0
            idf_over_words.append(np.log(len(corpus) / count))
        return idf_over_words


if __name__ == '__main__':
    df_train = pd.read_csv('train.csv', sep=';', dtype='unicode')
    df_test = pd.read_csv('test.csv', sep=';', dtype='unicode')
    df_edu = pd.read_csv('education.csv', sep=';', dtype='unicode')
    df_emp = pd.read_csv('employements.csv', sep=';', dtype='unicode')

    df_base = df_emp.loc[:, ['id', 'position', 'responsibilities']].dropna()
    model = TextModel(df_base)
    model.df_idf.to_csv('df_lemmatized.csv')
