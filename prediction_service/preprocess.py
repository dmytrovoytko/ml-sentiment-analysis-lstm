import warnings # supress warnings
warnings.filterwarnings('ignore')

import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from settings import DATA_DIR, MODEL_DIR, CLASSES_NUM, USE_ENCODER
from settings import DATA_FILE, TARGET, TEXT_COLUMN

# Display all columns
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


from settings import DEBUG  # isort:skip

DEBUG = True  # True # False # override global settings

import re
THRESHOLD = 70
REGX_RT = r"RT @[A-Za-z0-9$-_@.&+]+:"
REGX_USERNAME = r"@[A-Za-z0-9$-_@.&+]+"
REGX_URL = r"https?://[A-Za-z0-9./]+"
REGX_ASCII = r"[^\x00-\x7F]"
MIN_LENGTH = 10

def load_dataset(dataset=DATA_FILE, verbose=DEBUG):
    file_name = dataset['FILE_NAME']
    if file_name.lower().split('.')[-1]=='parquet':
        data = pd.read_parquet(DATA_DIR + file_name)
    else:
        try:
            data = pd.read_csv(DATA_DIR + file_name, encoding=dataset['ENCODING'])
        except Exception as e:
            print('!! Failed to read_csv', DATA_DIR + file_name, e)
            data = pd.read_csv(DATA_DIR + file_name, sep=';', encoding='utf-8')

    if verbose:
        print(f' Loaded {file_name}: {data.shape[0]} records.')
        data.info()
    return data


def print_missing_values_table(data, na_name=False):
    na_columns = [col for col in data.columns if data[col].isnull().sum() > 0]
    n_miss = data[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (data[na_columns].isnull().sum() / data.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

def preprocess_sentiment(text):
    pattern = re.search(r'(\w+)(\s)*(\((.*)%\))*', text)
    if not pattern:
        print('! error parsing', text)
        return text

    g1 = pattern.group(1) # sentiment
    g4 = pattern.group(4) # %% value
    if g4:
      g4 = '' if float(g4)>THRESHOLD else ' weak'
    else:
      g4 = ''
    return g1+g4

def decontract(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"let\'s", "let us", phrase)
    phrase = re.sub(r"t\'s", "t is", phrase) # it's that's
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    phrase = re.sub(r"1st", "first", phrase)
    phrase = re.sub(r"2nd", "second", phrase)
    phrase = re.sub(r"3rd", "third", phrase)

    return phrase

def preprocess_text(text):
    text = re.sub(REGX_RT, ' ', text)

    text = text.lower()

    # extra html line breaks, tags
    text = re.sub('<unk>', ' ', text)
    # text = re.sub('<br />', ' ', text)
    text = re.sub('\n', ' ', text)

    text = decontract(text)

    text = re.sub(REGX_USERNAME, ' ', text)
    text = re.sub(REGX_URL, ' ', text)
    text = re.sub(REGX_ASCII, ' ', text)
    text = re.sub('dlvr.it', ' ', text)
    text = re.sub('pic.twitter.com', ' ', text)
    text = re.sub('t.co', ' ', text)
    text = re.sub('youtube.com', ' ', text)
    text = re.sub('twitch.tv', ' ', text)
    text = re.sub('instagram.com', ' ', text)
    text = re.sub('google.com', ' ', text)
    text = re.sub('https:', ' ', text)

    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[,.:~_%<>\[\]\(\)\"|\s\-\s|\-\-]', ' ', text)  # Remove punctuation
    text = re.sub(r'\$\s', ' ', text)  # non ticker $
    text = re.sub(r'\s\$$', ' ', text)  # trailing $
    text = re.sub(' +', ' ', text) # multiple spaces
    text = re.sub(r'^\s+', '', text) # leading spaces
    text = re.sub(r'\s+$', '', text) # trailing spaces
    return text

def transform_df(df, verbose=DEBUG):
    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(preprocess_text)
    if verbose:
        print(df[TEXT_COLUMN].str.len().describe())
    # remove too short text rows
    df = df[df[TEXT_COLUMN].str.len()>=MIN_LENGTH]

    return df

def preprocess_df(data, verbose=DEBUG):
    print('\nPreprocessing data...')
    target = TARGET
    total_rows_number = data.shape[0]

    # normalizing column names - lowercase, no spaces
    data.columns = data.columns.str.lower().str.replace(" ", "_")
    columns = data.columns.to_list()

    # 1.1 drop useless columns
    useless_columns = [] # can add extra filtering
    for col in useless_columns:
        if col in columns:  # exists in df
            if verbose:
                print(f'Dropping column {col}. Total rows: {total_rows_number}, unique {col.upper()}s: {data[col].nunique()}')
            data.drop([col], axis=1, inplace=True)

    # 1.2 drop columns with nulls
    # inspect missing values
    nan_cols = data.columns[data.isnull().any()].to_list()
    if verbose and nan_cols:
        # list of columns with missing values and its percentage
        print(f'\nColumns with nulls:\n{nan_cols}')
        print_missing_values_table(data, na_name=True)

    if target in columns:
        data = data[~data[target].isnull()]
    data = data[~data[TEXT_COLUMN].isnull()]

    # drop duplicates
    data = data.drop_duplicates()

    # 1.3 transform text & sentiment values
    data = transform_df(data, verbose)
    if verbose and target in columns:
        print('\n by', data[target].value_counts().to_string())            

    # drop duplicates again after transformation
    data = data.drop_duplicates()

    # 2.1 inspect categorical columns
    categorical = data.dtypes[data.dtypes == 'object'].keys()
    if target in categorical:
        # some datasets have trailing spaces in TARGET column - stripping
        data[target] = data[target].str.strip()
        # specifics of some datasets, TARGET must be encoded to 0/1 BEFORE using OrdinalEncoder
        if CLASSES_NUM==2:
            # data.loc[data[target] == 'FALSE', target] = 0
            # data.loc[data[target] == 'TRUE', target] = 1
            data.loc[data[target] == 'Negative', target] = 0
            data.loc[data[target] == 'Positive', target] = 1
            data.loc[data[target] == 'Negative emotion', target] = 0
            data.loc[data[target] == 'Positive emotion', target] = 1
            data.loc[data[target] == 'Objective', target] = 0
            data.loc[data[target] == 'Subjective', target] = 1
            data = data[data[target].isin([0, 1])] # dropping the rest
        elif CLASSES_NUM==3:
            data.loc[data[target] == 'Negative', target] = 0
            data.loc[data[target] == 'Neutral', target] = 1
            data.loc[data[target] == 'Positive', target] = 2

            data.loc[data[target] == 'Negative emotion', target] = 0
            data.loc[data[target] == 'No emotion toward brand or product', target] = 1
            data.loc[data[target] == 'Positive emotion', target] = 2
            # some more - definitely positive/negative emotions 
            data.loc[data[target] == 'Bad', target] = 0
            data.loc[data[target] == 'Hate', target] = 0
            data.loc[data[target] == 'Happy', target] = 2
            data = data[data[target].isin([0, 1, 2])] # dropping the rest
        else:
            print('!! Error! CLASSES_NUM should be 2 or 3!')
            return data.head(0)

        data[target] = data[target].astype(int)
        # update categorical
        categorical = list(data.dtypes[data.dtypes == 'object'].keys())
        if verbose:
            print('\n by', data[target].value_counts().to_string())            
            print(target, 'encoded.')

    # 2.2 inspect boolean columns
    boolean = data.dtypes[data.dtypes == 'bool'].keys()
    if target in boolean:
        # specifics of some datasets, TARGET must be encoded to 0/1 BEFORE using OrdinalEncoder
        data.loc[data[target] == False, target] = 0
        data.loc[data[target] == True, target] = 1
        data = data[data[target].isin([0, 1])] # dropping the rest
        data[target] = data[target].astype(int)
        if verbose:
            print('\n by', data[target].value_counts().to_string())            
            print(target, 'encoded.')

    print(
        f'\nFinal number of records: {data.shape[0]} / {total_rows_number} =',
        f'{data.shape[0]/total_rows_number*100:05.2f}%\n',
    )
    return data


def enc_save(enc, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(enc, f)


def enc_load(file_name):
    with open(file_name, 'rb') as f:
        enc = pickle.load(f)
        return enc
    return OrdinalEncoder()


def preprocess_data(df, selected_columns, ord_enc=None, fit_enc=False, verbose=DEBUG):
    df = df[selected_columns]
    # fix missing values, remove outliers
    df = preprocess_df(df, verbose)

    # encode categorical
    categorical_features = df.select_dtypes(exclude=[np.number]).columns
    if len(categorical_features):
        if USE_ENCODER:
            if verbose:
                print('OrdinalEncoder categorical_features:', list(categorical_features))
            # import ordinal encoder from sklearn
            # ord_enc = OrdinalEncoder()
            if fit_enc:
                # Fit and Transform the data
                df[categorical_features] = ord_enc.fit_transform(df[categorical_features])
                enc_save(ord_enc, f'{MODEL_DIR}encoder.pkl')
                if verbose:
                    print(' OrdinalEncoder categories:', ord_enc.categories_)
            else:
                # Only Transform the data (using pretrained encoder)
                df[categorical_features] = ord_enc.transform(df[categorical_features])

    return df


def load_data(dataset=DATA_FILE, verbose=DEBUG):
    df = load_dataset(dataset=dataset, verbose=verbose)
    # df.columns = df.columns.str.replace('created_at', 'timestamp')
    df.columns = df.columns.str.replace(dataset['SOURCE_TEXT_COLUMN'], TEXT_COLUMN)
    df.columns = df.columns.str.replace(dataset['SOURCE_TARGET'], TARGET)
    ord_enc = OrdinalEncoder()

    selected_columns = [TEXT_COLUMN, TARGET] 

    df = preprocess_data(df, selected_columns, ord_enc, fit_enc=USE_ENCODER, verbose=verbose)
    return df


if __name__ == '__main__':
    # quick test
    df = load_data(DATA_FILE)
    # df.head(5000).to_csv(DATA_DIR + 'clean.csv', index=False)
