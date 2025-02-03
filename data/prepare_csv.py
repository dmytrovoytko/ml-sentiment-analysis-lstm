import codecs
import pandas as pd
pd.set_option('display.width', 150) 
pd.set_option('display.max_colwidth', 100)

TEXT_COLUMN = "text"
TARGET = "sentiment" 

def convert2csv(files, out_file_name, test_size=0.2):
    dfs_train = []
    dfs_test = []

    for full_name in files:
        file_name = full_name.split('/')[-1]
        file_ext = file_name.split('.')[-1]
        if file_ext=='neg':
            label = 'Negative'
        elif file_ext=='pos':
            label = 'Positive'
        elif file_name.startswith('plot'):
            label = 'Objective'
        elif file_name.startswith('quote'):
            label = 'Subjective'
        else:
            print('! Unrecognized sentiment! Skipped.')
            continue

        with codecs.open(file_name, encoding="ISO-8859-1") as in_file:
            stripped = [line.strip() for line in in_file]
            print(file_name, len(stripped), label)
            df = pd.DataFrame({TEXT_COLUMN:stripped})
            df[TARGET] = label
            full_size = df.shape[0]
            print(full_size, df.head().reset_index()[['index', 'sentiment']])
            train_size = int(full_size*(1-test_size))
            dfs_train.append(df[:train_size])
            dfs_test.append(df[train_size:])

    # making final train and test files interleaved - like pos/neg rows mix
    df_train = pd.concat(dfs_train).sort_index(kind='stable')
    df_test = pd.concat(dfs_test).sort_index(kind='stable')
    print('==========')
    print('- train:', df_train.shape[0], df_train[TARGET].value_counts().to_string())
    print('- test:', df_test.shape[0], df_test[TARGET].value_counts().to_string())
    df_train.to_csv(out_file_name+'_train.csv', index=False)
    df_test.to_csv(out_file_name+'_test.csv', index=False)
    print(df_train.head(4))

# files = ['rt-polaritydata/rt-polarity.neg','rt-polaritydata/rt-polarity.pos']
# convert2csv(files, 'rt-polarity', test_size=0.2)

files = ['rotten_imdb/plot.tok.gt9.5000','rotten_imdb/quote.tok.gt9.5000']
convert2csv(files, 'subjectivity', test_size=0.2)
