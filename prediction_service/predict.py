import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except:
    pass

from preprocess import enc_load, preprocess_data, preprocess_text
from settings import DEFAULT_CLASSIFIER, DATA_DIR, MODEL_DIR, MODEL_NAME, MAX_LEN, USE_ENCODER
from settings import TEST_FILE, TARGET, TEXT_COLUMN

from settings import DEBUG  # isort:skip

DEBUG = True # True # False # override global settings


class Predictor():
    def __init__(self, classifier=DEFAULT_CLASSIFIER, model_dir=MODEL_DIR, verbose=DEBUG):
        self.classifier = classifier
        if verbose:
            print(f'\nLoading predicting model "{classifier}" from {model_dir}...')
            # print(' model.get_config:', model.get_config())
        if classifier=='LSTM':
            from keras.saving import load_model
            self.model = load_model(model_dir+MODEL_NAME)
            self.tokenizer = enc_load(model_dir+'tokenizer.pkl')
            self.labelencoder = enc_load(model_dir+'labelencoder.pkl')
            self.onehotencoder = enc_load(model_dir+'onehotencoder.pkl')
            if verbose and DEBUG:
                print(self.model.summary())
        else:
            model_name = f'{classifier}.pkl'
            self.model = pickle.load(open(f'{model_dir}{model_name}', 'rb'))
            if verbose and DEBUG:
                print(' model.get_params:', self.model.get_params())
            self.count_vect = enc_load(model_dir+'CountVectorizer.pkl')
            self.transformer = enc_load(model_dir+'TfidfTransformer.pkl')

    def predict_text(self, text, verbose=DEBUG):
        text = preprocess_text(text)
        if self.classifier=='LSTM':
            # Pad the test sequences
            text_sequences = self.tokenizer.texts_to_sequences([text])
            text_padded = pad_sequences(text_sequences, padding='post', truncating='post', maxlen=MAX_LEN)
            text_features_list = [text_padded]
            _pred = (np.asarray(self.model.predict(text_features_list))).round()
            # pred would be [0. 1.] or [1. 0.] 
            pred = np.argmax(_pred) # decoding onehotencoder -> class number 
        else:
            text_counts = self.count_vect.transform([text])
            text_tfidf = self.transformer.transform(text_counts)
            pred = self.model.predict(text_tfidf)[0]
        if verbose:
            print(f"\nPrediction:\n text: {text} -> {pred}")
        return pred

    def encode_labels(self, y):
        if self.classifier=='LSTM':
            y_label = self.labelencoder.transform(y)
            y_test = self.onehotencoder.transform(y_label.reshape(-1, 1)).toarray()
            return y_test
        else:
            return y
    
    def evaluate_prediction(self, texts, labels, verbose=DEBUG):
        y_pred = []
        i = 0
        for text in texts:
            _pred = self.predict_text(text, False if verbose=='difference' else verbose)
            y_pred.append(_pred)
            if verbose=='difference' and labels[i]!=_pred:
                print(f' ! {text} // label != pred: {labels[i]} != {_pred}')
            i+=1

        if verbose and DEBUG:
            # y_test = self.encode_labels(labels)
            print('\nevaluate_prediction (sample 10 records):')
            print('labels:', list(labels[:10]))
            print('y_pred:', y_pred[:10])
            # print('y_test:', y_test[:10])

        accuracy, f1, recall, precision, roc_auc = evaluate_results(self.classifier, y_pred, labels, verbose)
        # accuracy, f1, recall, precision = evaluate_results(self.classifier, y_pred, y_test, verbose)
        return accuracy, f1, recall, precision, roc_auc

    def predict_df(self, df, verbose=DEBUG):
        print(f'\nPredicting using model {self.classifier}')

        selected_columns = [TEXT_COLUMN, TARGET] 
        test_data = preprocess_data(df, selected_columns, verbose=verbose)

        cols = test_data.columns.to_list()
        X_test = pd.DataFrame(test_data, columns=cols)
        if TARGET in cols:
            y_test = list(X_test.pop(TARGET))
        else:
            y_test = []

        try:
            y_pred = []
            for i in range(0, len(X_test)):
                text = X_test[TEXT_COLUMN].values[i]
                pred = self.predict_text(text, verbose)
                y_pred.append(pred)
                if verbose and y_test:
                    print(f" Comparing: pred: {pred} / test: {y_test[i]}")
            return y_pred
        except Exception as e:
            print(f'!!! Exception while predicting {TARGET}:', e)
            return []


def evaluate_results(classifier, y_test, y_predict, verbose=DEBUG):
    print(f'\n================\n{classifier}')
    # if verbose:
    #     print(classification_report(y_test, y_predict, digits=3))
    #     print(confusion_matrix(y_test, y_predict))
    # print(f' balanced accuracy: {balanced_accuracy_score(y_test, y_predict):05.3f}')
    accuracy = accuracy_score(y_test, y_predict)
    f1 = f1_score(y_test, y_predict) #, average="micro")
    recall = recall_score(y_test, y_predict) #, average="micro")
    precision = precision_score(y_test, y_predict) #, average="micro")
    roc_auc = roc_auc_score(y_test, y_predict) #, average="micro")
    print(f' accuracy: {accuracy:05.3f}')
    print(f' f1 score: {f1:05.3f}')
    print(f' recall score: {recall:05.3f}')
    print(f' precision score: {precision:05.3f}')
    print(f' roc auc score: {roc_auc:05.3f}')
    return accuracy, f1, recall, precision, roc_auc

def predict_test(df, classifier_name):
    if classifier_name=='LSTM':
        from keras.saving import load_model
        classifier = load_model(MODEL_DIR+MODEL_NAME)
        tokenizer = enc_load(MODEL_DIR+'tokenizer.pkl')
        labelencoder = enc_load(MODEL_DIR+'labelencoder.pkl')
        onehotencoder = enc_load(MODEL_DIR+'onehotencoder.pkl')

        pad_type = 'post'
        trunc_type = 'post'

        maxlen = MAX_LEN

        # Pad the test sequences
        test_sequences = tokenizer.texts_to_sequences(df[TEXT_COLUMN].values)
        test_padded = pad_sequences(test_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)
        # print(df1.tail(1), test_padded[-1])

        test_features_list = [test_padded]

        test_predict = (np.asarray(classifier.predict(test_features_list))).round()

        y_label = labelencoder.transform(df[TARGET].values)
        y_test = onehotencoder.transform(y_label.reshape(-1, 1)).toarray()
        # print(f'{y_test=} {test_predict=}')
    else:
        import pandas as pd
        model_name = f'{classifier_name}.pkl'
        classifier = pickle.load(open(f'{MODEL_DIR}{model_name}', 'rb'))
        if DEBUG:
            print(' model.get_params:', classifier.get_params())
        count_vect = enc_load(MODEL_DIR+'CountVectorizer.pkl')
        transformer = enc_load(MODEL_DIR+'TfidfTransformer.pkl')
        cols = df.columns.to_list()
        X_test = pd.DataFrame(df, columns=cols)
        if TARGET in cols:
            y_test = X_test.pop(TARGET)
        else:
            y_test = pd.Series()

        _pred = []
        _y_test = list(y_test)
        for i in range(0,len(X_test)):
            x_test_counts = count_vect.transform([X_test[TEXT_COLUMN].values[i]])
            x_test_tfidf = transformer.transform(x_test_counts)
            # _pred[i] = model.predict(x_test_tfidf)
            _pred.append(classifier.predict(x_test_tfidf))
            print(X_test[TEXT_COLUMN].values[i], _y_test[i], _pred[-1])
        # y_pred = pd.DataFrame(_pred, columns=[TARGET])
        test_predict = _pred

    _test_f1 = f1_score(y_test, test_predict, average='micro')
    _test_recall = recall_score(y_test, test_predict, average='micro')
    _test_precision = precision_score(y_test, test_predict, average='micro')
    print(" test — test_f1: %f — test_precision: %f — test_recall %f" %(_test_f1, _test_precision, _test_recall))        
    return test_predict

def predict_df_Keras(test_data, classifier=DEFAULT_CLASSIFIER, model_dir=MODEL_DIR, verbose=DEBUG):
    try:
        from keras.saving import load_model
        model = load_model(MODEL_DIR+MODEL_NAME)
        tokenizer = enc_load(MODEL_DIR+'tokenizer.pkl')
        labelencoder = enc_load(MODEL_DIR+'labelencoder.pkl')
        onehotencoder = enc_load(MODEL_DIR+'onehotencoder.pkl')
        if DEBUG:
            # print(' model.get_config:', model.get_config())
            print(model.summary())
    except Exception as e:
        print('!!! Exception while loading model:', e)
        return pd.Series()

    cols = test_data.columns.to_list()
    X_test = pd.DataFrame(test_data, columns=cols)
    if TARGET in cols:
        _y_test = X_test.pop(TARGET)
        y_label = labelencoder.transform(_y_test.values)
        y_test = onehotencoder.transform(y_label.reshape(-1, 1)).toarray()
    else:
        y_test = []

    try:
        # Pad the test sequences
        test_sequences = tokenizer.texts_to_sequences(df[TEXT_COLUMN].values)
        test_padded = pad_sequences(test_sequences, padding='post', truncating='post', maxlen=MAX_LEN)

        test_features_list = [test_padded]

        y_pred = (np.asarray(model.predict(test_features_list))).round()
        if verbose:
            print(f"\nPrediction:\ny_pred: {list(y_pred)[:10]}")
            if len(y_test):
                print(f"y_test: {list(y_test)[:10]}")
        return y_pred, y_test
    except Exception as e:
        print(f'!!! Exception while predicting {TARGET}:', e)
        return pd.Series()


def predict_df(df, classifier=DEFAULT_CLASSIFIER, model_dir=MODEL_DIR, verbose=DEBUG):
    print(f'\nPredicting using model {classifier} {model_dir}')

    selected_columns = [TEXT_COLUMN, TARGET] 
    try:
        test_data = preprocess_data(df, selected_columns)
    except:
        test_data = df

    if classifier in ['LSTM']:
        return predict_df_Keras(df, classifier, model_dir, verbose)

    # the rest is for sklearn models

    cols = test_data.columns.to_list()
    X_test = pd.DataFrame(test_data, columns=cols)
    if TARGET in cols:
        y_test = X_test.pop(TARGET)
    else:
        y_test = pd.Series()

    model_name = f'{classifier}.pkl'
    try:
        model = pickle.load(open(f'{MODEL_DIR}{model_name}', 'rb'))
        if verbose and DEBUG:
            print(' model.get_params:', model.get_params())
        count_vect = enc_load(MODEL_DIR+'CountVectorizer.pkl')
        transformer = enc_load(MODEL_DIR+'TfidfTransformer.pkl')
    except Exception as e:
        print('!!! Exception while loading model:', e)
        return pd.Series()

    try:
        _pred = []
        for i in range(0,len(X_test)):
            # print(X_test[TEXT_COLUMN].values[i])
            x_test_counts = count_vect.transform([X_test[TEXT_COLUMN].values[i]])
            x_test_tfidf = transformer.transform(x_test_counts)
            # _pred[i] = model.predict(x_test_tfidf)
            _pred.append(model.predict(x_test_tfidf))
        # y_pred = pd.DataFrame(_pred, columns=[TARGET])
        y_pred = _pred
        if verbose and DEBUG:
            print(y_pred)
        # print('\n\nPrediction debug:', X_test.shape, X_train_counts.shape, X_train_tfidf.shape, y_pred.shape)
        if verbose:
            print(f"\nPrediction:\ny_pred: {list(y_pred)[:10]}")
            if len(y_test):
                print(f"y_test: {list(y_test)[:10]}")
        return y_pred
    except Exception as e:
        print(f'!!! Exception while predicting {TARGET}:', e)
        return pd.Series()


def predict_dict(object, classifier=DEFAULT_CLASSIFIER, verbose=False):
    # transform dict to create df from it
    dct = {k: [v] for k, v in object.items()}
    df = pd.DataFrame(dct)
    selected_columns = [TEXT_COLUMN] # , TARGET 
    df = preprocess_data(df, selected_columns)
    y_pred = predict_df(df, MODEL_DIR, verbose=verbose)
    result = {
        # str(TARGET).lower(): int(list(y_pred)[0])
        str(TARGET).lower(): int(np.argmax(y_pred)[0]) # decoding onehotencoder
    }  # explicit int() is reqiered for serialization
    if DEBUG:
        print('result:', result)

    return result

# TODO DECODE to 'Positive' 'Negative' for app?

if __name__ == '__main__':
    # quick tests
    print('\nTesting predict_df...')
    # batch testing with 10 records 
    from preprocess import load_data

    classifier = DEFAULT_CLASSIFIER
    test_file = TEST_FILE
    df = load_data(test_file).tail(200)

    data = df.to_dict('records')
    print(data[:10])
    texts = [d[TEXT_COLUMN] for d in data]
    labels = [d[TARGET] for d in data]
    print('labels:', labels[:10])
    # exit()
    y_pred = predict_test(df, DEFAULT_CLASSIFIER)
    print('y_pred:', y_pred[:10])
    predictor = Predictor(classifier=DEFAULT_CLASSIFIER, model_dir=MODEL_DIR, verbose=False)

    if classifier=='LSTM':
        y_test = predictor.encode_labels(labels)
        print('y_test:', y_test[:10])
        print('y_test argmax:', np.argmax(y_test[:10], axis=1))
    
    accuracy, f1, recall, precision, roc_auc = predictor.evaluate_prediction(texts, labels, verbose='difference')

