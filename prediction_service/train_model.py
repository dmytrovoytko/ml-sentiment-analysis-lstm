import warnings # supress warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier


import numpy as np
from numpy import array, asarray, zeros

# to suppress TF warnings
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, Bidirectional, Dense
    # , Reshape, MaxPool2D, Concatenate, Flatten, Dropout, Conv1D, MaxPool1D
    from tensorflow.keras.callbacks import EarlyStopping #, ModelCheckpoint, TensorBoard
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import LSTM
    from keras import regularizers

    from kerastuner.tuners import Hyperband #, RandomSearch
    from kerastuner.engine.hyperparameters import HyperParameters
except:
    pass

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
onehotencoder = OneHotEncoder()

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

import datetime, os
import pickle

from predict import evaluate_results, predict_test, Predictor

from settings import DEBUG, DATA_DIR, MODEL_DIR, MODEL_NAME, CLASSES_NUM # isort:skip
from settings import DATA_FILE, TEST_FILE, TARGET, TEXT_COLUMN, MAX_LEN, PATIENCE
DEBUG = True # True # False # override global settings

SAVE_MODEL = True # False # True


def enc_save(enc, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(enc, f)


def enc_load(file_name):
    with open(file_name, 'rb') as f:
        enc = pickle.load(f)
        return enc

def prepare_training(df, params, encode=False):
    # dataset in df is already preprocessed - cleaned, OrdinalEncoder applied
    val_size = params.get('val_size', 0.15)
    test_size = params.get('test_size', 0.15)
    # target labels column
    target = TARGET
    cols = df.columns.to_list()

    random_state = params.get('random_state', 42)
    cols.remove(target)

    os.makedirs(MODEL_DIR, exist_ok=True)

    if encode:
        y_label = labelencoder.fit_transform(df[target].values)
        y_hot = onehotencoder.fit_transform(y_label.reshape(-1, 1)).toarray()
        enc_save(labelencoder, MODEL_DIR+'labelencoder.pkl')
        enc_save(onehotencoder, MODEL_DIR+'onehotencoder.pkl')
    else:
        y_hot = df[target]

    X_train, X_test, y_train, y_test = train_test_split(df[cols], y_hot, 
                                                        test_size=val_size+test_size, random_state=random_state)
    if val_size and test_size:
        # both needed - split again
        X_val, X_test, y_val, y_test = train_test_split(X_test[cols], y_test, 
                                                        test_size=test_size/(val_size+test_size), random_state=random_state)

        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        return X_train, X_test, y_train, y_test, None, None


# global variables (to avoid many parameters in functions)
vocab_size = 0
train_padded = []
val_padded = []
test_padded = []
y_val = []
callback_list = []

def prepare_embeddings(df, df_train, df_val, df_test):
    global vocab_size, train_padded, val_padded, test_padded

    total_unique_words = len(set((' ').join(df[TEXT_COLUMN].values).split()))

    num_words = total_unique_words
    oov_token = '<UNK>'
    pad_type = 'post'
    trunc_type = 'post'

    X_train_final_text = df_train[TEXT_COLUMN].values
    X_val_final_text = df_val[TEXT_COLUMN].values
    X_test_final_text = df_test[TEXT_COLUMN].values


    # Tokenize our training data
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token, filters='!"#$%&()*+,-/:;<=>?@[\\]^`{|}~\t\n') #not filtering "_"
    tokenizer.fit_on_texts(X_train_final_text)

    enc_save(tokenizer, MODEL_DIR+'tokenizer.pkl')

    # Get our training  data word index
    # word_index = tokenizer.word_index
    vocab_size = len(tokenizer.word_index) + 1

    # Encode training and test data sentences into sequences
    train_sequences = tokenizer.texts_to_sequences(X_train_final_text)
    val_sequences = tokenizer.texts_to_sequences(X_val_final_text)
    test_sequences = tokenizer.texts_to_sequences(X_test_final_text)

    # Get max training and test sequence length
    maxlen = MAX_LEN

    # Pad the training  and test sequences
    train_padded = pad_sequences(train_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)
    val_padded = pad_sequences(val_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)
    test_padded = pad_sequences(test_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

    if DEBUG:
        print(f'{vocab_size=}')
        #print("Word index:\n", word_index)
        #print("\nTraining sequences:\n", train_sequences)
        #print("\nPadded training sequences:\n", train_padded)
        print("\nPadded training shape:", train_padded.shape)
        print("Training sequences data type:", type(train_sequences))
        print("Padded Training sequences data type:", type(train_padded))
        print("\nPadded validation shape:", val_padded.shape)
        print("validation sequences data type:", type(val_sequences))
        print("Padded validation sequences data type:", type(val_padded))
        print("\nPadded test shape:", test_padded.shape)
        print("test sequences data type:", type(test_sequences))
        print("Padded test sequences data type:", type(test_padded))

    return vocab_size, train_padded, val_padded, test_padded

def build_model(hp):

    model = Sequential()

    model.add(Input(shape=(MAX_LEN,), name=TEXT_COLUMN))
    model.add(Embedding(vocab_size, 64)) # 128
    
    hp_drop = hp.Float('drop_rate', min_value=0.2, max_value=0.45, step=0.2)
    model.add(SpatialDropout1D(hp_drop))
    model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=hp_drop, recurrent_dropout=hp_drop)))
    model.add(Bidirectional(LSTM(64, dropout=hp_drop, recurrent_dropout=hp_drop)))
    # model.add(LSTM(32, dropout=hp_drop)) # 
    # 		kernel_regularizer bias_regularizer 
    # model.add(Dense(16, 
    #     # activation='relu', 
    #       kernel_regularizer=regularizers.l2(0.01), 
    #     #   activity_regularizer=regularizers.l1(0.01)
    #     )
    # )
    
    model.add(Dense(units=CLASSES_NUM, activation='softmax'))

    hp_learning_rate = hp.Choice('learning_rate', values=[0.005, 0.01]) # 0.001,  
    adam = Adam(
        learning_rate=hp_learning_rate,
        # beta_1=0.9,
        # beta_2=0.999,
        # epsilon=1e-07,
    ) 

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics = ['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer=adam, metrics = ['AUC'])

    print(model.summary())

    return model


# Define Keras model function
def train_model_LSTM(df, params, random_state=42):
    global y_val, callback_list
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_training(df, {
                                                        'val_size': params['val_size'],
                                                        'test_size': params['test_size'],
                                                        'random_state': random_state
                                                        }, encode=True)

    # vocab_size, train_padded, val_padded, test_padded = 
    prepare_embeddings(df, X_train, X_val, X_test)
    X_test1 = X_test.copy()
    X_test1[TARGET] = 0
 
    classifier_name = "LSTM"
    print(f"\n>>>>> Starting Keras Hyperband tuning {classifier_name}...")

    # # quick test
    # import keras_tuner
    # model = build_model(keras_tuner.HyperParameters())
    # # print(model.summary())

    tuner = Hyperband(
            build_model,
            objective='val_accuracy',
    #        objective='val_AUC', # val_accuracy val_loss 
            # max_trials=2, # number of different models (for random!)
            # executions_per_trial=1,
            max_epochs=20,
            factor=5,
            overwrite=True,
            directory=f"logss{CLASSES_NUM}",
            project_name="sentiment_lstm",
    )

    print(tuner.search_space_summary())

    earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.005, patience=PATIENCE, verbose=1, mode="max")
    # earlystop = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, patience=PATIENCE, verbose=1)

    # analog of GridSearchCV for Keras
    tuner.search(train_padded, y_train, epochs=5, validation_data=(val_padded, y_val), callbacks=[earlystop])
    
    best_model = tuner.get_best_models(num_models=1)[0]
    print(best_model.summary())
    print(tuner.results_summary())

    # Get the best set of hyperparameters.
    best_hps = tuner.get_best_hyperparameters(5)
    print("\nBest hyperparameters:", best_hps[0])
    # # Build the model with the best hp.
    # model = build_model(best_hps[0])
    # # Fit with the entire dataset.
    # x_all = np.concatenate((x_train, x_val))
    # y_all = np.concatenate((y_train, y_val))
    # model.fit(x=x_all, y=y_all, epochs=1)
    key_metric1 = 0 # TODO ?
    key_metric2 = 0
    # try:
    #     y_pred = (np.asarray(best_model.predict([test_padded]))).round()
    #     key_metric1 = accuracy_score(y_test, y_pred)
    #     key_metric2 = balanced_accuracy_score(y_test, y_pred)
    # except Exception as e:
    #     print('!! accuracy_score error:', e) # multilabel-indicator is not supported

    if SAVE_MODEL:
        os.makedirs(MODEL_DIR, exist_ok=True)
        best_model.save(MODEL_DIR+MODEL_NAME)

        predictor = Predictor(classifier, model_dir=MODEL_DIR, verbose=False)
        try:
            accuracy, f1, recall, precision, roc_auc = predictor.evaluate_prediction(X_test[TEXT_COLUMN].values, np.argmax(y_test, axis=1), verbose=False)
        except Exception as e:
            print('!! evaluate_prediction error:', e)

    return best_model, best_hps, key_metric1, key_metric2


def train_model(df, params, random_state=42):
    classifier_name = params['classifier']
    if classifier_name=='LSTM':
        # Keras model, seriously different from sklearn models and GridSearchCV
        return train_model_LSTM(df, params, random_state)

    X_train, X_test, y_train, y_test = train_test_split(df[TEXT_COLUMN],df[TARGET], 
                                                        test_size=params['test_size'], 
                                                        random_state=random_state)

    print(f'\n-------------- {classifier_name}')
    print(f'{X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}')

    # Applying N-gram and TfidfTransformer
    count_vect = CountVectorizer(ngram_range=(1, 2))
    transformer = TfidfTransformer(norm='l2', sublinear_tf=True)
    X_train_counts = count_vect.fit_transform(X_train)
    X_train_tfidf = transformer.fit_transform(X_train_counts)
    if SAVE_MODEL:
        os.makedirs(MODEL_DIR, exist_ok=True)
        enc_save(count_vect, MODEL_DIR+'CountVectorizer.pkl')
        enc_save(transformer, MODEL_DIR+'TfidfTransformer.pkl')

    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = transformer.transform(X_test_counts)

    if DEBUG:
        print ('tfidf:', X_train_tfidf.shape, X_test_tfidf.shape, y_train.shape, y_test.shape)

    print(f"\n>>>>> Starting GridSearchCV: {classifier_name}...")
    # Define the parameter grid to tune the hyperparameters
    if classifier_name=='LogisticRegression':
        param_grid = {
            'C': [0.5, 1.0, 2.0, 3.0],
            'max_iter': [25, 50, 100, 200],
        }
        classifier = LogisticRegression(solver='liblinear', random_state=random_state)
    elif classifier_name=='MultinomialNB':
        param_grid = {
            'alpha': [0.00001, 0.0001, 0.001, 0.1, 1, 10]
        }
        classifier = MultinomialNB() # no random_state
    elif classifier_name=='DecisionTreeClassifier':
        param_grid = {
            'max_depth': [10, 50, None],
            'min_samples_leaf': [1, 5],
            'min_samples_split': [2, 10],
        }
        classifier = DecisionTreeClassifier(random_state=random_state)
    elif classifier_name=='RandomForestClassifier':
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [10, 50, None],
        }
        classifier = RandomForestClassifier(class_weight='balanced', random_state=random_state)
    elif classifier_name=='AdaBoostClassifier':
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.2, 1.0, 1.5, 2.0], 
        }
        classifier = AdaBoostClassifier(algorithm='SAMME', random_state=random_state)
    else:
        print('!! Unexpected classifier:', classifier_name)
        return

    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid,
                               cv=params['cv'],
                               n_jobs=-1,
                               verbose=1, #3, #2,
                               scoring=params['estimator'] # balanced_accuracy accuracy neg_mean_squared_error roc_auc_ovr
                               )
    # grid_search.fit(X_train, y_train)
    grid_search.fit(X_train_tfidf, y_train)
    best_classifier = grid_search.best_estimator_ # Get the best estimator from the grid search
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f'\nHPO grid search: {grid_search.scorer_}, {grid_search.n_splits_} splits')
    print(f"Best parameters: {best_params}")

    if DEBUG:
        # simple prediction
        text = 'The movie is surprising, with plenty of unsettling plot twists'
        print('\n\n1 item test prediction:')
        print(text)
        mc = count_vect.transform([text])
        m = transformer.transform(mc)
        y_pred = best_classifier.predict(m)
        print('>>', text, '->', y_pred)

    # calculating test prediction metrics
    # y_pred = best_classifier.predict(X_test)
    y_pred = best_classifier.predict(X_test_tfidf)
    evaluate_results(f"{classifier_name}, optimized for {params['estimator']}", y_test, y_pred, verbose=True)
    key_metric1 = accuracy_score(y_test, y_pred)
    key_metric2 = balanced_accuracy_score(y_test, y_pred)
    print(f" {params['estimator']} best_score_: {best_score:.3f}")
    if DEBUG:
        features_ = list(df.columns)
        features_.remove(TARGET)

    if SAVE_MODEL:
        os.makedirs(MODEL_DIR, exist_ok=True)
        filename = f'{MODEL_DIR}{classifier_name}.pkl'
        pickle.dump(best_classifier, open(filename, 'wb'))

        # testing saved model
        print('\nQuick test of just saved and loaded model:')
        predictor = Predictor(classifier_name, model_dir=MODEL_DIR, verbose=False)
        accuracy, f1, recall, precision, roc_auc = predictor.evaluate_prediction(X_test, y_test, verbose=False)

    return best_classifier, best_params, key_metric1, key_metric2


def models_comparison(evaluation_results):
    # [key_metric1, key_metric2, accuracy, f1, recall, precision, roc_auc]
    import matplotlib.pyplot as plt

    # Data for the bar chart
    models = []
    final_accuracy = []
    final_precision = []
    final_recall = []
    final_f1_score = []
    final_roc_auc = []
    for classifier, result in evaluation_results.items():
        print(classifier, result)
        models.append(classifier)
        final_accuracy.append(result[2])
        final_precision.append(result[3])
        final_recall.append(result[4])
        final_f1_score.append(result[5])
        final_roc_auc.append(result[6])

    bar_width = 0.15

    # Set the positions of the bars on the x-axis
    r1 = np.arange(len(models))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(r1, final_accuracy, bar_width, label='Accuracy')
    bars2 = ax.bar(r2, final_precision, bar_width, label='Precision')
    bars3 = ax.bar(r3, final_recall, bar_width, label='Recall')
    bars4 = ax.bar(r4, final_f1_score, bar_width, label='F1-score')
    bars5 = ax.bar(r5, final_roc_auc, bar_width, label='ROC AUC')

    # Add labels, title, and legend
    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('Model Comparison')
    ax.set_xticks([r + bar_width for r in range(len(models))])
    ax.set_xticklabels(models)
    ax.legend(loc='lower right')

    # Add value labels on top of bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{:.3f}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    autolabel(bars4)
    autolabel(bars5)

    # Display the chart
    plt.tight_layout()
    # plt.show()
    plt.savefig('models_comparison.png')

if __name__ == '__main__':
    from time import time

    from preprocess import load_data
    df = load_data()
    # if train and test data from the same file, reserve records for extra testing
    reserved = 200 if TEST_FILE['FILE_NAME']==DATA_FILE['FILE_NAME'] else 0
    df = df.head(df.shape[0]-reserved) # "reserved"

    df_test = load_data(TEST_FILE)
    test_samples = df_test.shape[0]
    df_test = df_test
    if reserved:
        df_test = df_test.tail(reserved) # if the train and test file is the same, use "reserved" rows
    
    evaluation_results = {}
    estimator = 'accuracy' # 'accuracy' 'roc_auc_ovr'
    for classifier in [
                    # 'RandomForestClassifier',  # slow
                    # 'AdaBoostClassifier',  # slow
                    # 'DecisionTreeClassifier', # fast
                    'LogisticRegression', # fast
                    'MultinomialNB', # fast
                    # 'LSTM', # slow
                    ]:
        t_start = time()
        params = {'classifier': classifier,
                    'estimator': estimator,
                    'cv': 2,
                    'val_size': 0.15,
                    'test_size': 0.15,
                    }
        best_classifier, best_params, key_metric1, key_metric2 = train_model(df, params, random_state=77)
        print(f'\n Training {classifier} finished in {(time() - t_start):.3f} second(s)\n')
        # continue

        if classifier=='LSTM': 
            # prediction is really slow, limiting test sample
            df_test = df_test.tail(200)
        print(f"====================== Testing model on {df_test.shape[0]} rows of {TEST_FILE['FILE_NAME']}:")
        if DEBUG:
            # all rows with label/prediction comparison
            predict_test(df_test, classifier)

        predictor = Predictor(classifier, model_dir=MODEL_DIR, verbose=False)
        accuracy, f1, recall, precision, roc_auc = predictor.evaluate_prediction(df_test[TEXT_COLUMN].values, df_test[TARGET].values, 
                                                                        verbose='difference')
        evaluation_results[classifier] = [key_metric1, key_metric2, accuracy, f1, recall, precision, roc_auc]

    print(f'\n Finished in {(time() - t_start):.3f} second(s)\n')
    
    # compare performance
    models_comparison(evaluation_results)
