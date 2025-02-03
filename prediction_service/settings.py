# show extra information for checking execution
DEBUG = False  # True # False

DATA_DIR = f'./data/'  # ! with '/' at the end!
MODEL_DIR = f'./model/'  # ! with '/' at the end!
MODEL_NAME = 'model.keras' # for Keras models

# Data set

TEXT_COLUMN = "text"  # texts column, should be lowercased, space -> '_'
TARGET = "sentiment"  # labels column, should be lowercased, space -> '_'
CLASSES_NUM = 2

# DATA_FILE = {
# 	'FILE_NAME': 'rt-polarity_train.csv',
# 	'ENCODING': "UTF-8",
# 	'SOURCE_TEXT_COLUMN': "text",
# 	'SOURCE_TARGET': "sentiment",  # labels column, should be lowercased, space -> '_'
# }

# TEST_FILE = {
# 	'FILE_NAME': 'rt-polarity_test.csv',
# 	'ENCODING': "UTF-8",
# 	'SOURCE_TEXT_COLUMN': "text",
# 	'SOURCE_TARGET': "sentiment",  # labels column, should be lowercased, space -> '_'
# }

DATA_FILE = {
	'FILE_NAME': 'subjectivity_train.csv',
	'ENCODING': "UTF-8",
	'SOURCE_TEXT_COLUMN': "text",
	'SOURCE_TARGET': "sentiment",  # labels column, should be lowercased, space -> '_'
	'LABELS': {0:'Objective', 1:'Subjective'}
}

TEST_FILE = {
	'FILE_NAME': 'subjectivity_test.csv',
	'ENCODING': "UTF-8",
	'SOURCE_TEXT_COLUMN': "text",
	'SOURCE_TARGET': "sentiment",  # labels column, should be lowercased, space -> '_'
	'LABELS': {0:'Objective', 1:'Subjective'}
}


# preprocessing
USE_ENCODER = False

# training, prediction
# max sequence length
MAX_LEN = 120
# early stopping
PATIENCE = 3

# prediction
# DEFAULT_CLASSIFIER = "LSTM" 
DEFAULT_CLASSIFIER = 'MultinomialNB' # "LogisticRegression" 

# web app settings
PORT = 5555

