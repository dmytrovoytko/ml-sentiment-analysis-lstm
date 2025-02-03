# ML project Sentiment analysis for movie reviews

Project for DataTalks.Club Machine Learning ZoomCamp`24:

![ML Sentiment analysis](/EDA/word-cloud-all.png)

Project can be tested and deployed in **GitHub CodeSpaces** (the easiest option, and free), cloud virtual machine (AWS, Azure, GCP), or just locally.
For the GitHub CodeSpace option you don't need to use anything extra at all - just your favorite web browser + GitHub account is totally enough.

## Problem statement

How do you choose, and what do you take into account?

Every day we make decisions - what to buy, what to watch, what to read, etc. And quite often we make our choices based on articles, recommendations and reviews. Amazon, Goodreads, Rotten Tomatoes, IMDB provide us simple numbers in stars ⭐️ and user reviews. However, two different people can easily give 4/5 to a movie with a different attitude. 4 for something like "a good movie with a couple of great actors", and 4 for "a not so good movie as I expected this to be a perfect Oscar-winning story". So 4 can easily be positive and negative. This is where sentiment analysis can become a challenge.

However there is another aspect of reviews analysis - **Objectivity-Subjectivity**. Textual information can be broadly categorized into two main types: facts and opinions. Facts are objective expressions about entities, events and their properties. Opinions are usually subjective expressions that describe people’s sentiments, appraisals or feelings toward entities, events and their properties. If you want to make a better decision, you better consider recommendations based on facts.

I decided to find out how Machine Learning could help with Objectivity-Subjectivity detection. This type of sentiment analysis is not as spread as Positive-Negative.
I chose a [subjectivity dataset v1.0 from Movie Review Data](https://www.cs.cornell.edu/people/pabo/movie-review-data/). Let's see how well we can predict subjectivity in it!

## 🎯 Goals

This is my 2nd project in [Machine Learning ZoomCamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)'24.

**The main goal** is straight-forward: build an end-to-end Machine Learning project:
- choose an interesting dataset
- load data, conduct exploratory data analysis (EDA), clean it
- train & test ML model(s)
- deploy the model (as a web service) using containerization

## 🔢 Dataset

[Subjectivity dataset v1.0 from Movie Review Data](https://www.cs.cornell.edu/people/pabo/movie-review-data/) includes 5000 subjective and 5000 objective processed sentences. Dataset introduced in Pang/Lee ACL.

- rotten_imdb.tar.gz: contains readme and two data files that were used in the experiments described in Pang/Lee ACL 2004.

Specifically: 
  * quote.tok.gt9.5000 contains 5000 subjective sentences (or snippets);
  * plot.tok.gt9.5000 contains 5000 objective sentences.

> Each line in these two files corresponds to a single sentence or snippet; all sentences (or snippets) are down-cased.  Only sentences or snippets containing at least 10 tokens were included. The sentences and snippets were labeled automatically: authors assumed that all snippets from the Rotten Tomatoes pages are subjective, and all sentences from IMDb plot summaries are objective. This is mostly true; but plot summaries can occasionally contain subjective sentences that are mis-labeled as objective.

I combined all records into one table and then split into [train](/data/subjectivity_train.csv) and [test](/data/subjectivity_test.csv) (20%) CSV files, resulting 8000 and 2000 records with 2 columns - `text` and `sentiment`. Data preparation script [prepare_csv.py](/data/prepare_csv.py)


## 📊 EDA

Dataset is well prepared, without duplicates and null values, balanced.
You can explore detailed information in [Jupyter notebook](/sentiment-analysis-subjectivity.ipynb)

For Natural language processing (NLP) classification helpful visual tool is word cloud:

- word cloud for full dataset is presented on title picture
- objective

![Objective](/EDA/word-cloud-Objective.png)

- subjective

![Subjective](/EDA/word-cloud-Subjective.png)



## 🎛 Model training

I started with Keras LSTM model for classification, 
then tried several sklearn ML models:

- MultinomialNB (fast)
- LogisticRegression (fast)
- DecisionTreeClassifier (fast)
- RandomForestClassifier (slow)
- AdaBoostClassifier (slow)

Also I experimented with hyperparameter tuning to improve performance.

**Comparison of performance** for better performing models trained with hyperparameter tuning:

![Models comparison1](/EDA/models_comparison1.png)


## Python scripts for data pre-processing and training

- [preprocess.py](/prediction_service/preprocess.py)
- [train_model.py](/prediction_service/train_model.py)

`train_model.py` includes a more advanced hyperparameter tuning for all models (including LSTM).
I used GridSearchCV and measured time for training each ML classifier, and Kerastuner Hyperband for LSTM.
You can find results in [sklearn_lstm-subj.txt](/sklearn_lstm-subj.txt).


## 🚀 Instructions to reproduce

- [Setup environment](#hammer_and_wrench-setup-environment)
- [Train model](#arrow_forward-train-model)
- [Test prediction service](#mag_right-test-prediction-service)
- [Deployment](#inbox_tray-deployment)

### :hammer_and_wrench: Setup environment

1. **Fork this repo on GitHub**. Or use `git clone https://github.com/dmytrovoytko/ml-sentiment-analysis-lstm.git` command to clone it locally, then `cd ml-sentiment-analysis-lstm`.
2. Create GitHub CodeSpace from the repo.
3. **Start CodeSpace**
4. **Go to the prediction service directory** `prediction_service`
5. The app works in docker container, **you don't need to install packages locally to test it**.
6. Only if you want to develop the project locally, you can run `pip install -r requirements.txt` (project tested on python 3.11/3.12).
7. If you want to rerun [Jupyter notebook](/churn-prediction-3.ipynb) you will probably need to install packages using `pip install -r requirements.txt` which contains all required libraries with their tested together versions.

### :arrow_forward: Train model

1. **Run `bash deploy.sh` to build and start app container**. As packages include TensorFlow/Keras, building container takes some time. When new log messages will stop appearing, press enter to return to a command line (service will keep running in background).

![docker-compose up](/screenshots/docker-compose-00.png)

When you see these messages app is ready

![docker-compose up](/screenshots/docker-compose-01.png)

2. To reproduce training process run `bash train.sh` which starts model training in docker container. If you run it locally, execute `python train_model.py`. 

The dataset is small enough. By default, only 2 fast ML models are enabled in training script (LogisticRegression and MultinomialNB), so it should finish quickly - a minute. If you want to enable training LSTM just uncomment corresponding line, and be ready to wait much longer.



![Training prediction models in dockerl](/screenshots/model-training-1.png)

As a result you will see log similar to [sklearn_lstm-subj.txt](/sklearn_lstm-subj.txt).

![Training prediction models in dockerl](/screenshots/model-training-2.png)


