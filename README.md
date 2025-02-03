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

[Subjectivity dataset v1.0 from Movie Review Data](https://www.cs.cornell.edu/people/pabo/movie-review-data/) includes 5000 subjective and 5000 objective processed sentences. Dataset ntroduced in Pang/Lee ACL.
I combined all records into one table and then split into [train](/data/subjectivity_train.csv) and [test](/data/subjectivity_test.csv) (20%) CSV files includes, resulting 8000 and 2000 records with 2 columns - `text` and `sentiment`.

