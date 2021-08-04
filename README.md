# udacity-disaster-response

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains tweet and messages from real-life disaster events. The project's aim is to build a Natural Language Processing (NLP) pipeline to classify disaster messages. Using the web app an emergency worker can input a new message and get classification results in several categories so to have an idea what kind of help is needed: "water", "shelter", "food", etc.


## This project is divided in the following key sections:

- Processing data, building an ETL pipeline to extract data from source, clean the data and save them in a SQLite DB
- Build a machine learning pipeline to train the which can classify text message in various categories
- Run a web app which can show model results in real time


## Install

This project requires Python 3.x and the following Python libraries installed:

- NumPy
- Pandas
- Matplotlib
- Json
- Plotly
- Nltk
- Flask
- Sklearn
- Sqlalchemy
- Sys
- Re
- Pickle


## Executing Program:

You can run the following commands in the project's directory to set up the database, train model and save the model.

- To run ETL pipeline to clean data and store the processed data in the database `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db`
- To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file `python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl`
- Run the following command in the app's directory to run your web app: `python run.py`

Go to http://0.0.0.0:3001/


## Code and data
- process_data.py: this code extracts data from both CSV files: messages.csv (containing message data) and categories.csv (classes of messages) and creates an SQLite database containing a merged and cleaned version of this data.
- train_classifier.py: this code takes the SQLite database produced by process_data.py as an input and uses the data contained within it to train and tune a ML model for categorizing messages. The output is a pickle file containing the fitted model. Test evaluation metrics are also printed as part of the training process.
- ETL Pipeline Preparation.ipynb: the code and analysis contained in this Jupyter notebook was used in the development of process_data.py. process_data.py automates this notebook.
- ML Pipeline Preparation.ipynb: the code and analysis contained in this Jupyter notebook was used in the development of train_classifier.py. In particular, it contains the analysis used to tune the ML model and determine which model to use. train_classifier.py automates the model fitting process contained in this notebook.
- disaster_messages.csv, disaster_categories.csv contain sample messages (real messages that were sent during disaster events) and categories datasets in csv format.
- templates folder: this folder contains all of the files necessary to run and render the web app.

## Screenshots

![alt text](https://github.com/kwlwman2/udacity-disaster-response/blob/main/screenshots/Screen%20Shot%202021-08-03%20at%208.54.19%20PM.png?raw=true)


## Acknowledgements

- Udacity for providing an amazing Data Science Nanodegree Program
- Figure Eight for providing the relevant dataset to train the model


