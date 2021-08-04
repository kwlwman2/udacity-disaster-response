# udacity-disaster-response

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains tweet and messages from real-life disaster events. The project's aim is to build a Natural Language Processing (NLP) pipeline to categorize messages to differet categories of disasters on a real time basis.


## This project is divided in the following key sections:

- Processing data, building an ETL pipeline to extract data from source, clean the data and save them in a SQLite DB
- Build a machine learning pipeline to train the which can classify text message in various categories
- Run a web app which can show model results in real time


## Executing Program:

You can run the following commands in the project's directory to set up the database, train model and save the model.

- To run ETL pipeline to clean data and store the processed data in the database `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db`
- To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file `python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl`
- Run the following command in the app's directory to run your web app: `python run.py`

Go to http://0.0.0.0:3001/


