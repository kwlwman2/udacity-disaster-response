import sys

# import libraries
import pandas as pd
import sqlite3
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # read feature data
    messages = pd.read_csv(messages_filepath)
    # read predictor data
    categories = pd.read_csv(categories_filepath)
    # merge two dataframe
    df = pd.merge(messages, categories, how = 'inner', on='id')
    # return df
    return df 

def clean_data(df):
    y_labels = df['categories'].str.split(';',expand=True)
    y_labels.columns = [label[0] for label in y_labels.iloc[0,:].str.split('-')]

    # loop over all columns in y_labels
    for column in y_labels:
        # set each value to be the last character of the string
        y_labels[column] = y_labels[column].apply(lambda x: x.split('-')[1])
    
        # convert column from string to numeric
        y_labels[column] = pd.to_numeric(y_labels[column], errors='coerce')
        
    
    # drop the original categories column from `df`
    df = df.drop('categories',axis = 1)
    
    # concat the feature df and categories df
    df = pd.concat([df, y_labels],axis = 1 )
    
    # remove duplicatess
    df = df.drop_duplicates(keep='first')
    
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + str(database_filename))
    df.to_sql('MessagesCategories',
              engine, 
              index=False, 
              if_exists = 'replace')
    return None


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()