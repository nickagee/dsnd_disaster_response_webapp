import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load & merge messages and categories ds
    
    messages_filepath: path for csv with messages
    categories_filepath: path for csv with cats
       
    Returns:
    df: dataframe
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df =pd.merge(messages,categories,how='inner',on='id')
    
    return df

def clean_data(df):
    """
    Loads df and cleans
    
    Input: df with joined messages and categories data
    
    Return: Clean dataframe free from duplicates
    """
    categories = df.categories.str.split(';',expand=True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] =  categories[column].apply(lambda x: x[-1:])
        categories[column] = categories[column].astype('int') 
        
    categories.drop('child_alone', axis = 1, inplace = True)
    df.drop('categories',axis=1,inplace=True) 
    df = pd.concat([df,categories],axis=1)
    df.drop_duplicates(inplace=True)
    df = df[df['related'] != 2]
    return df

def save_data(df, database_filename):
    """
    Save cleaned data into an SQLite database.
    
    df: dataframe to be saved
    database_filename: path/db name
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('my_disaster_response_table', engine, index=False)

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
        print('Please provide : python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db')


if __name__ == '__main__':
    main()