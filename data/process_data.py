import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Loads messages and categories CSVs from filepath args into dataframes
    Merges dataframes and returns

    Args:
    messages_filepath - string filepath to 'disaster_messages.csv'
    categories_filepath - string filepath to 'disaster_categories.csv'

    Returns:
    df - Pandas DataFrame
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    return pd.merge(messages, categories, how='outer')


def clean_data(df):
    '''
    Adds categories columns as values (1 or 0) and removes duplicates

    Args:
    df - pandas dataframe

    Returns:
    df - pandas dataframe
    '''
    # Split categories col into separate categories columns - categories are separated by ';'
    categories = df.categories.str.split(';', expand=True)

    # Get categories as column headers
    categories.columns = categories.iloc[0].apply(lambda cat_str: cat_str.split('-')[0])
    
    # set value to be the last character in each string, convert to numeric
    for col in categories.columns:
        categories[col] = pd.to_numeric(categories[col].str[-1])
    
    # append categories columns to df
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # remove duplicates
    return df.drop_duplicates()


def save_data(df, database_filename):
    '''
    Saves dataframe to a sqlite3 database

    Args:
    df - pandas dataframe to be saved
    database_filename - path where df should be saved

    Returns:
    True/False - bool indicating whether this was successfully saved
    '''
    try:
        engine = create_engine(f'sqlite:///{database_filename}')
        df.to_sql(
            TABLE_NAME,
            con=engine,
            index=False,
            if_exists='replace'
        )
        return True
    except:
        return False

    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        
        print('Cleaning data...')
        df = clean_data(df)

        print(df.sum())
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        
        if save_data(df, database_filepath):
            print(f'Data was saved to {database_filepath}')
        else:
            print(f'Data could not be saved to {database_filepath}')
        
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