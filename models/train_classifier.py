import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import LinearSVC


def load_data(database_filepath):
    '''
    Load dataset from db, define feature and target variables X and Y
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    query = '''
    SELECT * FROM Messages_Categories
    '''
    # load data
    df = pd.read_sql(
        query,
        con=engine,
    )
    # remove columns of all zeros
    df = df.loc[:, (df != 0).any(axis=0)]
    
    # define feature, target (categories) variables 
    X = df.message.values
    df = df.drop(columns=['message', 'id', 'original', 'genre'])
    y = df.apply(pd.to_numeric).values
    return X, y, list(df.columns)


def tokenize(text):
    # tokenize words
    tokens = word_tokenize(text)
    # lemmatize words 
    lemmatizer = WordNetLemmatizer()

    stop_words = stopwords.words("english")
    clean_tokens = []
    for token in tokens:
        if token not in stop_words:
            clean_token = lemmatizer.lemmatize(token).lower().strip()
            clean_tokens.append(clean_token)
    
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(
            tokenizer=tokenize,
            ngram_range=(1,3),
            max_df=1.0,
        )),
        ('tfidf', TfidfTransformer(
            use_idf=True,
        )),
        ('clf', MultiOutputClassifier(LinearSVC()))
        
    ])
    print(pipeline.get_params())
    # return pipeline
    parameters = {
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=1)
    return cv


def evaluate_model(model, X_test, y_test, category_names, database_filepath):

    
    # model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f'\nBest params: {model.best_params_}')
    print(f'\nBest score: {model.best_score_}')

    # save accuracy values by category
    accuracy_dict = {}
    for ix, category in enumerate(category_names):
        report = classification_report(y_test[:,ix], y_pred[:,ix], output_dict=True)
        print(report)
        accuracy_dict[category] = report['accuracy']
        print(f'Saving accuracy score for {category}: {report['accuracy']}')
    
    engine = create_engine(f'sqlite:///{database_filepath}')

    accuracy_df = pd.DataFrame(accuracy_dict, index=['accuracy_score'])
    print(accuracy_df)

    df = accuracy_df.to_sql(
        "Accuracy_Scores",
        con=engine,
        index=False,
        if_exists='replace'
    )
    print(f'Successfully saved accuracy_df to Accuracy_Scores table at {database_filepath}')


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))        


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model() 

        # Train pipeline
        print('Training model...')
        model.fit(X_train, y_train)

        # predict on test data
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names, database_filepath)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()