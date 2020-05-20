import json
import plotly
import pandas as pd
from pathlib import Path
import os

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

# var - /data/process_data.py
DB_NAME = 'DisasterResponse'
TABLE_NAME = 'Messages_Categories'

# var - /models
MODEL_FP = (os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'models/classifier.pkl'))

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine(f'sqlite:///data/{DB_NAME}.db')
df = pd.read_sql_table(TABLE_NAME, engine)
# load model
model = joblib.load(MODEL_FP)
# load accuracy scores
accuracy_scores_df = pd.read_sql_table('Accuracy_Scores', engine).T.rename(columns={0:'accuracy_score'}).sort_values(by='accuracy_score')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_counts_series = df.drop(columns=['message', 'id', 'original', 'genre']).sum().sort_values()
    category_names = list(category_counts_series.index)
    category_counts = list(category_counts_series.values)
    
    accuracy_category_names = list(accuracy_scores_df['accuracy_score'].index)
    accuracy_values = list(accuracy_scores_df['accuracy_score'].values)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
      {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                },
                'automargin':True,
            }
        },
          {
            'data': [
                Bar(
                    x=accuracy_category_names,
                    y=accuracy_values
                )
            ],
            'layout': {
                'title': 'Model Accuracy Score by Message Category',
                'yaxis': {
                    'title': "Accuracy"
                },
                'xaxis': {
                    'title': "Category"
                },
                'automargin':True,
            }
        },
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    print(classification_labels)
    classification_results = dict(zip(df.columns[4:], classification_labels))
    print(classification_results)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()