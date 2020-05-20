# Disaster Response Pipeline Project

### Summary:

This repo contains the Disaster Response Pipeline Project for the Udacity Data Scientist Nanodegree Program course.

The project sets up an ETL pipeline to read in data, clean, and save data from CSV files provided by the course.
* These CSV files contain emergency messages categorized by need.
Then, the cleaned dataset is used to train a classifier to label new messages with a category.
Finally, this project deploys a web application that provides an overview of the dataset and uses the trained model to label new messages.

Follow the below instructions to set up and run...

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
