# Disaster Response Pipeline Project Developer Notes

### Instructions for Processing and Training Model:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database

        ```bash
        $ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        ```

    - To run ML pipeline that trains classifier and saves

        ```bash
        $ python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        ```

2. Run the following command in the app's directory to run your web app.

    ```bash
    $ python run.py
    ```

3. Go to http://0.0.0.0:3001/
