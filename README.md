# Disaster Response Web App

## Table of Contents

1. [Project Overview](#overview)
2. [Setup/Testing Instructions](#setup)
3. [File/Folder Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Overview <a name="overview"></a>

In this project I was given the task of developing a machine learning (ML) model that can take input of message, and then classifies the message to help direct the message sender to the appropriate assistance needed.  To build this model I developed a script to process the training/test data for the ML model. Once the data was successfully processed, I ran the data through a `jupyter notebook` with various hyper-parameters find the best parameters for this project.  Finally, after the model was successfully trained, I then configured a Flask app that was provided by Udacity to utilize my database and ML model developed.

## Setup/Testing Instructions <a name="setup"></a>

Git clone this repo to your local environment and then run the following commands in a new terminal.

```bash
$ cd dsn_disaster_response_webapp
$ conda env create -f environment.yml
$ conda activate dr-app
$ cd main/app
$ python run.py
```

Now open a new web browser and [http://localhost:3001/](http://localhost:3001/)

![App when first launched](docs/app01.png)

When you enter a message into the text field and click **Classify Message**. You will see the category(ies) that best represent the message.

![Message Classified](docs/app02.png)

The diagram below displays the general method of how the system operates.  

Within the section titiled **Data Processing & Model Training** is where `process_data.py` is utilized to clean and store the data within a sqlite database.  Once the data is stored in a uniform format, you can run the `train_classifier.py` file to start training the ML model.  The outputs from both python scripts are utilized in Disaster application.  

![System Design](docs/sysarch.png)

## File/Folder Descriptions <a name="files"></a>

Within the `main/` folder is where the mechanics of the application are found.

**app/run.py** is the Flask application which defines how the application will be served from the flask web server. This file has all the API endoints defined and what methods/html templates it will utilize.

**app/templates/** contains the `html` templates that are utilized in this web application.

**data/process_data.py** is the python script that performs the data processing steps on the `.csv` files located within this folder.  The output of this file is a sqlite database.

**models/train_classifier.py** is the training python script that will generate the ML model utilized in the web application.  This model is called to classify a message that is entered by the user and the classification job returns the categories that it believes best fit this message.

## Licensing, Authors, Acknowledgements <a name="licensing"></a>

The data that was utilized in this project was obtained from [Figure Eight](https://appen.com/open-source-datasets/) via Udacity Data Science Nano Degree.
