# Project Overiew 

E-commerce companies usually sells different product online. Customers can leave product feedback across various online channels (sending email, writing chat FAQ messages on your website, maybe calling into your support center, or posting messages on your company’s mobile app, popular social networks, or partner websites). And as a business, it's important to be able to capture customer feedbacks as quickly as possible to spot any change in market trends or customer behavior and then be alerted about potential product issues.

The main task of this project is to build an Natural Language Processing (NLP) models that will take those product reviews as input. The model will then be used to classify the sentiment of the reviews into 3 classes ( 1- positive, 0 - neutral, -1  - negative). For example, a review such as, 
- “I simply love it!” should be classified into the positive class, 
- "It's OK" should be classified into the neutral class, and 
- "I'm not satisfied" should be classified into the negative class. 


## [Analyze Dataset](https://github.com/jssaladinoo/nlp/blob/main/1_analyze/Analyze%20dataset.ipynb)

This notebook ingest and transform the customer product reviews dataset. The transformed data is saved in a S3 bucket and analyzed using AWS Data Wrangler


## [Bias Detection](https://github.com/jssaladinoo/nlp/blob/main/2_bias_detection/Bias%20Detection.ipynb)

Bias can be present in your data before any model training occurs. Inspecting the dataset for bias can help detect collection gaps, inform your feature engineering, and understand societal biases the dataset may reflect. This notebook used Sagemaker Clarify to analyze bias on the generate and analyze bias report, and prepare the dataset for the model training.

## [AutoML Text Classifier](https://github.com/jssaladinoo/nlp/blob/main/3_automl_text_classifier/AutoML.ipynb)

This notebook utilizes AutoML to inspect the raw dataset, apply feature processors, pick the best set of algorithms, train and tune multiple models, and then rank the models based on performance - all with just a few clicks.  Sagemaker Autopilot transparently generates a set of Python scripts and notebooks for a complete end-to-end pipeline including data analysis, candidate generation, feature engineering, and model training/tuning.

The best model candidate, which has 0.6150 validation accuracy, is a xgboost classifier with 'MultiColumnTfidfVectorizer' and 'RobustStandardScaler' data transformation. 

## [Feature Engineering](https://github.com/jssaladinoo/nlp/blob/main/4_feature_engineering/Feature%20engineering.ipynb)

This notebook transforms the customer's product review into a format that is amenable for training BERT-based natural language processing (NLP) models.

## [Text classifier with BERT](https://github.com/jssaladinoo/nlp/blob/main/5_BERT_text_classifier/Text%20Classifier%20with%20BERT.ipynb)

Utiizing the engineered features in the previous task, this notebook trains a text classifier using a variant of BERT called [RoBERTa](https://arxiv.org/abs/1907.11692) - a Robustly Optimized BERT Pretraining Approach - within a PyTorch model. 

The model achieved a 0.6914 validation accuracy which is slightly higher than the AutoML model.

## [Build ML Pipelines](https://github.com/jssaladinoo/nlp/blob/main/6_build_ml_pipeline/Building%20ML%20Pipeline.ipynb)

This notebook builds a SageMaker Pipeline to train and deploy a BERT-based test classifier. The pipeline follows a typical machine learning application pattern of pre-processing, training, evaluation, and model registration.  

In the processing step, feature engineering is performed to transform the `review_body` text into BERT embeddings using the pre-trained BERT model and split the dataset into train, validation and test files. The transformed dataset is stored in a feature store. To optimize for Tensorflow training, the transformed dataset files are saved using the TFRecord format in Amazon S3.

In the training step, the BERT model is fine-tuned to the customer reviews dataset and a new classification layer is added to predict the `sentiment` for a given `review_body`.

In the evaluation step, the trained model and the test dataset were taken as input, and produce a JSON file containing classification evaluation metrics.

In the condition step, trained models that were able to exceed the pre-determined threshold value for the model accuracy are added in the registry.


## [Hyperparameter tuning](https://github.com/jssaladinoo/nlp/blob/main/7_hyperparameter_tuning/Hyperparameter%20tuning.ipynb)

When training ML models, hyperparameter tuning is a step taken to find the best performing training model. In this notebook a random algorithm of Automated Hyperparameter Tuning to train a BERT-based natural language processing (NLP) classifier is applied. The model analyzes customer feedback and classifies the messages into positive (1), neutral (0), and negative (-1) sentiments.

The hyperparameter space is limited to 'learning_rate' which is a continuous parameter between (0.00001, 0.00005) and train_batch_size which is categorical  parameter with values 128 and 256. The best model have 0.000047 learning_rate and 128 train_batch_size and has a 0.7383 validation accuracy. 


## [Model deployment](https://github.com/jssaladinoo/nlp/blob/main/8_model_deployment/Model%20deployment.ipynb)

This notebook creates an endpoint with multiple variants, splitting the traffic between them.

## [Human-in-the-loop Pipelines](https://github.com/jssaladinoo/nlp/blob/main/9_human_in_thel_pipeline/Human-in-the-loop%20Pipelines.ipynb)

This notebook aims to create a human workforce, a human task UI, and then define the human review workflow to perform data labeling. 
