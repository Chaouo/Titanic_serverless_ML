 We implemented a serverless machine learning system for the Titanic data. In our system,  the model has a user interface deployed by Gradio on HuggingFace. It also provides a prediction service that uses Modal to run the feature pipeline, inference pipeline, and batch inference pipeline once a day and train the model with the training program. The Modal platform will publish features, labels, and prediction results to our user interface. Our system will generate a synthetic passenger once per day, and the model will predict whether the passenger is deceased or rescued. Then the UI is going to show the actual status of the passenger, as well as our predicted result.  

File Description:

titan-feature-pipeline.py
 Insert features and labels into our serverless model. Specifically, we created our feature group with a specified name, version, primary key, etc. Then insert the titanic dataflow into our feature group.

titan-training-pipeline.py
It reads training data with a Feature View from Hopsworks and trains a binary classifier model to predict if a particular passenger survived the Titanic or not. In our case, we use VotingClassifier, which chooses the best model among MultinomialNB, LogisticRegression, LinearSVC, AdaBoostClassifier,  and BaggingClassifier. Then we create a model registry and upload our model to it. 


titanic-feature-pipeline-daily.py
It generates synthetic data at a proper cadence (in our case, it runs once per day) and update our feature pipeline on Hopswork to allow it to add new synthetic passengers.

titanic-batch-inference-pipeline-daily.py
It predicts if the synthetic passengers survived or not, and build a Gradio application to show the most recent synthetic passenger prediction and
outcome, and a confusion matrix with historical prediction performance.
