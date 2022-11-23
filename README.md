# Titanic_serverless_ML

We implemented a serverless machine learning system for the Titanic data. In our system,  the model has a user interface deployed by Gradio on HuggingFace. It also provides prediction service which uses Modal to run the feature pipeline, inference pipeline and a batch inference pipeline once a day and train the model with the training program. Modal will publish features, labels and the prediction results to our user interface. Our system will generate a synthetic passenger once per day, and the model will predict whether the passenger is deceased or rescued. Then the UI is going to show the actual status of the passenger, as well as our predicted result.  

File Descriptions:

