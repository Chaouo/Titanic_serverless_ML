import os
import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install(["hopsworks==3.0.4", "seaborn", "joblib", "scikit-learn"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def g():
    import hopsworks
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import OneClassSVM
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import VotingClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    import seaborn as sns
    from matplotlib import pyplot
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    import joblib

    # You have to set the environment variable 'HOPSWORKS_API_KEY' for login to succeed
    project = hopsworks.login()
    # fs is a reference to the Hopsworks Feature Store
    fs = project.get_feature_store()

    # The feature view is the input set of features for your model. The features can come from different feature groups.    
    # You can select features from different feature groups and join them together to create a feature view
    try: 
        feature_view = fs.get_feature_view(name="titanic_modal", version=3)
    except:
        titan_fg = fs.get_feature_group(name="titanic_modal", version=3)
        query = titan_fg.select_all()
        feature_view = fs.create_feature_view(name="titanic_modal",
                                          version=3,
                                          description="Read from pre-processed Titanic dataset",
                                          labels=["Survived"],
                                          query=query)    

    # You can read training data, randomly split into train/test sets of features (X) and labels (y)        
    X_train, X_test, y_train, y_test = feature_view.train_test_split(0.2)

    # Train our model with the Scikit-learn K-nearest-neighbors algorithm using our features (X_train) and labels (y_train)
    #model = KNeighborsClassifier(n_neighbors=20)
    #model.fit(X_train, y_train.values.ravel())
    # BaggingClassifier
    # model=BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=5)
    # model.fit(X_train, y_train.values.ravel())
    mnb = MultinomialNB().fit(X_train, y_train)
    lr=LogisticRegression(max_iter=1000)
    svm=LinearSVC(C=0.0001)
    adb = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=10,max_depth=4),n_estimators=15,learning_rate=0.001)
    bc = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=15)

    model=VotingClassifier(estimators=[('mnb',mnb),('lr',lr),('bc',bc),('svm',svm),('adb', adb)],voting='hard')
    model.fit(X_train, y_train.values.ravel())
    #from PIL import Image
    #import requests
    # Evaluate model performance using the features from the test set (X_test)
    #res = model.predict(X_test)
    #survivor_url = "https://raw.githubusercontent.com/Chaouo/Titanic_serverless_ML/main/image/"+ str(res[0]) + ".png"
    #img = Image.open(requests.get(survivor_url, stream=True).raw)
    #img.save("./test.png")
    y_pred = model.predict(X_test)
    #print(y_pred)
    # Compare predictions (y_pred) with the labels in the test set (y_test)
    metrics = classification_report(y_test, y_pred, output_dict=True)
    results = confusion_matrix(y_test, y_pred)
    print("score on test: "  + str(model.score(X_test, y_test)))
    print("score on train: " + str(model.score(X_train, y_train)))
    # Create the confusion matrix as a figure, we will later store it as a PNG image file
    # df_cm = pd.DataFrame(results, ['True Survival', 'True Dead'],
    #                      ['Pred Survival', 'Pred Dead'])
    # cm = sns.heatmap(df_cm, annot=True)
    # fig = cm.get_figure()

    # We will now upload our model to the Hopsworks Model Registry. First get an object for the model registry.
    mr = project.get_model_registry()
    
    # The contents of the 'iris_model' directory will be saved to the model registry. Create the dir, first.
    model_dir="titan_model"
    if os.path.isdir(model_dir) == False:
        os.mkdir(model_dir)

    # Save both our model and the confusion matrix to 'model_dir', whose contents will be uploaded to the model registry
    joblib.dump(model, model_dir + "/titan_model.pkl")
    #fig.savefig(model_dir + "/confusion_matrix.png")    


    # Specify the schema of the model's input/output using the features (X_train) and labels (y_train)
    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema, output_schema)

    # Create an entry in the model registry that includes the model's name, desc, metrics
    titan_model = mr.python.create_model(
        name="titan_modal", 
        metrics={"accuracy" : metrics['accuracy']},
        model_schema=model_schema,
        description="Titanic Survival Predictor"
    )
    
    # Upload the model to the model registry, including all files in 'model_dir'
    titan_model.save(model_dir)
    
if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
