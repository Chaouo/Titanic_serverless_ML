import os
import modal
    
LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4","joblib","seaborn","scikit-learn","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    status_dict = {0: "dead", 1: "survived"}

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("titan_modal", version=59)
    model_dir = model.download()
    model = joblib.load(model_dir + "/titan_model.pkl")
    
    feature_view = fs.get_feature_view(name="titanic_modal", version=3)
    print(feature_view)
    batch_data = feature_view.get_batch_data()
    
    y_pred = model.predict(batch_data)
    #print(y_pred)
    offset = 1
    res = y_pred[y_pred.size-offset]
    res_url = "https://raw.githubusercontent.com/Chaouo/Titanic_serverless_ML/main/image/"+ str(res) + ".png"
    print("Passenger predicted status: " + status_dict[res])
    img = Image.open(requests.get(res_url, stream=True).raw)            
    img.save("./latest_passenger_result.png")
    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./latest_passenger_result.png", "Resources/images", overwrite=True)
   
    titan_fg = fs.get_feature_group(name="titanic_modal", version=3)
    df = titan_fg.read() 
    #print(df)
    label = df.iloc[-offset]["survived"]
    label_url = "https://raw.githubusercontent.com/Chaouo/Titanic_serverless_ML/main/image/"+ str(int(label)) + ".png"
    print(str(int(label)))
    print("Passenger actual status: " + status_dict[label])
    img = Image.open(requests.get(label_url, stream=True).raw)            
    img.save("./actual_passenger_status.png")
    dataset_api.upload("./actual_passenger_status.png", "Resources/images", overwrite=True)
    
    monitor_fg = fs.get_or_create_feature_group(name="passener_status_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Passenger Status Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [res],
        'label': [label],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])


    df_recent = history_df.tail(4)
    dfi.export(df_recent, './df_recent.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our iris_predictions feature group has examples of all 3 iris flowers
    print("Number of different passenger status predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 2:
        results = confusion_matrix(labels, predictions)
    
        df_cm = pd.DataFrame(results, ['True Dead', 'True Survived'],
                             ['Pred Dead', 'Pred Survived'])
    
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix.png")
        dataset_api.upload("./confusion_matrix.png", "Resources/images", overwrite=True)
    else:
        print("You need 2 different flower predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 2 different predictions") 


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

