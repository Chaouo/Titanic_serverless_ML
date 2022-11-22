import os
import modal
import numpy as np
import pandas as pd
from IPython.display import display
import random
import hopsworks

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("titanic_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def generate_passenger(survived):
    """
    Returns a single iris flower as a single row in a DataFrame
    """

    if survived == 1:
        pclass = np.random.choice(np.arange(1, 4), p=[0.42, 0.3, 0.28])
        sex = np.random.choice(np.arange(0, 2), p=[0.32, 0.68])
        family = np.random.choice(np.arange(0, 7), p=[0.46, 0.26, 0.18, 0.07, 0.01, 0.01, 0.01])
        age_group = np.random.choice(np.arange(0, 4), p=[0.28, 0.53, 0.17, 0.02])
        if age_group == 0:
            age = np.random.randint(1, 20)
        elif age_group == 1:
            age = np.random.randint(21, 40)
        elif age_group == 2:
            age = np.random.randint(41, 60)
        else:
            age = np.random.randint(61, 80)

        fare_group = np.random.choice(np.arange(0, 5), p=[0.16, 0.27, 0.23, 0.21, 0.13])
        if fare_group == 0:
            fare = np.random.randint(1, 10)
        if fare_group == 1:
            fare = np.random.randint(11, 25)
        if fare_group == 2:
            fare = np.random.randint(26, 50)
        if fare_group == 3:
            fare = np.random.randint(51, 100)
        else:
            fare = np.random.randint(101, 300)

        df = pd.DataFrame({"Pclass": [pclass], "Sex": [sex], "Age": [float(age)], "Fare": [float(fare)], "Family": [family]})
        df["Survived"] = 0

    elif survived == 0:
        pclass = np.random.choice(np.arange(1, 4), p=[0.15, 0.21, 0.64])
        sex = np.random.choice(np.arange(0, 2), p=[0.85, 0.15])
        family = np.random.choice(np.arange(0, 7), p=[0.65, 0.16, 0.09, 0.02, 0.02, 0.04, 0.02])
        age_group = np.random.choice(np.arange(0, 4), p=[0.23, 0.55, 0.18, 0.04])
        if age_group == 0:
            age = np.random.randint(1, 20)
        elif age_group == 1:
            age = np.random.randint(21, 40)
        elif age_group == 2:
            age = np.random.randint(41, 60)
        else:
            age = np.random.randint(61, 80)

        fare_group = np.random.choice(np.arange(0, 5), p=[0.44, 0.26, 0.21, 0.07, 0.02])
        if fare_group == 0:
            fare = np.random.randint(1, 10)
        if fare_group == 1:
            fare = np.random.randint(11, 25)
        if fare_group == 2:
            fare = np.random.randint(26, 50)
        if fare_group == 3:
            fare = np.random.randint(51, 100)
        else:
            fare = np.random.randint(101, 300)

        df = pd.DataFrame({"Pclass": [pclass], "Sex": [sex], "Age": [float(age)], "Fare": [float(fare)], "Family": [family]})
        df["Survived"] = 1

    display(df)
    return df


def get_random_passenger():
    """
    Returns a DataFrame containing one random sample
    """

    survivor_df = generate_passenger(1)
    non_survivor_df = generate_passenger(0)

    # randomly pick one of these 2 and write it to the featurestore
    pick_random = random.uniform(0,2)
    if pick_random >= 1:
        passenger_df = survivor_df
        print("survivor added")
    else:
        passenger_df = non_survivor_df
        print("victim added")

    return passenger_df


def g():

    project = hopsworks.login()
    fs = project.get_feature_store()

    passenger_df = get_random_passenger()

    titanic_fg = fs.get_feature_group(name="titanic_modal",version=3)
    titanic_fg.insert(passenger_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":

    if LOCAL == True :
        g()
    else:
        stub.deploy("titanic_daily")
        with stub.run():
            f()