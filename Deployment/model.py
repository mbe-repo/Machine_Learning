import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify, render_template
import pickle



# load csv
df = pd.read_csv("winequality.csv")

#clean
df = df.dropna()
del df["type"]


# Features & Target
y = df["quality"]
X = df.drop(["quality"], axis=1)

# Split the dataset  train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#  Feature scaling & Imputing , instancier modèle
sc = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler())])

X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

# Instancier le modele
classifier = RandomForestClassifier()

# Fit the model
classifier.fit(X_train, y_train)

print(print("Accuracy: {:.2f}".format(classifier.score(X_test, y_test))))

# pickle file of our model
#pickle , enregistrez  le modèle sur le disque avec la fonction dump() 
# #et désélectionnez-le dans votre code python avec la fonction load()
pickle.dump(classifier, open("model.pkl", "wb"))