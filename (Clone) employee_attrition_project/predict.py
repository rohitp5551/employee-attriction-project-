import joblib
from preprocessing import preprocess_data
model = joblib.load("random_forest_model.pkl")
x,y=preprocess_data()
prediction = model.predict(x)
print(prediction)