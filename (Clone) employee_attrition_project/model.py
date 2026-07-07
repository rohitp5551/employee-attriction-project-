from preprocessing import preprocess_data
x,y=preprocess_data()
from sklearn.preprocessing import LabelEncoder,StandardScaler
from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#Building Model
def Build_model(model,X_train,Y_train,X_test,Y_test):
  model.fit(X_train,Y_train)
  y_test_pred=model.predict(X_test)
  Model_score(y_test_pred,Y_test,model)

def Model_score(y_pred,y_test,model):
  Accuracy_Score=accuracy_score(y_pred,y_test)*100
  Precision_Score=precision_score(y_pred,y_test)*100
  Recall_Score=recall_score(y_pred,y_test)*100
  F1_Score=f1_score(y_pred,y_test)*100


  print(f"Accuracy Score of {model}= {Accuracy_Score}")
  print(f"Precision_Score of {model}= {Precision_Score}")
  print(f"Recall_Score of {model}= {Recall_Score}")
  print(f"F1_Score of {model}= {F1_Score}")

#Logistic Regression
lr=LogisticRegression()
Build_model(lr,x_train,y_train,x_test,y_test)
#Random Forest Classifier
rf=RandomForestClassifier(n_estimators=10)
Build_model(rf,x_train,y_train,x_test,y_test)
#Decision Tree
decision_tree=DecisionTreeClassifier()
Build_model(decision_tree,x_train,y_train,x_test,y_test)
#XGBoost
xgb=XGBClassifier()
Build_model(xgb,x_train,y_train,x_test,y_test)
import joblib
joblib.dump(rf,"random_forest_model.pkl")
print("model saved successfully!")
# Git test