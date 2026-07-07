from load_data import load_data
def preprocess_data():
    df=load_data()
    #import libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    import warnings
    warnings.filterwarnings("ignore")


    from sklearn.preprocessing import LabelEncoder,StandardScaler
    from imblearn.over_sampling import RandomOverSampler

    df.head()
    df.shape
    df.info
    df.isnull().sum().sum()
    df.duplicated().sum()
    df.describe().T
    df.columns
    df.nunique()
    Employee_id=df['EmployeeNumber']
    df.drop(["Over18","StandardHours","EmployeeCount","EmployeeNumber"],axis=1,inplace=True)
    df.shape
    # Categorical Columns in dataset
    df_categorical=df.select_dtypes(include="object").columns
    print(df_categorical)
    print("_"*75)
    print("Numbers of category=",len(df_categorical))
    #Numerical Columns in Dataset
    df_numerical=df.select_dtypes(include="int").columns
    print(df_numerical)
    print("_"*75)
    print("number of numerical =",len(df_numerical))
    #Chart -1 (Distribution plot of numerical variables)
    for num_col in df_numerical:
        plt.figure(figsize=(8,5))
        sns.histplot(data=df,x=num_col,kde=True)
    #Chart-2 Correlation Matrix(Heatmap)
    plt.figure(figsize=(17,10))
    sns.heatmap(df[df_numerical].corr(),annot=True,annot_kws={'size':10})
    plt.title("Correlation Matrix of Numerical Columns")
    plt.show()
    df["Attrition"]=df["Attrition"].apply(lambda x:1 if x=="Yes" else 0)
    #Chart-3 Categorical Feature Vs Attrition
    for col in df_categorical:
        plt.figure(figsize=(10,5))
        sns.barplot(data=df,x=col,y="Attrition")
        plt.xlabel(col)
        plt.ylabel("Attrition")
        plt.show()
    #3.Data Preprocessing
    #Encoding
    df['Gender'] = df['Gender'].apply(lambda x: 1 if  x== 'Male' else 0)
    df['OverTime'] = df['OverTime'].apply(lambda x: 1 if  x== 'Yes' else 0)
    #LabelEncoding
    le = LabelEncoder()
    columns_to_encode = ["BusinessTravel", "EducationField", "Department", "JobRole", "MaritalStatus"]
    for column in columns_to_encode:
        df[column] = le.fit_transform(df[column])
    df.head()
    #Checking Datatype
    df.info()
    df.shape
    new_df=df.copy()
    new_df.shape
    sns.countplot(data=new_df,x="Attrition")
    plt.show()
    X = new_df.drop('Attrition', axis=1)
    Y = new_df['Attrition']
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, Y)

    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled['Attrition'] = y_resampled

    new_df = df_resampled
    sns.countplot(data=new_df,x="Attrition")
    plt.show()
    new_df.shape
    #Scaling the Numerical Variables
    skewed_columns=['DistanceFromHome','JobLevel','MonthlyIncome','NumCompaniesWorked','PercentSalaryHike','PerformanceRating','StockOptionLevel','TotalWorkingYears','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    new_df[skewed_columns] = scaler.fit_transform(new_df[skewed_columns])
    new_df.head()
    new_df.shape
    x=new_df.drop("Attrition",axis=1)
    y=new_df["Attrition"]
    return x,y