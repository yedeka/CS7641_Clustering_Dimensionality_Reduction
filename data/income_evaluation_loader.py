import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def loadExploreDS():
    incomeDF = pd.read_csv("data/income_evaluation.csv")
    return incomeDF

def preprocess_data(incomeDF):
     columns = incomeDF.columns
     for col in incomeDF.columns:
         col_stripped = col.strip()
         incomeDF = incomeDF.rename(columns={col: col_stripped})
     for column in incomeDF[["education", "marital-status", "occupation", "race", "relationship", "sex", "workclass"]]:
         incomeDF[column] = incomeDF[column].str.strip()
     le = LabelEncoder()

     incomeDF['education'] = le.fit_transform(incomeDF['education'])
     incomeDF['marital-status'] = le.fit_transform(incomeDF['marital-status'])
     incomeDF['occupation'] = le.fit_transform(incomeDF['occupation'])
     incomeDF['race'] = le.fit_transform(incomeDF['race'])
     incomeDF['relationship'] = le.fit_transform(incomeDF['relationship'])
     incomeDF['sex'] = le.fit_transform(incomeDF['sex'])
     incomeDF['workclass'] = le.fit_transform(incomeDF['workclass'])
     incomeDF['native-country'] = le.fit_transform(incomeDF['native-country'])
     incomeDF['income'] = le.fit_transform(incomeDF['income'])


     return incomeDF

def loadData():
    incomeDF = loadExploreDS()
    processedData = preprocess_data(incomeDF)
    features = processedData.drop(['income'], axis=1)
    output = processedData['income']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return {
        'features': scaled_features,
        'labels': output
    }
