import pandas as pd
from sklearn.preprocessing import StandardScaler

def loadBankDataSet():
    dataset = pd.read_csv("data/bank.csv");
    return dataset

def get_dummy_from_bool(row, column_name):
    return 1 if row[column_name] == 'yes' else 0

def get_correct_values(row, column_name, threshold, df):
    ''' Returns mean value if value in column_name is above threshold'''
    if row[column_name] <= threshold:
        return row[column_name]
    else:
        mean = df[df[column_name] <= threshold][column_name].mean()
        return mean

def clean_data(df):
    cleaned_df = df.copy()
    # Convert boolean columns into 0/1 columns
    truthy_clmns = ['default', 'deposit', 'housing', 'loan']
    for truthy_col in truthy_clmns:
        cleaned_df[truthy_col + '_bool'] = df.apply(lambda row: get_dummy_from_bool(row, truthy_col), axis=1)

    cleaned_df = cleaned_df.drop(columns=truthy_clmns)
    # Transform categorical columns into equivalent dummy values
    cat_columns = ['contact', 'education', 'job', 'marital', 'month', 'poutcome']
    for cat_col in cat_columns:
        cleaned_df = pd.concat([cleaned_df.drop(cat_col, axis=1),
                                pd.get_dummies(cleaned_df[cat_col], prefix=cat_col, prefix_sep='_',
                                               drop_first=True, dummy_na=False)], axis=1)

    # drop irrelevant columns
    '''cleaned_df = cleaned_df.drop(columns=['pdays'])

    # impute noisy columns
    cleaned_df['campaign_cleaned'] = df.apply(lambda row: get_correct_values(row, 'campaign', 34, cleaned_df), axis=1)
    cleaned_df['previous_cleaned'] = df.apply(lambda row: get_correct_values(row, 'previous', 34, cleaned_df), axis=1)

    cleaned_df = cleaned_df.drop(columns=['campaign', 'previous']) '''
    return cleaned_df

def load_cleanse_data():
    dataset = loadBankDataSet()
    preprocessed_data = clean_data(dataset)
    features = preprocessed_data.drop(['deposit_bool'], axis=1)
    output = preprocessed_data['deposit_bool']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_features = pd.DataFrame(scaled_features,columns = features.columns)
    return {
        'features': scaled_features,
        'labels': output
    }
