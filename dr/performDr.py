from data.deposit_data_loader import load_cleanse_data
from data.income_evaluation_loader import loadData
from dr import pca,ica, randomprojection
from sklearn.feature_selection import RFE

def performDepositPCA():
    data = load_cleanse_data()
    pca.perform_pca(data['features'], 'deposit')
    pca.validate_pca_nn(data, [7, 10, 15, 20, 25, 30, 35, 40, 41], 'deposit')

def performIncomePCA():
    data = loadData()
    pca.perform_pca(data['features'], 'income')
    pca.validate_pca_nn(data, [4, 6, 8, 10, 12,14], 'income')

def performDepositICA():
    data = load_cleanse_data()
    ica.perform_ica(data['features'], 'deposit', [7, 10, 15, 20, 25, 30, 35, 40, 41])
    ica.validate_ica_nn(data, [7, 10, 15, 20, 25, 30, 35, 40, 41], 'deposit')

def performIncomeICA():
    data = loadData()
    ica.perform_ica(data['features'], 'income', [4, 6, 8, 10, 12, 14])
    ica.validate_ica_nn(data, [4, 6, 8, 10, 12, 14], 'income')

def performDepositRandomProjection():
    data = load_cleanse_data()
    randomprojection.apply_rp(data['features'], 10,'deposit')

def performDR():
    #performDepositPCA()
    #performIncomePCA()
    performDepositICA()
    performIncomeICA()
    # performDepositRandomProjection()