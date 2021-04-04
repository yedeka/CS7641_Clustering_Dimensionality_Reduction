from data.deposit_data_loader import load_cleanse_data
from data.income_evaluation_loader import loadData
from dr import pca,ica, randomprojection

def performDepositPCA():
    data = load_cleanse_data()
    pca.perform_pca(data['features'], 'Term_Deposit')

def performIncomePCA():
    data = loadData()
    pca.perform_pca(data['features'], 'Income_Analysis')

def performDepositICA():
    data = load_cleanse_data()
    ica.perform_ica(data['features'], 'Term_Deposit')

def performIncomeICA():
    data = loadData()
    ica.perform_ica(data['features'], 'Income_Analysis')

def performDepositRandomProjection():
    data = load_cleanse_data()
    randomprojection.apply_rp(data['features'], 10,'Term_Deposit')

def performDR():
    # performDepositPCA()
    # performIncomePCA()
    performDepositICA()
    performIncomeICA()
    #performDepositRandomProjection()