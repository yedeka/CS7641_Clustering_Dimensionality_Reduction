from data.deposit_data_loader import load_cleanse_data
from data.income_evaluation_loader import loadData
from dr import pca


def performDepositPCA():
    data = load_cleanse_data()
    pca.perform_pca(data['features'], 'Term_Deposit')

def performIncomePCA():
    data = loadData()
    pca.perform_pca(data['features'], 'Income_Analysis')

def performPCA():
    performDepositPCA()
    performIncomePCA()