import pandas as pd

pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import yfinance as yf
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import accuracy_score, classification_report, roc_curve


def get_variables(ret):
    '''

    :param ret: The return dataframe
    :return: new data frame
    '''
    df = pd.DataFrame()
    df['lag1'] = ret['SSE Composite Index'].shift(1)
    df['lag2'] = ret['SSE Composite Index'].shift(2)
    for i in corrlist:
        df[str(i)] = ret[str(i)].shift(1)
    df['direction'] = np.where(ret['SSE Composite Index'].values > 0, 1, -1)
    df = df.dropna()
    return df


def fit_model(name, model, X_train, y_train, X_test, pred, bmd):
    """

    :param name: The name of model
    :param model: Model
    :param X_train: X_train
    :param y_train: y_train
    :param X_test: X_test
    :param bm: Best Model Dictionary
    :return: Directly get the model score
    """
    # Fit and predict the model on the training, and then test, data
    model.fit(X_train, y_train)
    pred[name] = model.predict(X_test)
    score = accuracy_score(pred['Actual'], pred[name])
    bmd[str(name)] = [score]
    print("%s Model: %.3f" % (name, score))


StockIndexs = {
    '000001.SS': 'SSE Composite Index',
    '^IXIC': 'NASDAQ Composite',
    '^HSI': 'HANG SENG INDEX',
    '^FVX': 'U.S. Treasury Yield 5 Years',
    '^TNX': 'Treasury Yield 10 Years',
    '^IRX': '13 Week Treasury Bill'}


print("Please input your choice of target stock index\n")
print('Choice should be one of ')
print(StockIndexs.values())
#target_index = input("choice: ")
target_index = 'SSE Composite Index'
print("Please input start date")
#start_date = input('start_date(yyyy-mm-dd):')
start_date = '2020-09-01'
print("Please input end date")
#end_date = input('end_date(yyyy-mm-dd):')
end_date = '2021-03-10'
data = pd.DataFrame()
for code, name in StockIndexs.items():
    print(code, name)
    data[name] = yf.download(code, start=start_date, end=end_date, interval="1d")['Close']
print(data)
data.to_csv('database/some_index(short).csv')
# data.set_index('Date', inplace=True)

data.fillna(method='ffill',inplace=True)
print(data)
(data / data.iloc[0]).plot(figsize=(14, 6))
plt.title('Cumulative net value of major global indexes return of rate\n', size=15)
plt.show()
ret = data.apply(lambda x: (x / x.shift(1) - 1) * 100).dropna()
sns.clustermap(ret.corr())
plt.title('Correlation of major global indexes', size=15)
plt.show()
corrlist = ret.corr()[str(target_index)].sort_values()[-5:-1].index.tolist()

variables = get_variables(ret)
print(variables)
y = variables['direction']
X = variables.drop(columns='direction')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pred = pd.DataFrame(index=y_test.index)
pred["Actual"] = y_test
# Use LoR, LDA and QDA models
print("Model Accuracy:")
models = [("LoR", LogisticRegression(solver='liblinear')),
          ("LDA", LDA()),
          ("QDA", QDA()),
          ("SVM", SVC(kernel='rbf')),
          ("DTC", DTC(max_depth=5)),
          ("KNN", knn(n_neighbors=8))]  # some of them need to select parms
bmd = {}
for m in models:
    fit_model(m[0], m[1], X_train, y_train, X_test, pred, bmd)

print(max(bmd, key=bmd.get))
bmn = max(bmd, key=bmd.get)
for m in models:
    if bmn in [m[0]]:
        model = m[1]
    else:
        pass

model.fit(X_train, y_train)
test_pred = model.predict(X_test)
cm = pd.crosstab(y_test, test_pred)
sns.heatmap(cm, annot=True, cmap='GnBu', fmt='d')
print('The best model accuracy：\n', accuracy_score(y_test, test_pred))
print('The best model report：\n', classification_report(y_test, test_pred))
y_score = model.predict_proba(X_test)[:, 1]
fpr, tpr, threshold = roc_curve(y_test, y_score)
roc_auc = metrics.auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.stackplot(fpr, tpr, color='steelblue', alpha=0.5, edgecolor='black')
plt.plot(fpr, tpr, color='black', lw=1)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.text(0.5, 0.3, 'ROC Curve (area = %0.2f)' % roc_auc)
plt.title('ROC Curve of the Best Model', size=15)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.show()
