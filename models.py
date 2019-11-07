# models imports
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score
import sklearn.svm as svm

def _roc_auc_scorer(estimator, X, y):
    
    y_pred = estimator.predict(X)
    
    return roc_auc_score(y, y_pred)

def _get_best_model(X_train, y_train):

    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gammas = [0.001, 0.01, 0.1, 1, 5, 10, 100]
    param_grid = {'kernel':['rbf'], 'C': Cs, 'gamma' : gammas}

    svc = svm.SVC(probability=True)
    clf = GridSearchCV(svc, param_grid, cv=10, scoring=_roc_auc_scorer, verbose=0)

    clf.fit(X_train, y_train)

    model = clf.best_estimator_

    return model

def svm_model(X_train, y_train):

    return _get_best_model(X_train, y_train)

def rfe_svm_model(X_train, y_train, n_components=1):

    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gammas = [0.001, 0.01, 0.1, 1, 5, 10, 100]
    param_grid = [{'estimator__C': Cs, 'estimator__gamma' : gammas}]

    estimator = svm.SVC(kernel="linear")
    selector = RFECV(estimator, step=1, cv=4, verbose=1)
    clf = GridSearchCV(selector, param_grid, cv=5, verbose=1, scoring=_roc_auc_scorer)
    clf.fit(X_train, y_train)

    return clf.best_estimator_


def get_trained_model(choice, X_train, y_train):

    if choice == 'svm_model':
        return svm_model(X_train, y_train)

    if choice == 'rfe_svm_model':
        return rfe_svm_model(X_train, y_train)