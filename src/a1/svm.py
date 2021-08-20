# Author: Wuli Zuo, a1785343
# Date: 2020-08-12 17:41


from sklearn import svm
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Perceptron



# function of training svm with sklearn
def sk(X_train, y_train, X_test, y_test):

    print('\n## Grid search: decide optimal C for sklearn SVM')
    Cs = [0.1, 1, 10, 100]
    tuned_parameters = [{'C': Cs}]

    # use GridSearchCV，to search for best parameters for SVC()
    grid_search = GridSearchCV(svm.SVC(kernel='linear', decision_function_shape='ovo'), tuned_parameters, cv=5)
    # train
    grid_search.fit(X_train, y_train.ravel())
    scores = grid_search.cv_results_['mean_test_score']
    params = grid_search.cv_results_['params']

    # print best parameter and best score
    best_C = grid_search.best_params_['C']
    print('\n* Best parameters:', grid_search.best_params_)
    print('* Best score: %.2f%%' % (100 * grid_search.best_score_))

    # train svm classifier with best C
    # kernel='linear'，ovo: one vs. one
    svm_model = svm.SVC(C=best_C, kernel='linear', decision_function_shape='ovo')
    svm_model.fit(X_train, y_train.ravel())

    # predict
    acc_train = svm_model.score(X_train, y_train)
    acc_test = svm_model.score(X_test, y_test)
    print('training accuracy = %f, test accuracy = %f' % (acc_train, acc_test))