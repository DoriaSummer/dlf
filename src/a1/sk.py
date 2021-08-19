# Author: Wuli Zuo, a1785343
# Date: 2020-08-12 17:41


from sklearn import svm
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Perceptron



# function of training svm with sklearn
def sk(X_train, y_train, X_test, y_test):
    '''
    print('\n## Grid search: decide optimal C for sklearn SVM')
    Cs = [0.1, 1, 10, 100]
    tuned_parameters = [{'C': Cs}]

    # use GridSearchCV，to search for best parameters for SVC()
    grid_search = GridSearchCV(svm.SVC(kernel='linear', decision_function_shape='ovo'), tuned_parameters, cv=5)
    # train
    grid_search.fit(data_train, label_train.ravel())
    scores = grid_search.cv_results_['mean_test_score']
    params = grid_search.cv_results_['params']
    '''
    '''
    for score, param in zip(scores, params):
        print('## K-fold cross validation，', param)
        print('     mean score: %.2f%%' % float(100 * score))
    '''
    '''
    # print best parameter and best score
    best_C = grid_search.best_params_['C']
    print('\n* Best parameters:', grid_search.best_params_)
    print('* Best score: %.2f%%' % (100 * grid_search.best_score_))


    # train svm classifier with best C
    # kernel='linear'，ovo: one vs. one
    svm_model = svm.SVC(C=best_C, kernel='linear', decision_function_shape='ovo')
    svm_model.fit(data_train, label_train.ravel())
    '''
    # define perceptron
    model_sk = Perceptron(
        fit_intercept=False,  # 不计算偏置
        shuffle=False  # 在每个epoch重新打乱洗牌
    )
    # train
    model_sk.fit(X_train, y_train)
    '''
    # output w and b
    w = model_sk.coef_
    b = model_sk.intercept_
    print('w = ', w)
    print('b = ', b)
    '''
    # predict
    result = model_sk.score(X_train, y_train)
    print('training accuracy： ', result)
    result = model_sk.score(X_test, y_test)
    print('test accuracy： ', result)

    # decision function
    # decision_f_param = sk_svm_model.decision_function(data_train)
    # print('\nDecision function:\n', decision_f_param)

    #classifier = np.vstack((svm_model.coef_.T, svm_model.intercept_))
    # print('\n* w, b: ')
    # print(classifier[::, :1])

    # save svm as a file for analysis
    # np.savetxt('../../output/svm_model_sk', classifier[::, :1], fmt="%.16f", delimiter=',')

    #return svm_model #, classifier
