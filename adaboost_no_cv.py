### TEST FUNCTION: test_parameter_tuning_no_cv
# DO NOT REMOVE THE LINE ABOVE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def adaBoostGrid(X, y, **hyperparams):
     """Fill this function to run the Adaboost grid search on linear SVM without cross-validation as described above"""
     X_train_all, X_train, X_val, X_test, y_train_all, y_train, y_val, y_test = train_val_test_split(X, y)
     best_val_score = -1
     best_params = {}
     default_params = {
        'algorithm': 'SAMME',
        'C': 1,
        'dual': True,
        'learning_rate': 1.0,
        'n_estimators': 50,
        'tol': 1e-4,
        'penalty': 'l2'
        }
     for key in default_params.keys():
        if key in hyperparams:
            default_params[key] = hyperparams[key][0] if isinstance(hyperparams[key], list) else hyperparams[key]

     for C in hyperparams.get('C', [default_params['C']]):
        for dual in hyperparams.get('dual', [default_params['dual']]):
            for tol in hyperparams.get('tol', [default_params['tol']]):
                for penalty in hyperparams.get('penalty', [default_params['penalty']]):
                    # dual = False if penalty == 'l1' else default_params['dual']
                    estimator = LinearSVC(C=C, dual=dual, tol=tol, penalty=penalty, random_state=0)
                    ada_clf = AdaBoostClassifier(
                        estimator=estimator,
                        algorithm=default_params['algorithm'],
                        learning_rate=default_params['learning_rate'],
                        n_estimators=default_params['n_estimators'],
                        random_state=0
                    )

                    ada_clf.fit(X_train, y_train)
                    val_score = accuracy_score(y_val, ada_clf.predict(X_val))

                    if val_score > best_val_score:
                        best_val_score = val_score
                        best_params = {
                            'C': C,
                            'dual': dual,
                            'tol': tol,
                            'penalty': penalty,
                            'algorithm': default_params['algorithm'],
                            'learning_rate': default_params['learning_rate'],
                            'n_estimators': default_params['n_estimators']
                        }
  
     estimator = LinearSVC(
        C=best_params['C'],
        dual=best_params['dual'],
        tol=best_params['tol'],
        penalty=best_params['penalty'],
        random_state=0
    )
     best_ada_clf = AdaBoostClassifier(estimator=estimator, algorithm=best_params['algorithm'], learning_rate=best_params['learning_rate'], n_estimators=best_params['n_estimators'], random_state=0)

     best_ada_clf.fit(X_train_all, y_train_all)

   
     test_score = accuracy_score(y_test, best_ada_clf.predict(X_test))

     print("Best hyperparameter combination:", best_params)
     print(f"Best model's test set score: {test_score:.2f}")
     
     return best_ada_clf, best_params, best_val_score, test_score

param_grid = {
    'algorithm': ['SAMME'],
    'C': [1000, 1100],
    'dual': ['auto', False],
    'learning_rate': [1e-9, 1e-2],
    'n_estimators': [1,5],
    'tol': [0.01,0.1],
    'penalty': ['l2', 'l1']
    
}
best_model, best_params, best_val_score, test_score = adaBoostGrid(X_norm, y_encoded, **param_grid)



