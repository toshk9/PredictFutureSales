from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from sklearn.metrics import root_mean_squared_error


class HyperparameterOpt:
    def __init__(self, model, space):
        self.model = model
        self.space = space
        self.best_params = None

    def objective(self, params: dict, add_params: dict, X_train, y_train, X_test, y_test) -> dict:
        model = self.model(**params, **add_params)        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, pred)
        return {'RMSE loss': rmse, 'status': STATUS_OK}
    
    def tpe_hyperopt(self):
        trials = Trials()
        best = fmin(
            fn=self.objective,
            space=self.space,
            algo=tpe.suggest,
            max_evals=100, 
            trials=trials
        )

        self.best_params = space_eval(self.space, best)
        return self.best_params
