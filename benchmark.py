from NiaPy.benchmarks import Benchmark


class ClassificationBenchmark(Benchmark):
    def __init__(self, model_fn, eval_fn, x_train, y_train, x_test, y_test):
        super().__init__(0, 1)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model_fn = model_fn
        self.eval_fn = eval_fn

    def get_length(self):
        return self.x_train.shape[1]

    def select_columns(self, solution_vec):
        return self.x_train.columns[solution_vec >= 0.5].tolist()

    def function(self):
        def evaluate(solution):
            selected_columns = self.select_columns(solution)

            if len(selected_columns) == 1 and not isinstance(selected_columns[0], str):
                selected_columns = selected_columns[0]

            if not selected_columns:
                return 1

            clf = self.model_fn()
            clf = clf.fit(self.x_train[selected_columns], self.y_train)

            y_pred = clf.predict(self.x_test[selected_columns])
            score = self.eval_fn(self.y_test, y_pred)

            return 1 - score

        return evaluate
