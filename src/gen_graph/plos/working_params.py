class WorkingParams():
    weights: float

    def __init__(self, weights: float):
        self.weights = weights

    def __str__(self):
        return f"weights={self.weights}"

    def scale(self, ratio):
        self.weights *= ratio
