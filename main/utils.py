class Avg:
    def __init__(self):
        self.sum = 0.
        self.n = 0
    def add(self, v: float):
        self.sum += v
        self.n += 1
    def avg(self):
        return self.sum/self.n