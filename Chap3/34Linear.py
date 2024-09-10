import torch as t
import utils

class LinearRegressionScratch(utils.HyperParameters):
    def __init__(self, num_inputs, lr, sigma = 0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = t.normal(0, sigma, (num_inputs, 1), requires_grad = True)
        self.b = t.zeros(1, requires_grad = True)
        
    def forward(self, X):
            return t.matmul(X, self.w) + self.b

    def loss(self, y_hat, y):
            l = (y_hat - y) ** 2 / 2
            return l.mean()
        
    def configure_optimizers(self):
            return SGB([self.w, self.b], self.lr)

    def training_step(self, batch):
            l = self.loss(self(*batch[:-1]), batch[-1])
            return l

    def validation_step(self, batch):
            l = self.loss(self(*batch[:-1]), batch[-1])
            return l

class SyntheticRegressionData(utils.HyperParameters):
    def __init__(self, root = '../data', num_workers = 4, w = 0, b = 0, noise = 0.01, num_train = 1000, num_val = 1000):
        self.save_hyperparameters()
        n = num_train + num_val
        self.x = t.randn(n, len(w))
        noise = t.randn(n, 1) * noise
        self.y = t.matmul(self.x, w.reshape((-1, 1))) + b + noise

    def get_dataloader(self, train):
        if train:
            indices = list(range(0, self.num_train))
        else:
            indices = list(range(self.num_train, self.num_train + self.num_val))
        for i in range(0, len(indices), self.batch_size):
            batch_indices = t.tensor(indices[i: i + self.batchsize])
            yield self.x[batch_indices], self.y[batch_indices]

    def train_dataloader(self):
        return self.get_dataloader(train = True)

    def val_dataloader(self):
        return self.get_dataloader(train = False) 



class SGD(utils.HyperParameters):
    def __init__(self, params, lr):
        self.save_hyperparameters()

    def step(self):
        for param in self.params:
            params -= self.lr * param.grad
        
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_() 

Data = SyntheticRegressionData(w = t.tensor([2, -4.2]), b = 4.2)
model = LinearRegressionScratch(num_inputs)
for batch in Data.train_dataloader:
    print(batch)
    exit()
    pred = model.forward(batch)
    
