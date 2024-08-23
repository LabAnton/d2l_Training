import time 
import numpy as np
import torch as t
from torch import nn
import collections
import inspect
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline

 1 import time
  2 import numpy as np
  3 import torch as t
  4 from torch import nn
  5 import collections
  6 import inspect
  7 import matplotlib.pyplot as plt
  8 from matplotlib_inline import backend_inline
class HyperParameters:
    def save_hyperparameters(self, ignore=[]):
        #Save function arguments into class attributes
        frame = inspect.currentframe().f_back #find the callers frame -> fetches variables from class that calls save_hyperparameters
        _, _, _, local_vars = inspect.getargvalues(frame) #get values from callers frame
        self.hparams = {k:v for k,v in local_vars.items() if k not in set(ignore+["self"]) and not k.startswith('_')} #cannot overwrite inheret functions starting with __
        for k, v in self.hparams.items():
            setattr(self, k, v) #if object has __dict__() method then new variable can be given to class


class B(HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a = ', self.a, 'self.b = ', self.b) 
        print('There is no self.c = ', not hasattr(self, 'c'))

b = B(a=1, b=2, c=3)

class ProgressBoard(HyperParameters):
    def __init__(self, xlabel = None, ylabel = None, xlim = None, ylim = None, xscale = 'linear', yscale = 'linear', ls = ['-', '--', '-.', ':'], colors = ['C0', 'C1', 'C2', 'C3'], fig = None, axes = None, figsize = (3.5, 2.5), display = True):
        self.save_hyperparameters()
    
    def draw(self, x, y, label, every_n=1):
        Point = collections.namedtuple('Point', ['x', 'y'])
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]), mean([p.y for p in points])))
        points.clear()
        if not self.display:
            return
        backend_inline.set_matplotlib_formats('png')
        if self.fig is None:
            self.fig = plt.figure(figsize = self.figsize)
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(plt.plot([p.x for p in v], [p.y for p in v], linestyle = ls, color = color)[0])
            labels.append(k)

        #Figure specifications
        axes = self.axes if self.axes else plt.gca()
        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        if not self.xlabel: self.xlabel = self.x
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        plt.show()
        plt.pause(0.001) #Displays only for the pause time

board = ProgressBoard('x', 'Test')
plt.ion()
for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), 'sin', every_n=2)
    board.draw(x, np.cos(x), 'cos', every_n=10)
    
class Module(nn.Module, HyperParameters):
    def __init_(self, plot_train_per_epoch = 2, plot_valid_per_epoch = 1):
        super().__init__()
        self.save_hyperparameters()
        self.board = Prograssboard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net')
        return self.net(X)

    def plot(self, key, value, train):
        assert hasattr(self, 'trainer')
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / self.trainer.num_train_batches
            n = self.trainer.num_train_batches / self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / self.plot_valid_per_epoch
        self.board.draw(x, value.detach().cpu().numpy(), ('train_' if train else "val_") + key, every_n = int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train = True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError 

class DataModule(HyperParameters):
    def __init__(self, root = '../data', num_workers = 4):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train = True)

    def val_dataloader(self):
        return self.get_dataloader(train = False)

class Trainer(HyperParameters):
    def __init__(self, max_epochs, num_gpus = 0, gradient_clip_val = 0):
        self.save_hyperparameters()
        assert num_gpus ==0

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader) if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model
    
    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
    
    def fit_epoch(self):
        raise NotImplementedError

#Exercises
#2. Remove the save_hyperparameters statement in the B class. Can you still print self.a and self.b ? Optional: If you have dived into the full implementation of the HyperParameters class, can you explain why?
# No it cannot print self.a and self.b because at no point are they initiliased which is done in the HyperParameters class.
