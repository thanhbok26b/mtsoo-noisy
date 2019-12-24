from .functions import *
from scipy.io import loadmat
import os

DIRNAME = os.path.dirname(__file__)

class CI_HS:

    def __init__(self):
        mat = loadmat(os.path.join(DIRNAME, 'data/CI_H.mat'))
        self.M1 = mat['Rotation_Task1']
        self.M2 = mat['Rotation_Task2']
        self.functions = [self.f1, self.f2]
        self.dim = 50

    def f1(self, x):
        return moderate_noise(griewank(self.M1 @ (x * 200 - 100)))

    def f2(self, x):
        return moderate_noise(rastrigin(self.M2 @ (x * 100 - 50)))

class CI_MS:

    def __init__(self):
        mat = loadmat(os.path.join(DIRNAME, 'data/CI_M.mat'))
        self.M1 = mat['Rotation_Task1']
        self.M2 = mat['Rotation_Task2']
        self.functions = [self.f1, self.f2]
        self.dim = 50

    def f1(self, x):
        return moderate_noise(ackley(self.M1 @ (x * 100 - 50)))

    def f2(self, x):
        return moderate_noise(rastrigin(self.M2 @ (x * 100 - 50)))

class CI_LS:

    def __init__(self):
        mat = loadmat(os.path.join(DIRNAME, 'data/CI_L.mat'))
        self.M1 = mat['Rotation_Task1']
        self.O1 = mat['GO_Task1'][0]
        self.functions = [self.f1, self.f2]
        self.dim = 50

    def f1(self, x):
        return moderate_noise(ackley(self.M1 @ (x * 100 - 50 - self.O1)))

    def f2(self, x):
        return moderate_noise(schwefel(x * 1000 - 500))

class NI_HS:

    def __init__(self):
        mat = loadmat(os.path.join(DIRNAME, 'data/NI_H.mat'))
        self.O1 = np.ones([50])
        self.M2 = mat['Rotation_Task2']
        self.functions = [self.f1, self.f2]
        self.dim = 50

    def f1(self, x):
        return moderate_noise(rosenbrock(x * 100 - 50 - self.O1))

    def f2(self, x):
        return moderate_noise(rastrigin(self.M2 @ (x * 100 - 50)))

class NI_MS:

    def __init__(self):
        mat = loadmat(os.path.join(DIRNAME, 'data/NI_M.mat'))
        self.M1 = mat['Rotation_Task1']
        self.O1 = mat['GO_Task1'][0]
        self.M2 = mat['Rotation_Task2']
        self.functions = [self.f1, self.f2]
        self.dim = 50

    def f1(self, x):
        return moderate_noise(griewank(self.M1 @ (x * 200 - 100 - self.O1)))

    def f2(self, x):
        return moderate_noise(weierstrass(self.M2 @ (x - 0.5)))
