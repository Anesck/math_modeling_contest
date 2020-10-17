from random import sample

import numpy as np
import matplotlib.pyplot as plt


class SA():
    def __init__(self, temperature, solve, inner_maxiter, outer_maxiter, \
            annealing_func, newsolve_func, cost_func, \
            annealing_args=None, newsolve_args=None, cost_args=None, \
            inner_termination_func=lambda x: False, \
            outer_termination_func=lambda x: False):

        self.temperature = temperature
        self.solve = solve
        self.inner_maxiter = inner_maxiter
        self.outer_maxiter = outer_maxiter

        self.annealing = annealing_func
        self.newsolve = newsolve_func
        self.cost_func = cost_func

        self.annealing_args = annealing_args
        self.cost_args = cost_args
        self.newsolve_args = newsolve_args

        self.is_inner_termination = inner_termination_func
        self.is_outer_termination = outer_termination_func

        self.history = {"temperature": [], "solve": [], "cost": []}

    def simulated_annealing(self):
        self.cost = self.cost_func(self.solve, self.cost_args)
        self.history["temperature"].append(self.temperature)
        self.history["solve"].append(self.solve)
        self.history["cost"].append(self.cost)

        for outer in range(self.outer_maxiter):
            for inner in range(self.inner_maxiter):
                newsolve = self.newsolve(self.solve, self.newsolve_args)
                newcost = self.cost_func(newsolve, self.cost_args)
                prob = np.min([1, np.exp(-(newcost-self.cost)/self.temperature)])
                if np.random.random() < prob:
                    self.solve = newsolve
                    self.cost = newcost

                self.history["solve"].append(newsolve)
                self.history["cost"].append(newcost)
                if self.is_inner_termination(self.history):
                    break
            
            self.temperature = self.annealing(self.temperature, self.annealing_args)
            self.history["temperature"].append(self.temperature)
            if self.is_outer_termination(self.history):
                break

def get_newpath(oldpath, args=None):
    newpath = oldpath.copy()
    swap_indexes = sample(range(oldpath.shape[0]), 2)
    newpath[swap_indexes] = newpath[swap_indexes[::-1]]
    return newpath

def termination(history):
    len_cost = len(history["cost"]) - 5
    if len_cost >= 0:
        return np.std(history["cost"][len_cost:]) < 0.001
    return False

def myplot(path, coord):
    plt.plot(coord[0, :], coord[1, :], "ok")
    for i in range(-1, len(path)-1):
        plt.plot(coord[0, path[[i, i+1]]], coord[1, path[[i, i+1]]], "-b")
        x = np.sum(coord[0, path[[i, i+1]]]) / 2
        y = np.sum(coord[1, path[[i, i+1]]]) / 2
        if i == -1: i = 9
        plt.text(x, y, i+1)
    plt.plot(coord[0, path[[-1, 0]]], coord[1, path[[-1, 0]]], "-b")
    plt.show()

if __name__ == "__main__":
    coordinate = np.array([[0.6683, 0.6195, 0.4,    0.2439, 0.1707, 0.2293, 0.5171, 0.8732, 0.6878, 0.8488], \
                           [0.2536, 0.2634, 0.4439, 0.1463, 0.2293, 0.761,  0.9414, 0.6536, 0.5219, 0.3609]])
    
    cities = coordinate.shape[1]
    index = range(coordinate.shape[1])
    distance = np.zeros((cities, cities))
    for i in range(1, coordinate.shape[1]):
        distance[index[:-i], index[i:]] = np.sqrt(np.sum( \
                np.power(coordinate[:, :-i]-coordinate[:, i:], 2), axis=0))
        distance[index[i:], index[:-i]] = distance[index[:-i], index[i:]]

    sa = SA(1, np.arange(cities), 30, 50, \
            lambda x, args=None: x*0.95, get_newpath, \
            lambda x, args=None: np.sum(args[x[0:-1], x[1:]]) + args[x[-1], x[0]], \
            cost_args = distance, \
            inner_termination_func=termination, \
            outer_termination_func=termination)

    sa.simulated_annealing()

    print(sa.solve, sa.cost)
    myplot(sa.solve, coordinate)
