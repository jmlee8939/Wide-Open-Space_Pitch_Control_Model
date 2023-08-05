import numpy as np
import pandas as pd
import math
from scipy.stats import multivariate_normal


def influence_radius(ball, position):
    distance = np.linalg.norm(ball - position)
    output = np.minimum(3/200*(distance)**2 + 4, 10)
    return output


def influence_function2(position, locations, velocity, ball):
    mu = position + 0.5*velocity
    srat = (velocity[0]**2 + velocity[1]**2)/13**2
    theta = np.arctan(velocity[1]/(velocity[0]+1e-7))
    R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    R_inv = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
    Ri = influence_radius(ball, position)
    S = np.array([[(1 + srat)*Ri/2, 0],[0, (1-srat)*Ri/2]])
    Cov = np.matmul(np.matmul(np.matmul(R, S), S), R_inv)
    new_gaussian = multivariate_normal(mu, Cov)
    out = new_gaussian.pdf(locations)
    out /= new_gaussian.pdf(position)
    return out

def influence_function(position, location):
    mv_gaussian = multivariate_normal(position, [[12, 0], [0, 12]])
    out = mv_gaussian.pdf(location)
    out /= mv_gaussian.pdf(position)
    return out