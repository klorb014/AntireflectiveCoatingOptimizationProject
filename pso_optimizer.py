import operator
import random

import numpy
import math

from tmm import TransferMatrixMethod

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style


class ParticleSwarmOptimizer:

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
                   smin=None, smax=None, best=None)

    def __init__(self, layers: int, generations: int, nmin=1.000293, nmax=3, lmin=0, lmax=1, smin=1.5, smax=2.5, phi1=2.0, phi2=2.0):
        self.layers = layers
        self.generations = generations
        self.toolbox = self.generate_toolbox(
            layers, nmin, nmax, smin, smax, lmin, lmax, phi1, phi2)

    @staticmethod
    def generate(size, nmin, nmax, smin, smax, lmin, lmax):
        part = creator.Particle([random.uniform(nmin, nmax) for _ in range(
            size)])
        part.speed = [random.uniform(smin, smax) for _ in range(size)]
        part.smin = smin
        part.smax = smax
        part.nmin = nmin
        part.nmax = nmax
        return part

    @staticmethod
    def updateParticle(part, best, phi1, phi2):
        u1 = (random.uniform(0, phi1) for _ in range(len(part)))
        u2 = (random.uniform(0, phi2) for _ in range(len(part)))
        v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
        v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
        part.speed = list(map(operator.add, part.speed,
                              map(operator.add, v_u1, v_u2)))
        for i, speed in enumerate(part.speed):
            if abs(speed) < part.smin:
                part.speed[i] = math.copysign(part.smin, speed)
            elif abs(speed) > part.smax:
                part.speed[i] = math.copysign(part.smax, speed)
        part[:] = list(map(operator.add, part, part.speed))

    @staticmethod
    def feasible(individual):
        """Feasibility function for the individual. Returns True if feasible False
        otherwise."""

        for refractive_ind in individual:
            if not(individual.nmin < refractive_ind < individual.nmax):
                return False
        return True

    @staticmethod
    def distance(individual):
        """A distance function to the feasibility region."""
        return (individual[0] - 5.0)**2

    def generate_toolbox(self, size, nmin, nmax, smin, smax, lmin, lmax, phi1, phi2):
        toolbox = base.Toolbox()
        toolbox.register("particle", ParticleSwarmOptimizer.generate, size=size,
                         nmin=nmin, nmax=nmax, smin=smin, smax=smax, lmin=lmin, lmax=lmax)
        toolbox.register("population", tools.initRepeat,
                         list, toolbox.particle)
        toolbox.register(
            "update", ParticleSwarmOptimizer.updateParticle, phi1=phi1, phi2=phi2)

        toolbox.register("evaluate", TransferMatrixMethod.evaluate_solution)
        toolbox.decorate("evaluate", tools.DeltaPenalty(
            ParticleSwarmOptimizer.feasible, -1.0))
        return toolbox

    def optimize(self):
        pop = self.toolbox.population(n=5)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)

        logbook = tools.Logbook()
        logbook.header = ["gen", "evals"] + stats.fields

        GEN = self.generations
        best = None

        for g in range(GEN):
            for part in pop:
                part.fitness.values = self.toolbox.evaluate(part)
                if not part.best or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if not best or best.fitness < part.fitness:
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values
            for part in pop:
                self.toolbox.update(part, best)

            # Gather all the fitnesses in one list and print the stats
            logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
            # print(logbook.stream)

        return pop, logbook, best
