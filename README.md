# Antireflective Coating Optimization Project

This repo outlines an analysis of multilayer antireflective coatings. The goal is find an optimal configuration of layers, layer thicknesses, and layer refractive indices in order to maximize the amount of light absorbed by a solar cell. Maximizing the amount absorbed by a solar cell is critical to increase power efficiency. The progagation of light through the multilayer system is computed using the Transfer Matrix method.

This repo aims to compare a few different optimization algorithms for this use case. Currently, the only implemented algorithm is a Particle Swarm algorithm.

On top of searching for the best refractive index for each layer, the program will then attempt to map the refractive index to a corresponding material. The material-refractive index dataset is not very comprehensive, so the some of the combinations are not very practical in practice. 

Read the README.ipynb for more information about the underlying theory.
