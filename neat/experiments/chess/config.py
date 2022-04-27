import logging
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch import autograd

from neat.phenotype.feed_forward import FeedForwardNet


logger = logging.getLogger(__name__)

class ChessConfig:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # DEVICE = "cpu"
    print(f"device: {DEVICE}")
    np_rng = np.random.default_rng(13)
    VERBOSE = True

    NUM_INPUTS = 8
    NUM_OUTPUTS = 1
    USE_BIAS = True

    ACTIVATION = 'relu'
    SCALE_ACTIVATION = 1

    FITNESS_THRESHOLD = 1950

    POPULATION_SIZE = 150
    NUMBER_OF_GENERATIONS = 150
    SPECIATION_THRESHOLD = 3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ACTIVATION_MUTATION_RATE = 0.2
    ADD_NODE_MUTATION_RATE = 0.1
    ADD_CONNECTION_MUTATION_RATE = 0.5

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.30

    print("loading chess data")
    with open("../data/KPK/indices", "rb") as f:
        data = pickle.load(f)

    inputs_list = []
    outputs_list = []
    np_rng.shuffle(data)
    # data_proportion = int(np.sqrt(len(data)))
    data_proportion = 2000
    data = data[:data_proportion]
    for xy in data:
        # xy[0] should already be a numpy array
        inputs_list.append(xy[0] / 8)
        outputs_list.append(xy[1])
    inputs = torch.tensor(np.array(inputs_list)).to(DEVICE)
    targets = torch.tensor(np.array(outputs_list)).reshape(-1,1).to(DEVICE)


    def fitness_fn(self, genome):
        fitness = self.data_proportion  # Max fitness

        phenotype = FeedForwardNet(genome, self)
        phenotype.to(self.DEVICE)
        criterion = nn.MSELoss()
        num_inputs = len(self.inputs)

        pred = phenotype(self.inputs)
        loss = criterion(pred, self.targets)

        fitness -= self.data_proportion * loss.item()

        return fitness


    def get_preds_and_labels(self, genome):
        phenotype = FeedForwardNet(genome, self)
        phenotype.to(self.DEVICE)

        predictions = []
        labels = []
        for input, target in zip(self.inputs, self.targets):
            input, target = input.to(self.DEVICE), target.to(self.DEVICE)

            predictions.append(float(phenotype(input)))
            labels.append(float(target))

        return predictions, labels
