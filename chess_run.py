import logging
import os
import json
import pandas as pd

import neat.population as pop
import neat.experiments.chess.config as c
from neat.visualize import draw_net
from tqdm import tqdm
from neat.phenotype.feed_forward import FeedForwardNet

logger = logging.getLogger(__name__)

num_of_solutions = 0

avg_num_hidden_nodes = 0
min_hidden_nodes = 100000
max_hidden_nodes = 0
found_minimal_solution = 0

avg_num_generations = 0
min_num_generations = 100000

experiment_folders = os.listdir("./solutions/KRK")
experiment_folders = [int(i) for i in experiment_folders]
experiment_folders = sorted(experiment_folders)
previous_experiment = int(experiment_folders[-1])
new_experiment = previous_experiment + 1
directory = './solutions/KRK/{0}'.format(new_experiment)

fitnesses = []
def callback_func(fitness):
    fitnesses.append(fitness)

print("Running Experiment {}".format(new_experiment))
for i in tqdm(range(1)):
    neat = pop.Population(c.ChessConfig)
    logger.info(f'Max Generations: {neat.Config.NUMBER_OF_GENERATIONS}')
    solution, generation, fitnesses = neat.run()

    if solution is not None:
        os.mkdir(directory)
        avg_num_generations = ((avg_num_generations * num_of_solutions) + generation) / (num_of_solutions + 1)
        min_num_generations = min(generation, min_num_generations)

        num_hidden_nodes = len([n for n in solution.node_genes if n.type == 'hidden'])
        avg_num_hidden_nodes = ((avg_num_hidden_nodes * num_of_solutions) + num_hidden_nodes) / (num_of_solutions + 1)
        min_hidden_nodes = min(num_hidden_nodes, min_hidden_nodes)
        max_hidden_nodes = max(num_hidden_nodes, max_hidden_nodes)
        if num_hidden_nodes == 1:
            found_minimal_solution += 1

        logger.info("Solution: {}".format(solution))
        logger.info("Generation: {}".format(generation))

        # Draw Solution
        num_of_solutions += 1
        draw_net(solution, view=True, filename='./solutions/KRK/{0}/solution-'.format(new_experiment) + str(num_of_solutions), show_disabled=True)
        
        # Write Solution
        with open('./solutions/KRK/{0}/solution'.format(new_experiment), 'w') as convert_file:
            convert_file.write(str(solution))

        # Write Fitnesses
        with open('./solutions/KRK/{0}/fitnesses'.format(new_experiment), 'w') as convert_file:
            convert_file.write(json.dumps(fitnesses))

        # Write config
        with open(f'./solutions/KRK/{new_experiment}/config.txt', 'w') as f:
            lines = [
                'CONFIG\n',
                f'Activation: {c.ChessConfig.ACTIVATION}\n',
                f'Scale Activation: {c.ChessConfig.SCALE_ACTIVATION}\n',
                f'Fitness Threshold: {c.ChessConfig.FITNESS_THRESHOLD}\n',
                f'Population Size: {c.ChessConfig.POPULATION_SIZE}\n',
                f'Number of Generations: {c.ChessConfig.NUMBER_OF_GENERATIONS}\n',
                f'Speciation Threshold: {c.ChessConfig.SPECIATION_THRESHOLD}\n',
                f'Connection Mutation Rate: {c.ChessConfig.CONNECTION_MUTATION_RATE}\n',
                f'Connection Perturbation Rate: {c.ChessConfig.CONNECTION_PERTURBATION_RATE}\n',
                f'Activation Mutation Rate: {c.ChessConfig.ACTIVATION_MUTATION_RATE}\n',
                f'Add Node Mutation Rate: {c.ChessConfig.ADD_NODE_MUTATION_RATE}\n',
                f'Add Connection Mutation Rate: {c.ChessConfig.ADD_CONNECTION_MUTATION_RATE}\n',
                f'Crossover Reenable Connection Gene Rate: {c.ChessConfig.CROSSOVER_REENABLE_CONNECTION_GENE_RATE}\n',
                f'Percentage to Save: {c.ChessConfig.PERCENTAGE_TO_SAVE}\n',
                f'Divide by 8: {c.ChessConfig.NORMALIZE_INPUTS}\n'
            ]

            f.writelines(lines)

        # Test and write results
        best_network = FeedForwardNet(solution, c.ChessConfig)
        results = neat.Config.test(best_network)

        col_names = [
            'white rook height',
            'white rook width',
            'white king height',
            'white king width',
            'black rook height',
            'black rook width',
            'black king height',
            'black king width',
            'loss',
            'prediction',
        ]

        results_df = pd.DataFrame(results, columns=col_names)
        results_df.to_csv('./solutions/KRK/{0}/test_results.csv'.format(new_experiment), index=False)

logger.info('Total Number of Solutions: {}'.format(num_of_solutions))
logger.info('Average Number of Hidden Nodes in a Solution {}'.format(avg_num_hidden_nodes))
logger.info('Solution found on average in:{} generations'.format(avg_num_generations))
logger.info('Minimum number of hidden nodes: {}'.format(min_hidden_nodes))
logger.info('Maximum number of hidden nodes: {}'.format(max_hidden_nodes))
logger.info('Minimum number of generations: {}'.format(min_num_generations))
logger.info('Found minimal solution: {} times'.format(found_minimal_solution))
