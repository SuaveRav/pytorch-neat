import logging
import datetime

import neat.population as pop
import neat.experiments.chess.config as c
from neat.visualize import draw_net
from tqdm import tqdm

logger = logging.getLogger(__name__)

num_of_solutions = 0

avg_num_hidden_nodes = 0
min_hidden_nodes = 100000
max_hidden_nodes = 0
found_minimal_solution = 0

avg_num_generations = 0
min_num_generations = 100000

for i in tqdm(range(1)):
    neat = pop.Population(c.ChessConfig)
    solution, generation, stats = neat.run()

    if solution is not None:
        avg_num_generations = ((avg_num_generations * num_of_solutions) + generation) / (num_of_solutions + 1)
        min_num_generations = min(generation, min_num_generations)

        num_hidden_nodes = len([n for n in solution.node_genes if n.type == 'hidden'])
        avg_num_hidden_nodes = ((avg_num_hidden_nodes * num_of_solutions) + num_hidden_nodes) / (num_of_solutions + 1)
        min_hidden_nodes = min(num_hidden_nodes, min_hidden_nodes)
        max_hidden_nodes = max(num_hidden_nodes, max_hidden_nodes)
        if num_hidden_nodes == 1:
            found_minimal_solution += 1

        now = datetime.datetime.utcnow()
        draw_net(solution, view=True, filename='./images/solution-' + now.strftime('%Y%m%d-%H%M%S'), show_disabled=True)

        logger.info("Solution: {}".format(solution))
        logger.info("Generation: {}".format(generation))

        with open(f'./KRK-solutions/{now.strftime("%Y%m%d-%H%M%S")}.txt', 'w') as f:
            lines = (
                [
                    'CONFIG\n',
                    f'Activation: {c.ChessConfig.ACTIVATION}\n',
                    f'Scale Activation: {c.ChessConfig.SCALE_ACTIVATION}\n',
                    f'Fitness Threshold: {c.ChessConfig.FITNESS_THRESHOLD}\n',
                    f'Population Size: {c.ChessConfig.POPULATION_SIZE}\n',
                    f'Number of Generations: {c.ChessConfig.NUMBER_OF_GENERATIONS}\n',
                    f'Speciation Threshold: {c.ChessConfig.SPECIATION_THRESHOLD}\n',
                    f'Connection Mutation Rate: {c.ChessConfig.CONNECTION_MUTATION_RATE}\n',
                    f'Connection Perturbation Rate: {c.ChessConfig.CONNECTION_PERTURBATION_RATE}\n',
                    f'Add Node Mutation Rate: {c.ChessConfig.ADD_NODE_MUTATION_RATE}\n',
                    f'Add Connection Mutation Rate: {c.ChessConfig.ADD_CONNECTION_MUTATION_RATE}\n',
                    f'Crossover Reenable Connection Gene Rate: {c.ChessConfig.CROSSOVER_REENABLE_CONNECTION_GENE_RATE}\n',
                    f'Percentage to Save: {c.ChessConfig.PERCENTAGE_TO_SAVE}\n',
                    '\n',
                    f'SOLUTION\n {solution} \n\n',
                    f'GENERATION\n {generation} \n\n',
                    'STATS\n',
                ]
                + [
                    f'Gen: {g} | Fitness: {f} | Length {l}\n' for (g, f, l) in stats
                ]
                + [
                    '\n\n',
                    'Total Number of Solutions: {}\n'.format(num_of_solutions),
                    'Average Number of Hidden Nodes in a Solution {}\n'.format(avg_num_hidden_nodes),
                    'Solution found on average in:{}\n generations'.format(avg_num_generations),
                    'Minimum number of hidden nodes: {}\n'.format(min_hidden_nodes),
                    'Maximum number of hidden nodes: {}\n'.format(max_hidden_nodes),
                    'Minimum number of generations: {}\n'.format(min_num_generations),
                    'Found minimal solution: {}\n times'.format(found_minimal_solution),
                ]
            )

            f.writelines(lines)

logger.info('Total Number of Solutions: {}'.format(num_of_solutions))
logger.info('Average Number of Hidden Nodes in a Solution {}'.format(avg_num_hidden_nodes))
logger.info('Solution found on average in:{} generations'.format(avg_num_generations))
logger.info('Minimum number of hidden nodes: {}'.format(min_hidden_nodes))
logger.info('Maximum number of hidden nodes: {}'.format(max_hidden_nodes))
logger.info('Minimum number of generations: {}'.format(min_num_generations))
logger.info('Found minimal solution: {} times'.format(found_minimal_solution))
