import mlrose_hiive as mlr
import numpy as np
import time
import matplotlib.pyplot as plt


RANDOM_STATE = 42

def plot_data(x, np_arr, title="Figure 1", x_label="Predicted Y", y_label="Test Y", color="blue", label=None):
  plt.plot(x ,np_arr, label=label, color = color)
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend()


def randomized_hill_climb(problem, init_state, max_attempts, max_iters):
    start_time = time.time()
    best_state, best_fitness, fitness_curve = mlr.random_hill_climb(problem, max_attempts, max_iters, restarts=10, init_state=init_state, curve=True, random_state=RANDOM_STATE)
    end_time = time.time()
    total_time = end_time - start_time
    

    print('Random Hill Climb')
    print("Elapsed Time", total_time)                                                 
    print(best_state)
    print(best_fitness)
#     print(fitness_curve)
    return best_state, best_fitness, fitness_curve, total_time


def simulated_annealing(problem, init_state, max_attempts, max_iters):
    schedule = mlr.ExpDecay() # Tune this?

    start_time = time.time()
    best_state, best_fitness, fitness_curve  = mlr.simulated_annealing(problem, schedule = schedule,
                                                          max_attempts = max_attempts, max_iters = max_iters, init_state = init_state, curve=True, random_state = RANDOM_STATE)
    end_time = time.time()
    total_time = end_time - start_time
    

    print('Simulated Annealing')
    print("Elapsed Time", total_time)                                                 
    print(best_state)
    print(best_fitness)
#     print(fitness_curve)
    return best_state, best_fitness, fitness_curve, total_time


def genetic_algorithm(problem, init_state, max_attempts, max_iters):
    start_time = time.time()
    # Does the kind of mutation/algo matter?
    best_state, best_fitness, fitness_curve  = mlr.genetic_alg(problem, pop_size=200, mutation_prob=0.1, 
                                                          max_attempts = max_attempts, max_iters = max_iters, curve=True, random_state = RANDOM_STATE)
    end_time = time.time()
    total_time = end_time - start_time
    

    print('Genetic Algorithm')
    print("Elapsed Time", total_time)                                                 
    print(best_state)
    print(best_fitness)
#     print(fitness_curve)
    return best_state, best_fitness, fitness_curve, total_time


def mimic(problem, init_state, max_attempts, max_iters):
    start_time = time.time()
    best_state, best_fitness, fitness_curve = mlr.mimic(problem, pop_size=200, keep_pct=0.2, 
                                                          max_attempts = max_attempts, max_iters = max_iters, curve=True, random_state = RANDOM_STATE)
    end_time = time.time()
    total_time = end_time - start_time
    
    print('MIMIC')
    print("Elapsed Time", total_time)                                                 
    print(best_state)
    print(best_fitness)
#     print(fitness_curve)
    return best_state, best_fitness, fitness_curve, total_time


def flip_flop():
    sa = []
    rhc = []
    ga = []
    mim = []

    input_sizes = [100]

    for i in input_sizes:
        state = np.array([np.random.randint(0, 2) for i in range(i)])
        # state = np.zeros(100)
        # fitness = mlr.FourPeaks(0.1)
        fitness = mlr.FlipFlop()
        problem = mlr.DiscreteOpt(length = i, fitness_fn = fitness, maximize = True, max_val = 2)

        best_state, best_fitness, fitness_curve, time = randomized_hill_climb(problem, state, 10, 100)
        rhc.append((best_fitness, fitness_curve, time))

        best_state, best_fitness, fitness_curve, time = simulated_annealing(problem, state, 10, 100)
        sa.append((best_fitness, fitness_curve, time))

        best_state, best_fitness, fitness_curve, time = genetic_algorithm(problem, state, 10, 100)
        ga.append((best_fitness, fitness_curve, time))

        best_state, best_fitness, fitness_curve, time = mimic(problem, state, 10, 100)
        mim.append((best_fitness, fitness_curve, time))


def four_peaks():
    sa = []
    rhc = []
    ga = []
    mim = []

    input_sizes = [100]

    for i in input_sizes:
        state = np.array([np.random.randint(0, 2) for i in range(i)])
        # state = np.zeros(100)
        fitness = mlr.FourPeaks(0.1)
        problem = mlr.DiscreteOpt(length = i, fitness_fn = fitness, maximize = True, max_val = 2)

        best_state, best_fitness, fitness_curve, time = randomized_hill_climb(problem, state, 10, 1000)
        rhc.append((best_fitness, fitness_curve, time))

        best_state, best_fitness, fitness_curve, time = simulated_annealing(problem, state, 10, 1000)
        sa.append((best_fitness, fitness_curve, time))

        best_state, best_fitness, fitness_curve, time = genetic_algorithm(problem, state, 10, 1000)
        ga.append((best_fitness, fitness_curve, time))

        best_state, best_fitness, fitness_curve, time = mimic(problem, state, 10, 1000)
        mim.append((best_fitness, fitness_curve, time))


def one_max():
    sa = []
    rhc = []
    ga = []
    mim = []

    input_sizes = [1000]

    for i in input_sizes:
        state = np.array([np.random.randint(0, 2) for i in range(i)])
        # state = np.zeros(i)
        fitness = mlr.OneMax()
        problem = mlr.DiscreteOpt(length = i, fitness_fn = fitness, maximize = True, max_val = 2)

        best_state, best_fitness, fitness_curve, time = randomized_hill_climb(problem, state, 100, 10000)
        rhc.append((best_fitness, fitness_curve, time))

        best_state, best_fitness, fitness_curve, time = simulated_annealing(problem, state, 100, 10000)
        sa.append((best_fitness, fitness_curve, time))

        best_state, best_fitness, fitness_curve, time = genetic_algorithm(problem, state, 100, 1000)
        ga.append((best_fitness, fitness_curve, time))

        # best_state, best_fitness, fitness_curve, time = mimic(problem, state, 10, 1000)
        # mim.append((best_fitness, fitness_curve, time))

        # times = []
        # best_scores = []
        # times.append(elapsed_time*1000)
        # best_scores.append(best_fitness)
        # sa_times.append( *1000)
        # sa_evals.append(len(fitness_curve))

        
        # plot_data([i+1 for i in range(len(fitness_curve))], fitness_curve, title="Evaluations Required to Maximize OneMax (Input Size = "+str(len(state))+")", x_label="Evaluations", y_label="Fitness Score", color="blue", label='Simulated annealing')

        # title = 'SA - OneMax'

        # plt.savefig('output/'+title+'.png')
        # plt.close()

    # best_state, best_fitness = mlr.simulated_annealing(problem, schedule = schedule,
    #                                                   max_attempts = 10, max_iters = 1000,
    #                                                   init_state = initial_states[0], random_state = RANDOM_STATE)


if __name__ == "__main__":     
    # one_max()
    # four_peaks() # genetic algorithm
    flip_flop()


