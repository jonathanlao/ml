import mlrose_hiive as mlr
import numpy as np
import time
import matplotlib.pyplot as plt
import os.path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time


ROOT = os.path.abspath(os.path.dirname(__file__))
FIGURE = 0
RANDOM_STATE = 42

def add_data(x, np_arr, title="Figure 1", x_label="Predicted Y", y_label="Test Y", color="blue", label=None):
  plt.plot(x ,np_arr, label=label, color = color)

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
    # print(best_state)
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
    #print(best_state)
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
    #print(best_state)
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
    #print(best_state)
    print(best_fitness)
#     print(fitness_curve)
    return best_state, best_fitness, fitness_curve, total_time


def flip_flop():
    print('Flip Flop')
    sa = []
    rhc = []
    ga = []
    mim = []

    input_sizes = [200]

    for i in input_sizes:
        state = np.array([np.random.randint(0, 2) for i in range(i)])
        # state = np.zeros(100)
        # fitness = mlr.FourPeaks(0.1)
        fitness = mlr.FlipFlop()
        problem = mlr.DiscreteOpt(length = i, fitness_fn = fitness, maximize = True, max_val = 2)

        best_state, best_fitness, fitness_curve, time = randomized_hill_climb(problem, state, 100, 1000)
        rhc.append((best_fitness, fitness_curve, time))

        best_state, best_fitness, fitness_curve, time = simulated_annealing(problem, state, 100, 1000)
        sa.append((best_fitness, fitness_curve, time))

        best_state, best_fitness, fitness_curve, time = genetic_algorithm(problem, state, 100, 1000)
        ga.append((best_fitness, fitness_curve, time))

        best_state, best_fitness, fitness_curve, time = mimic(problem, state, 100, 1000)
        mim.append((best_fitness, fitness_curve, time))

    plot_data([i+1 for i in range(len(rhc[0][1]))], rhc[0][1], 
    title="FourPeaks (Input Size = "+str(len(state))+")", 
    x_label="Iterations", y_label="Fitness Score", color="blue", label='RHC')
    
    plot_data([i+1 for i in range(len(sa[0][1]))], sa[0][1], 
        title="FourPeaks (Input Size = "+str(len(state))+")", 
        x_label="Iterations", y_label="Fitness Score", color="orange", label='SA')

    plot_data([i+1 for i in range(len(ga[0][1]))], ga[0][1], 
        title="FourPeaks (Input Size = "+str(len(state))+")", 
        x_label="Iterations", y_label="Fitness Score", color="green", label='GA')
    
    plot_data([i+1 for i in range(len(mim[0][1]))], mim[0][1], 
        title="OneMax (Input Size = "+str(len(state))+")", 
        x_label="Iterations", y_label="Fitness Score", color="red", label='MIMIC')


    title = 'Flip Flop'

    plt.savefig('output/'+title+'.png')
    plt.close()


def four_peaks():
    print('Four Peaks')
    sa = []
    rhc = []
    ga = []
    mim = []

    input_sizes = [100]

    for i in input_sizes:
        state = np.array([np.random.randint(0, 2) for i in range(i)])
        # state = np.zeros(100)
        fitness = mlr.FourPeaks(0.15)
        problem = mlr.DiscreteOpt(length = i, fitness_fn = fitness, maximize = True, max_val = 2)

        best_state, best_fitness, fitness_curve, time = randomized_hill_climb(problem, state, 100, 1000)
        rhc.append((best_fitness, fitness_curve, time))

        best_state, best_fitness, fitness_curve, time = simulated_annealing(problem, state, 100, 1000)
        sa.append((best_fitness, fitness_curve, time))

        best_state, best_fitness, fitness_curve, time = genetic_algorithm(problem, state, 100, 1000)
        ga.append((best_fitness, fitness_curve, time))

        best_state, best_fitness, fitness_curve, time = mimic(problem, state, 100, 1000)
        mim.append((best_fitness, fitness_curve, time))


    plot_data([i+1 for i in range(len(rhc[0][1]))], rhc[0][1], 
    title="FourPeaks (Input Size = "+str(len(state))+")", 
    x_label="Iterations", y_label="Fitness Score", color="blue", label='RHC')
    
    plot_data([i+1 for i in range(len(sa[0][1]))], sa[0][1], 
        title="FourPeaks (Input Size = "+str(len(state))+")", 
        x_label="Iterations", y_label="Fitness Score", color="orange", label='SA')

    plot_data([i+1 for i in range(len(ga[0][1]))], ga[0][1], 
        title="FourPeaks (Input Size = "+str(len(state))+")", 
        x_label="Iterations", y_label="Fitness Score", color="green", label='GA')
    
    plot_data([i+1 for i in range(len(mim[0][1]))], mim[0][1], 
        title="OneMax (Input Size = "+str(len(state))+")", 
        x_label="Iterations", y_label="Fitness Score", color="red", label='MIMIC')


    title = 'Four Peaks'

    plt.savefig('output/'+title+'.png')
    plt.close()


def one_max():
    print('One Max') 
    sa = []
    rhc = []
    ga = []
    mim = []

    input_sizes = [100]

    for i in input_sizes:
        state = np.array([np.random.randint(0, 2) for i in range(i)])
        # state = np.zeros(i)
        fitness = mlr.OneMax()
        problem = mlr.DiscreteOpt(length = i, fitness_fn = fitness, maximize = True, max_val = 2)

        best_state, best_fitness, fitness_curve, time = randomized_hill_climb(problem, state, 100, 600)
        rhc.append((best_fitness, fitness_curve, time))

        best_state, best_fitness, fitness_curve, time = simulated_annealing(problem, state, 100, 600)
        sa.append((best_fitness, fitness_curve, time))

        best_state, best_fitness, fitness_curve, time = genetic_algorithm(problem, state, 100, 600)
        ga.append((best_fitness, fitness_curve, time))

        best_state, best_fitness, fitness_curve, time = mimic(problem, state, 100, 600)
        mim.append((best_fitness, fitness_curve, time))
    
    plot_data([i+1 for i in range(len(rhc[0][1]))], rhc[0][1], 
        title="OneMax (Input Size = "+str(len(state))+")", 
        x_label="Iterations", y_label="Fitness Score", color="blue", label='RHC')
    
    plot_data([i+1 for i in range(len(sa[0][1]))], sa[0][1], 
        title="OneMax (Input Size = "+str(len(state))+")", 
        x_label="Iterations", y_label="Fitness Score", color="orange", label='SA')

    plot_data([i+1 for i in range(len(ga[0][1]))], ga[0][1], 
        title="OneMax (Input Size = "+str(len(state))+")", 
        x_label="Iterations", y_label="Fitness Score", color="green", label='GA')
    
    plot_data([i+1 for i in range(len(mim[0][1]))], mim[0][1], 
        title="OneMax (Input Size = "+str(len(state))+")", 
        x_label="Iterations", y_label="Fitness Score", color="red", label='MIMIC')


    title = 'One Max'

    plt.savefig('output/'+title+'.png')
    plt.close()


# taken from sklearn documentaiton
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py

from sklearn.model_selection import validation_curve
def plot_model_complexity_curve(model, title, features, labels, x_label, param_name, param_range, optional_param_range=None):

    train_scores, test_scores = validation_curve(
        model, features, labels, param_name=param_name, param_range=param_range,
        scoring="accuracy")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    print(test_scores_mean)
    print(test_scores_std)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Accuracy")
    plt.ylim(0.4, 1.01)
    lw = 2

    if optional_param_range:
        param_range = optional_param_range

    plt.plot(param_range, train_scores_mean, label="Training score",
                color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2,
                    color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.2,
                    color="navy", lw=lw)
    plt.legend(loc="best")

    plt.savefig('output/'+title)
    plt.close()


def load_data(dataset, datatype):
    x_path = os.path.join(ROOT, "../data/"+ dataset + "_"+ datatype +"_features.csv")
    features = np.genfromtxt(x_path, delimiter=',')

    y_path = x_path = os.path.join(ROOT, "../data/"+ dataset + "_"+datatype+"_labels.csv")
    labels = np.genfromtxt(y_path, delimiter=',')

    return features, labels


def neural_network():
    features, labels = load_data('dataset1', 'train')

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2,
                                                        random_state=RANDOM_STATE)




    algorithm = 'random_hill_climb'
    rhc = mlr.NeuralNetwork(hidden_nodes = [10,10], activation = 'relu', \
                                 algorithm = algorithm, max_iters=5000, \
                                 bias = True, is_classifier = True, learning_rate = 0.1, \
                                 early_stopping = True, clip_max = 1, max_attempts = 100, \
                                 random_state = RANDOM_STATE, curve=True)
    
    

    rhc.fit(x_train, y_train)
    y_train_pred = rhc.predict(x_train)
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    print(y_train_accuracy)
    plot_data([i+1 for i in range(len(rhc.fitness_curve))], rhc.fitness_curve, 
        title="Neural Network Fitness", 
        x_label="Iterations", y_label="Log Loss", color="blue", label='RHC')





    algorithm = 'simulated_annealing'
    sa = mlr.NeuralNetwork(hidden_nodes = [10,10], activation = 'relu', \
                                 algorithm = algorithm, max_iters=5000, \
                                 bias = True, is_classifier = True, learning_rate = 0.1, \
                                 early_stopping = True, clip_max = 1, max_attempts = 100, \
                                 random_state = RANDOM_STATE, curve=True)
    
    

    sa.fit(x_train, y_train)
    y_train_pred = sa.predict(x_train)
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    print(y_train_accuracy)

    plot_data([i+1 for i in range(len(sa.fitness_curve))], sa.fitness_curve, 
        title="Neural Network Fitness", 
        x_label="Iterations", y_label="Log Loss", color="orange", label='SA')








    algorithm = 'genetic_alg'
    ga = mlr.NeuralNetwork(hidden_nodes = [10,10], activation = 'relu', \
                                 algorithm = algorithm, max_iters=1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.1, \
                                 early_stopping = False, clip_max = 1, max_attempts = 50, \
                                 random_state = RANDOM_STATE, curve=True)
    
    

    ga.fit(x_train, y_train)
    y_train_pred = ga.predict(x_train)
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    print(y_train_accuracy)

    plot_data([i+1 for i in range(len(ga.fitness_curve))], ga.fitness_curve, 
        title="Neural Network Fitness", 
        x_label="Iterations", y_label="Log Loss", color="green", label='GA')


    plt.savefig('output/'+"Neural Network Fitness"+'.png')
    plt.close()

    
    # print(nn.validation_scores_)
    # plot_model_complexity_curve(nn, 'RHC Iterations', features, labels, 'Iterations', 'max_iters', 
    #     (500,1000,1500,2000,2500,3000,3500,4000,4500,5000))


if __name__ == "__main__":    
    one_max()
    four_peaks() 
    flip_flop()
    # neural_network()

    







