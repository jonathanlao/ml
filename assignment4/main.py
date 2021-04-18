import hiive.mdptoolbox.example as example
import hiive.mdptoolbox.mdp as mdp
import matplotlib.pyplot as plt
import numpy as np

def plot_data(x, np_arr, title="Figure 1", x_label="Predicted Y", y_label="Test Y", color="blue", label=None):
    plt.plot(x, np_arr, label=label, color = color)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()


def print_stats(i, name):
    print(name)
    print('V', i.V)
    print('Iterations', i.iter)
    print('Time', i.time)
    print('Policy', i.policy)


def print_qstats(ql):
    print('Q')
    print(ql.V)
    # print('Q Matrix', ql.Q)
    print(ql.time)
    # print(ql.run_stats)
    print(ql.policy)
    # print(vi.policy)


def get_stats(iter, val, debug=False):
    value = []
    for i in iter.run_stats:
        value.append(i[val])
        if debug:
            print(i)
    return value


def forest():
    experiments = [
        {'num_states': 5, 'name': 'IterationsSmall', 'discount': 0.9, 'eps': 0.01},
        {'num_states': 100, 'name': 'IterationsLarge', 'discount': 0.9, 'eps': 0.01},
        {'num_states': 5, 'name': 'IterationsSmall', 'discount': 0.1, 'eps': 0.01},
        {'num_states': 100, 'name': 'IterationsLarge', 'discount': 0.1, 'eps': 0.01},
        {'num_states': 5, 'name': 'IterationsSmall', 'discount': 0.9, 'eps': 0.2},
        {'num_states': 100, 'name': 'IterationsLarge', 'discount': 0.9, 'eps': 0.2},
    ]

    counter = 0
    for e in experiments:
        counter += 1
        num_states = e['num_states']
        P, R = example.forest(S=num_states, r1=5, r2=50, p=0.1, is_sparse=False)

        vi = mdp.ValueIteration(P, R, e['discount'], epsilon=e['eps'])
        # vi.setVerbose()
        vi.run()

        pi = mdp.PolicyIteration(P, R, e['discount'])
        # pi.setVerbose()
        pi.run()
        
        print_stats(vi, 'VI')
        print_stats(pi, 'PI')

        if counter < 3:
            vi_value = get_stats(vi, 'Error', True)
            
            x_axis = [i for i in range(len(vi_value))]
            plot_data(x_axis, vi_value, 
                title="VI/PI Iteration, Num States=5", 
                x_label="Iterations", y_label="Error", color="blue", label='VI')


            pi_value = get_stats(pi, 'Error', False)
            
            plot_data([i for i in range(len(pi_value))], pi_value, 
                title="VI/PI Iteration, Num States=" + str(num_states), 
                x_label="Iterations", y_label="Error", color="orange", label='PI')

            plt.savefig('output/Forest'+e['name']+'.png')
            plt.close()


    experiments = [
        {'num_states': 5, 'eps': 0.99, 'alpha': 0.1, 'alpha_decay': 0.99, 'min_alpha': 0.001, 'color': 'blue', 'label': 'Default', 'adj': 'Epsilon', 'iter': 10000},
        {'num_states': 5, 'eps': 0.9999, 'alpha': 0.1,'alpha_decay': 0.99, 'min_alpha': 0.001, 'color': 'orange', 'label': 'Explorer', 'adj': 'Epsilon', 'iter': 10000},
        {'num_states': 5, 'eps': 0.95, 'alpha': 0.1,'alpha_decay': 0.99, 'min_alpha': 0.001, 'color': 'green', 'label': 'Exploiter', 'adj': 'Epsilon', 'iter': 10000},

        {'num_states': 5, 'eps': 0.9999, 'alpha': 1.0,'alpha_decay': 0.99, 'min_alpha': 0.001, 'color': 'blue', 'label': 'Default', 'adj': 'Alpha', 'iter': 10000},
        {'num_states': 5, 'eps': 0.9999, 'alpha': 1.0,'alpha_decay': 0.9999,'min_alpha': 0.1, 'color': 'orange', 'label': 'High Learning Rate', 'adj': 'Alpha', 'iter': 10000},
        {'num_states': 5, 'eps': 0.9999, 'alpha': 0.5,'alpha_decay': 0.9,'min_alpha': 0.001, 'color': 'green', 'label': 'Low Learning Rate', 'adj': 'Alpha', 'iter': 10000},

        {'num_states': 100, 'eps': 0.99, 'alpha': 0.1, 'alpha_decay': 0.99, 'min_alpha': 0.001, 'color': 'blue', 'label': 'Default', 'adj': 'Epsilon', 'iter': 100000},
        {'num_states': 100, 'eps': 0.9999, 'alpha': 0.1,'alpha_decay': 0.99, 'min_alpha': 0.001, 'color': 'orange', 'label': 'Explorer', 'adj': 'Epsilon', 'iter': 100000},
        {'num_states': 100, 'eps': 0.95, 'alpha': 0.1,'alpha_decay': 0.99, 'min_alpha': 0.001, 'color': 'green', 'label': 'Exploiter', 'adj': 'Epsilon', 'iter': 100000},

        {'num_states': 100, 'eps': 0.9999, 'alpha': 1.0,'alpha_decay': 0.99, 'min_alpha': 0.001, 'color': 'blue', 'label': 'Default', 'adj': 'Alpha', 'iter': 100000},
        {'num_states': 100, 'eps': 0.9999, 'alpha': 1.0,'alpha_decay': 0.9999,'min_alpha': 0.1, 'color': 'orange', 'label': 'High Learning Rate', 'adj': 'Alpha', 'iter': 100000},
        {'num_states': 100, 'eps': 0.9999, 'alpha': 0.5,'alpha_decay': 0.9,'min_alpha': 0.001, 'color': 'green', 'label': 'Low Learning Rate', 'adj': 'Alpha', 'iter': 100000},
    ]

    counter = 0
    for e in experiments:
        counter += 1
        num_states = e['num_states']
        P, R = example.forest(S=num_states, r1=5, r2=50, p=0.1, is_sparse=False)

        ql = mdp.QLearning(P, R, 0.9, 
            # alpha=0.1, alpha_decay=0.99, alpha_min=0.1,
            alpha=e['alpha'], alpha_decay=0.99, alpha_min=e['min_alpha'],
            epsilon=1.0, epsilon_min=0.1, epsilon_decay=e['eps'], 
            # epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.99, 
            n_iter=e['iter'], skip_check=True)
        
        ql.setVerbose()
        ql.run()

        value = get_stats(ql, 'Max V', False)

        x_axis = [i for i in range(len(value))] if counter < 7 else [i*10 for i in range(len(value))] 
        plot_data(x_axis, value, 
            title="Q-Learning, " +str(e['num_states']) + " States, "+ e['adj'] +" Adjusted", 
            x_label="Iterations", y_label="Max V", color=e['color'], label=e['label'])

        print_qstats(ql)

        if counter == 3:
            plt.savefig('output/ForestQSmallEpsilon.png')
            plt.close()
        
        if counter == 6:
            plt.savefig('output/ForestQSmallAlpha.png')
            plt.close()

        if counter == 9:
            plt.savefig('output/ForestQLargeEpsilon.png')
            plt.close()
        
        if counter == 12:
            plt.savefig('output/ForestQLargeAlpha.png')
            plt.close()

if __name__ == "__main__":
    np.random.seed(43)
    forest()
    # small_forest()
    # large_forest()