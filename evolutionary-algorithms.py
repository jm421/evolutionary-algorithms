import random
import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NUMBER_OF_ITEMS = 500


"""
Fitness function.

Returns difference between heaviest and lightest bins in a given solution.
Item weightings are determined by BPP type (bpp=10 -> 2i or bpp=100 -> 2i^2 , where i is the item number). 

"""
def fitness(solution, bpp):

    # Establish item weightings
    if bpp == 10:
        def weighting(i): return 2*i
    elif bpp == 100:
        def weighting(i): return 2*(i**2)
    else:
        raise ValueError

    # Populate bins according to solution and item weightings
    bins = [0] * bpp
    for item in range(NUMBER_OF_ITEMS):
        bin = solution[item]
        weight = weighting(item + 1)
        bins[bin - 1] += weight

    # Return difference between heaviest and lightest bin
    return max(bins) - min(bins)


"""
Generate initial population of solutions, of size p and for problem BPP1 (bpp=10) or BPP2 (bpp=100).

Returns a list of initial solutions (population) and a list of the corresponding fitness of
each solution (population_fitness). These two lists are used to represent the population. 

"""
def generate_initial_population(p, bpp):

    population = []
    population_fitness = []

    # Generate p solutions
    for _ in range(p):
        solution = []
        # Generate the random solution
        for _ in range(NUMBER_OF_ITEMS):
            solution.append(random.randint(1, bpp))
        # Store the solution and its fitness
        population.append(solution)
        population_fitness.append(fitness(solution, bpp))

    # Return initial population
    return population, population_fitness


"""
Perform binary tournament on a given population.

Returns winning solution.

"""
def binary_tournament_selection(population, population_fitness):

    # Choose two solutions from the population at random
    competitors = random.choices(list(population), k=2)
    # Evaluate each fitness
    competitor_fitness = [population_fitness[population.index(competitor)] for competitor in competitors]
    # Determine which has best fitness
    best_fitness = min(competitor_fitness)
    # Return fittest solution
    return competitors[competitor_fitness.index(best_fitness)]


"""
Perform single-point crossover on a given list of parent solutions.

Returns child solutions. 

"""
def single_point_crossover(parents):

    # Choose cross-over point at random
    crossover_point = random.randint(0, NUMBER_OF_ITEMS - 1)
    # Perform cross-over to produce children
    first_child = parents[0][0:crossover_point + 1] + parents[1][crossover_point + 1:]
    second_child = parents[1][0:crossover_point + 1] + parents[0][crossover_point + 1:]
    # Return children
    return first_child, second_child


"""
Apply Mk mutation operator to a given solution, for a given BPP problem.

Returns mutant solution.

"""
def mutate(solution, bpp, k):

    # Perform random mutation k times
    for _ in range(k):
        gene = random.randint(0, NUMBER_OF_ITEMS - 1)
        solution[gene] = random.randint(1, bpp)

    # Return mutant
    return solution


"""
Perform weakest replacement, given a new solution and the existing population.

Returns new population state.

"""
def weakest_replacement(new_solution, population, population_fitness, bpp):

    # Find worst fitness in current population
    worst_fitness = max(population_fitness)
    # Evaluate fitness of new solution
    new_fitness = fitness(new_solution, bpp)

    index = population_fitness.index(worst_fitness)
    # Replace weakest solution with new solution if its fitter
    if new_fitness < worst_fitness:
        population[index] = new_solution
        population_fitness[index] = new_fitness
    # Breaking ties randomly (50/50 chance of replacement)
    elif new_fitness == worst_fitness:
        if random.random() > 0.5:
            population[index] = new_solution
            population_fitness[index] = new_fitness
    # Otherwise leave the population state unchanged...

    # Return the population state
    return population, population_fitness


"""
Perform the EA until the termination criterion has been reached (10,000 fitness evaluations).

Parameters:
crossover (Boolean)       - Perform single-point crossover.
k         (int)           - Mutation operator (Mk), or k=0 for no mutations.
p         (int)           - Population size.
bpp       (int)           - BPP problem (bpp=10 for BPP1, bpp=100 for BPP2).
seed      (int, optional) - Seed for random number generator.

Returns list of corresponding fitness for each solution in the final population, and other relevant data for additional
experimentation.

"""
def evolutionary_algorithm(crossover, k, p, bpp, seed=False):

    if seed:
        random.seed(seed)

    # Initialise the population
    population, population_fitness = generate_initial_population(p, bpp)
    # Count the first p fitness evaluations (each random solution had its fitness evaluated)
    fitness_evaluations = p

    # Collect initial best and worst solution fitness
    y_min = [min(population_fitness)]
    initial_worst_fitness = max(population_fitness)

    while fitness_evaluations < 10000:
        # First binary tournament selection
        parent_a = binary_tournament_selection(population, population_fitness)
        seed += 1
        # Second binary tournament selection
        parent_b = binary_tournament_selection(population, population_fitness)

        # Single-point crossover
        if crossover:
            child_c, child_d = single_point_crossover([parent_a, parent_b])
        else:
            child_c, child_d = parent_a, parent_b

        # Mutations
        if k != 0:
            solution_e, solution_f = mutate(child_c, bpp, k), mutate(child_d, bpp, k)
        else:
            solution_e, solution_f = child_c, child_d

        # First weakest replacement
        population, population_fitness = weakest_replacement(solution_e, population, population_fitness, bpp)
        # Update termination condition and check if it has been met
        fitness_evaluations += 1
        if fitness_evaluations >= 10000:
            break
        # Collect data for line graph if necessary
        elif fitness_evaluations % 1000 == 0:
            y_min.append(min(population_fitness))

        # Second weakest replacement
        population, population_fitness = weakest_replacement(solution_f, population, population_fitness, bpp)
        # Update termination condition and check if it has been met
        fitness_evaluations += 1
        # Collect data for line graph if necessary
        if fitness_evaluations % 1000 == 0:
            y_min.append(min(population_fitness))

    # Calculate difference between final best/worst and initial best/worst, for the grouped bar chart
    y_difference_min = y_min[0] - y_min[-1]
    y_difference_max = initial_worst_fitness - max(population_fitness)

    data = [y_min, y_difference_min, y_difference_max]

    return population_fitness, data


"""
Perform the experiment based on the given parameters.

Runs 5 trials of the EA with the given parameters, and collects relevant data for analysis.
Parameters are passed directly to evolutionary_algorithm(). Refer to the corresponding documentation for more detail.
Returns various data for graphing and results.

"""
def experiment(crossover, k, p, bpp):

    # Data structures for results and graphs
    population_fitness_all = []
    results_data = []
    y_min_all = []
    y_difference_min_all = []
    y_difference_max_all = []

    # Run the 5 trials with seeds [1,2,3,4,5]. Collect and return relevant data after each trial.
    seed = 1
    for _ in range(5):
        # Run the trial
        population_fitness, data = evolutionary_algorithm(crossover=crossover, k=k, p=p, bpp=bpp, seed=seed)
        # Collect the data
        population_fitness_all += population_fitness
        results_data.append(min(population_fitness))
        y_min_all.append(data[0])
        y_difference_min_all.append(data[1])
        y_difference_max_all.append(data[2])
        # Update the seed for next iteration
        seed += 1

    # Post-processing (i.e. calculating averages)
    results_data.append(statistics.mean(results_data))
    y_min_avg = [float(sum(col)) / len(col) for col in zip(*y_min_all)]
    y_difference_min_avg = statistics.mean(y_difference_min_all)
    y_difference_max_avg = statistics.mean(y_difference_max_all)

    return population_fitness_all, results_data, y_min_avg, y_difference_min_avg, y_difference_max_avg


"""
Construct and print a DataFrame of the given results data.

"""
def show_results(results_data_all, title, bpp):

    # Prepare data
    data = {'Algorithm': [x for x in range(1, 7)],
            'Seed 1': [results_data_all[i][0] for i in range(6)],
            'Seed 2': [results_data_all[i][1] for i in range(6)],
            'Seed 3': [results_data_all[i][2] for i in range(6)],
            'Seed 4': [results_data_all[i][3] for i in range(6)],
            'Seed 5': [results_data_all[i][4] for i in range(6)],
            'Average': [results_data_all[i][5] for i in range(6)]
            }

    # Construct DataFrame
    df = pd.DataFrame(data, columns=['Algorithm', 'Seed 1', 'Seed 2', 'Seed 3', 'Seed 4', 'Seed 5', 'Average'])

    # Print title and DataFrame
    if bpp == 10:
        print("\t\t\t\t\t\t" + title)
    elif bpp == 100:
        print("\t\t\t\t\t\t\t\t" + title)
    print(df.to_string(index=False))


"""
Draw the three graphs using the provided data.
 
"""
def plot_graphs(population_fitness_all, best_avg_all, difference_min_avg_all, difference_max_avg_all):
    """
    Plot 1 - Line Graph. Average best fitness of each algorithm, from initial population to final population.
    """

    # Prepare x-axis data and line labels
    x = [1, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    labels = ["Algorithm 1", "Algorithm 2", "Algorithm 3", "Algorithm 4", "Algorithm 5", "Algorithm 6"]

    # Plot the line for each Algorithm
    for i in range(6):
        plt.plot(x, best_avg_all[i], label=labels[i])

    # Set title, x and y labels
    plt.title("Average best fitness of each algorithm, \nfrom initial population to final population")
    plt.xlabel('Generation')
    plt.ylabel('Average best fitness')

    # Plot legend
    plt.legend(loc='best')

    # Show the plot
    plt.show()

    """
    Plot 2 - Grouped bar chart. Average difference between initial best/worst solution and final best/worst solution.
    """

    # Credit: Adapted from example in Matplotlib documentation.
    # Available at https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html


    labels = [str(x + 1) for x in range(6)]
    loc = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(loc - width / 2, difference_min_avg_all, width, label='Best fitness difference')
    rects2 = ax.bar(loc + width / 2, difference_max_avg_all, width, label='Worst fitness difference')

    ax.set_ylabel('Fitness')
    ax.set_xlabel('Algorithm')
    ax.set_title('Difference between initial best/worst solutions and \nfinal best/worst solutions for each algorithm')
    ax.set_xticks(loc)
    ax.set_xticklabels(labels)
    ax.legend(loc='best')

    def autolabel(rects):
        for rect in rects:
            height = int(rect.get_height())
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show()

    """
    Plot 3 - Violin plot. Total fitness distribution of final populations in each algorithm.
    """

    # Build violin plot
    fig, ax = plt.subplots()
    ax.violinplot(population_fitness_all, showmeans=False, showmedians=True, vert=False)

    # Add title and axis labels
    ax.set_title('Violin plot showing total fitness distributions \nof the final populations in each algorithm')
    ax.set_xlabel('Fitness of final populations')
    ax.set_ylabel('Algorithm')

    # Add x-tick and y-tick labels
    ax.set_yticks(range(1, 7))
    ax.set_yticklabels(labels)

    # Add grid lines
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)

    # Show the plot
    plt.show()


if __name__ == "__main__":

    # Parameters for Algorithm 1-6
    parameters = [[True,  1, 10],
                  [True,  1, 100],
                  [True,  5, 10],
                  [True,  5, 100],
                  [False, 5, 10],
                  [True,  0, 10]]

    # BPP1
    population_fitness_all, results_data_all, best_avg_all, difference_min_avg_all, difference_max_avg_all = \
        zip(*[experiment(crossover=p[0],  k=p[1], p=p[2],  bpp=10) for p in parameters])

    show_results(results_data_all, "BPP1 Results", bpp=10)
    plot_graphs(population_fitness_all, best_avg_all, difference_min_avg_all, difference_max_avg_all)

    # BPP2
    population_fitness_all, results_data_all, best_avg_all, difference_min_avg_all, difference_max_avg_all = \
        zip(*[experiment(crossover=p[0], k=p[1], p=p[2], bpp=100) for p in parameters])

    show_results(results_data_all, "BPP2 Results", bpp=100)
    plot_graphs(population_fitness_all, best_avg_all, difference_min_avg_all, difference_max_avg_all)
