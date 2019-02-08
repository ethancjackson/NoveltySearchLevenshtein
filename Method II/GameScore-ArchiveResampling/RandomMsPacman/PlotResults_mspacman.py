import matplotlib.pyplot as plt

results_file_gs = open('./run_info_gamescore.txt', 'r')
results_file_novelty = open('./run_info_archive_resampling.txt', 'r')
results_file_random = open('./run_info_random.txt', 'r')

results_gs = []
for line in results_file_gs:
    if line.startswith('['):
        line = line.strip().strip('[]').split(',')
        results_gs.append([float(x) for x in line])

means_gs =  results_gs[0]
highs_gs = results_gs[1]
valid_gs = results_gs[2]

results_novelty = []
for line in results_file_novelty:
    if line.startswith('['):
        line = line.strip().strip('[]').split(',')
        results_novelty.append([float(x) for x in line])

means_novelty =  results_novelty[0]
highs_novelty = results_novelty[1]
valid_novelty = results_novelty[2]

results_random = []
for line in results_file_random:
    if line.startswith('['):
        line = line.strip().strip('[]').split(',')
        results_random.append([float(x) for x in line])

means_random =  results_random[0]
highs_random = results_random[1]
valid_random = results_random[2]

def add_to_plot(ls, color, label=None):
    if label is None:
        plt.plot(range(len(ls)), ls, color=color)
    else:
        plt.plot(range(len(ls)), ls, label=label, color=color)

add_to_plot(means_gs, 'blue', 'Game Score')
add_to_plot(means_novelty, 'red', 'Novelty')
add_to_plot(means_random, 'purple', 'Random')
plt.legend()
plt.title('MsPacman - GS vs. Archive Resampling - Population Mean')
plt.figure()

add_to_plot(highs_gs, 'blue', 'Game Score')
add_to_plot(highs_novelty, 'red', 'Novelty')
add_to_plot(highs_random, 'purple', 'Random')
plt.legend()
plt.title('MsPacman - GS vs. Archive Resampling - Top Score')
plt.figure()

add_to_plot(valid_novelty, 'red', 'Novelty')
add_to_plot(valid_gs, 'blue', 'Game Score')
add_to_plot(valid_random, 'purple', 'Random')
plt.legend()
plt.title('MsPacman - GS vs. Archive Resampling - Validation Score')
plt.show()
