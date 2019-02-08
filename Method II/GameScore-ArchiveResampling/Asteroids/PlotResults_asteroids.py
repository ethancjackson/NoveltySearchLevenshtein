import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

results_file_gs = open('./run_info_gamescore.txt', 'r')
results_file_novelty = open('./run_info_archive_resampling.txt', 'r')

results_gs = []
for line in results_file_gs:
    if line.startswith('['):
        line = line.strip().strip('[]').split(',')
        results_gs.append([float(x) for x in line])

means_gs =  results_gs[0][:1000]
highs_gs = results_gs[1][:1000]
valid_gs = results_gs[2][:1000]

results_novelty = []
for line in results_file_novelty:
    if line.startswith('['):
        line = line.strip().strip('[]').split(',')
        results_novelty.append([float(x) for x in line])

means_novelty =  results_novelty[0][:1000]
highs_novelty = results_novelty[1][:1000]
valid_novelty = results_novelty[2][:1000]

def add_to_plot(ls, color, label=None):
    if label is None:
        plt.plot(range(len(ls)), ls, color=color)
    else:
        plt.plot(range(len(ls)), ls, label=label, color=color)

add_to_plot(means_gs, 'blue', 'Base GA')
add_to_plot(means_novelty, 'red', 'Method II')
plt.legend()
# plt.title('Space Invaders - Base GA vs. Method II - Population Mean Game Score')
plt.savefig('./MethodII-Asteroids-Means.pdf', dpi=1000)
plt.figure()

add_to_plot(highs_gs, 'blue', 'Base GA')
add_to_plot(highs_novelty, 'red', 'Method II')
plt.legend()
# plt.title('Space Invaders - Base GA vs. Method II - Top Score')
plt.savefig('./MethodII-Asteroids-Highs.pdf', dpi=1000)
plt.figure()

add_to_plot(valid_gs, 'blue', 'Base GA')
add_to_plot(valid_novelty, 'red', 'Method II')
plt.axhline(1629)
plt.legend()
# plt.title('Space Invaders - Base GA vs. Method II - Validation Score')
plt.savefig('./MethodII-Asteroids-Validation.pdf', dpi=1000)
# plt.show()

