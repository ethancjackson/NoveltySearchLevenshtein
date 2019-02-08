from Individual import Individual, from_JSON
from Mutator import mutate
from numpy.random import randint, RandomState
from NetworkMapper import mutate_conv_module
from joblib import Parallel, delayed
import numpy as np
import KerasEvaluator
from numpy import fmax
import KerasConvModules
import gc
import sys
from itertools import product
import time
import matplotlib.pyplot as plt



def generate_individual(in_shape, out_shape, init_connect_rate, init_seed):
    return Individual(in_shape=in_shape, out_shape=out_shape, init_seed=init_seed, init_connect_rate=init_connect_rate)


def generate_init_population(in_shape, out_shape, num_individuals=10, init_connect_rate=0.5):
    population = []
    for i in range(num_individuals):
        # init_seed = randint(low=0, high=4294967295)
        ind = generate_individual(in_shape, out_shape, init_connect_rate, i)
        population.append(ind)
    return population



def play_atari(individual, game_name, game_iterations, env_seeds, vis=False, sleep=0.05):
    """
    Given an indivudual, play an Atari game and return the score.
    :param individual:
    :return: sequence of actions as numpy array
    """

    from cv2 import cvtColor, COLOR_RGB2GRAY, resize
    import keras as K
    import gym.spaces

    # Initialize Keras model
    conv_module = KerasConvModules.gen_DQN_architecture(input_shape=individual.in_shape,
                                                        output_size=individual.out_size,
                                                        seed=individual.init_seed)

    mutate_conv_module(individual, conv_module, mut_power=0.002)
    me = KerasEvaluator.KerasEvaluator(conv_module)

    # Integer RNG for initial game actions
    rnd_int = RandomState(seed=individual.init_seed).randint

    total_reward = 0
    episode_actions_performed = []
    env = gym.make(game_name)
    action_space_size = env.action_space.n

    for env_seed in env_seeds:

        actions_performed = []
        env.seed(env_seed)
        x = resize(env.reset(), (84, 84))
        x_lum = cvtColor(x, COLOR_RGB2GRAY).reshape(84, 84, 1)
        x = np.concatenate((x, x_lum), axis=2)
        x = x.reshape(1, 84, 84, 4)
        prev_frame = np.zeros_like(x)

        for _ in range(30):
            action = rnd_int(low=0, high=action_space_size)
            env.step(action)
            actions_performed.append(action)

        for i in range(game_iterations):

            if vis:
                env.render()
                time.sleep(sleep)


            # Evaluate network
            step = me.eval(fmax(x, prev_frame))
            
            # # Testing - store images
            # obs = fmax(x, prev_frame)[0][:,:,:3]
            # print(obs.shape)
            # plt.imshow(obs)
            # plt.savefig('./Images/{}.png'.format(i))

            # Store frame for reuse
            prev_frame = x

            # Compute one-hot boolean action - THIS WILL DEPEND ON THE GAME
            y_ = step.flatten()
            a = y_.argmax()
            actions_performed.append(a)
            observation, reward, done, info = env.step(a)

            if done:
                # Ensure all action sequences have the same length
                extra_actions = game_iterations - i - 1
                # for _ in range(extra_actions):
                #     actions_performed.append('x')
                break
            total_reward += reward

            # Use previous frame information to avoid flickering
            x = resize(observation, (84, 84))
            x_lum = cvtColor(x, COLOR_RGB2GRAY).reshape(84, 84, 1)
            x = np.concatenate((x, x_lum), axis=2)
            x = x.reshape(1, 84, 84, 4)

            # sys.stdout.write("Game iteration:{}\tAction:{}\r".format(i, a))
            # sys.stdout.flush()
        episode_actions_performed.append(actions_performed)

    env.close()
    K.backend.clear_session()
    del me

    sys.stdout.write('{}{}\n'.format(individual, total_reward))
    sys.stdout.flush()
    return (individual, total_reward, episode_actions_performed)


### Testing over 100 independent episodes
game_names = ['SpaceInvaders-v0', 'MsPacman-v0', 'Asteroids-v0', 'Assault-v0']
game_dirs = ['./SpaceInvaders', './MsPacman', './Asteroids', './Assault']
game_names = ['SpaceInvaders-v0']
game_dirs = ['./SpaceInvaders']
gs_score_dict = {}
rs_score_dict = {}
gs_lifetime_dict = {}
rs_lifetime_dict = {}

for index, game_name in enumerate(game_names):
    game_dir = game_dirs[index]
    gs_ind_file = open('{}/GameScore-Validation-Best.txt'.format(game_dir), 'r')
    rs_ind_file = open('{}/PureNovelty-Validation-Best.txt'.format(game_dir), 'r')
    gs_ind = gs_ind_file.readlines()[0]
    rs_ind = rs_ind_file.readlines()[0]
    gs_ind = from_JSON(gs_ind)
    rs_ind = from_JSON(rs_ind)
    # episodes = list(range(31,61))
    # episodes = list(range(1,31))
    episodes = list(range(203,204))
    gs_scores = []
    rs_scores = []
    gs_lifetimes = []
    rs_lifetimes = []
    for ep in episodes:
        print('{} - GameScore'.format(game_name))
        i, score, actions = play_atari(gs_ind, game_name, 20000, [ep], vis=True, sleep=0.01)
        print(len(actions[0]))
        gs_lifetimes.append(len(actions[0]))
        print(score)
        gs_scores.append(score)
        print('{} - PureNovelty'.format(game_name))
        i, score, actions = play_atari(rs_ind, game_name, 20000, [ep], vis=True, sleep=0.01)
        print(len(actions[0]))
        rs_lifetimes.append(len(actions[0]))
        print(score)
        rs_scores.append(score)
    gs_lifetime_dict[game_name] = gs_lifetimes
    rs_lifetime_dict[game_name] = rs_lifetimes
    gs_score_dict[game_name] = gs_scores
    rs_score_dict[game_name] = rs_scores

#
# results_file = open('./testing_results_100eps.txt', 'w')
# for game_name in game_names:
#     results_file.write('{}\n'.format(game_name))
#     results_file.write('GameScore: {}\n'.format(gs_score_dict[game_name]))
#     results_file.write('ArchiveResample: {}\n'.format(rs_score_dict[game_name]))
# results_file.flush()
# results_file.close()
#
# import numpy as np
# from scipy.stats import ttest_ind
#
# for game_name in game_names:
#     print(game_name)
#     gs_mean = np.mean(gs_score_dict[game_name])
#     gs_std = np.std(gs_score_dict[game_name])
#     rs_mean = np.mean(rs_score_dict[game_name])
#     rs_std = np.std(rs_score_dict[game_name])
#     print('GameScore Mean: {}'.format(gs_mean))
#     print('GameScore StDev: {}'.format(gs_std))
#     print('Resample Mean: {}'.format(rs_mean))
#     print('Resample StDev: {}'.format(rs_std))
#     print()
#     test_stat, p_value = ttest_ind(gs_score_dict[game_name], rs_score_dict[game_name])
#     print('p-value: {}'.format(p_value))
#     print()

##################################################
### TESTING RANDOM ARCHIVE ONLY MSPACMAN ###

# ### Testing over 100 independent episodes
# game_names = ['MsPacman-v0']
# game_dirs = ['./RandomMsPacMan']
# gs_score_dict = {}
# rs_score_dict = {}
#
# for index, game_name in enumerate(game_names):
#     game_dir = game_dirs[index]
#     gs_ind_file = open('{}/ArchiveResample-ValidationBest.txt'.format(game_dir), 'r')
#     rs_ind_file = open('{}/RandomArchive-ValidationBest.txt'.format(game_dir), 'r')
#     gs_ind = gs_ind_file.readlines()[0]
#     rs_ind = rs_ind_file.readlines()[0]
#     gs_ind = from_JSON(gs_ind)
#     rs_ind = from_JSON(rs_ind)
#     # episodes = list(range(31,61))
#     # episodes = list(range(1,31))
#     episodes = list(range(100, 200))
#     gs_scores = []
#     rs_scores = []
#     for ep in episodes:
#         print('{} - Novelty Resampling'.format(game_name))
#         i, score, actions = play_atari(gs_ind, game_name, 20000, [ep], vis=False, sleep=0.004)
#         print(score)
#         gs_scores.append(score)
#         print('{} - Random Resampling'.format(game_name))
#         i, score, actions = play_atari(rs_ind, game_name, 20000, [ep], vis=False, sleep=0.004)
#         print(score)
#         rs_scores.append(score)
#     gs_score_dict[game_name] = gs_scores
#     rs_score_dict[game_name] = rs_scores
#
# results_file = open('./random_archive_testing_results_100eps.txt', 'w')
# for game_name in game_names:
#     results_file.write('{}\n'.format(game_name))
#     results_file.write('ArchiveResample: {}\n'.format(gs_score_dict[game_name]))
#     results_file.write('RandomResample: {}\n'.format(rs_score_dict[game_name]))
# results_file.flush()
# results_file.close()
#
# import numpy as np
# from scipy.stats import ttest_ind
#
# for game_name in game_names:
#     print(game_name)
#     gs_mean = np.mean(gs_score_dict[game_name])
#     gs_std = np.std(gs_score_dict[game_name])
#     rs_mean = np.mean(rs_score_dict[game_name])
#     rs_std = np.std(rs_score_dict[game_name])
#     print('Novelty Mean: {}'.format(gs_mean))
#     print('Novelty StDev: {}'.format(gs_std))
#     print('Random Mean: {}'.format(rs_mean))
#     print('Random StDev: {}'.format(rs_std))
#     print()
#     test_stat, p_value = ttest_ind(gs_score_dict[game_name], rs_score_dict[game_name])
#     print('p-value: {}'.format(p_value))
#     print()
#
