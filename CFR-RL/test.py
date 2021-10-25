from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags

from config import get_config
from env import Environment
from game import CFRRL_Game
from model import Network

FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt', '', 'apply a specific checkpoint')
flags.DEFINE_boolean('eval_delay', False, 'evaluate delay or not')


def count_rc_optimal(solutions, flows_list, lp_links, method):
    rc = []

    for i in range(len(solutions) - 1):
        count = 0
        solution_0 = solutions[i][method]
        solution_1 = solutions[i + 1][method]

        for flow_idx in flows_list:
            for e in lp_links:
                if solution_1[flow_idx, e[0], e[1]] - solution_0[flow_idx, e[0], e[1]] != 0.0:
                    count += 1
                    break

        rc.append(count)
    rc = np.asarray(rc)
    return rc


def count_rc_cfr(solutions, crit_pairs, lp_links, method):
    rc = []
    for i in range(len(solutions) - 1):
        count = 0
        solution_0 = solutions[i][method]
        solution_1 = solutions[i + 1][method]
        pairs_0 = crit_pairs[i][method]
        pairs_1 = crit_pairs[i + 1][method]

        intersec_flows = np.intersect1d(pairs_0, pairs_1)

        # counting the number of flows added to the list
        new_cflows = np.setdiff1d(pairs_1, intersec_flows)
        count += new_cflows.shape[0]

        # counting the number of flows that have path changed
        for flow_idx in intersec_flows:
            for e in lp_links:
                if solution_1[flow_idx, e[0], e[1]] - solution_0[flow_idx, e[0], e[1]] != 0.0:
                    count += 1
                    break

        rc.append(count)
    rc = np.asarray(rc)

    return rc


def count_rc(solutions, crit_pairs, lp_links, num_pairs):
    method = ['cfr-rl', 'cfr-topk', 'topk', 'optimal']
    num_rc = {}
    for m in range(len(method)):

        if method[m] == 'optimal':
            flows_list = np.arange(num_pairs)
            rc = count_rc_optimal(solutions=solutions, flows_list=flows_list,
                                  lp_links=lp_links, method=m)
            num_rc[method[m]] = rc

        else:
            rc = count_rc_cfr(solutions=solutions, crit_pairs=crit_pairs,
                              lp_links=lp_links, method=m)
            num_rc[method[m]] = rc

    return num_rc


def sim(config, network, game):


    f = open("PlotMAction1024.txt", "a+")
    for tm_idx in game.tm_indexes:
        state = game.get_state(tm_idx)

        if config.method == 'actor_critic':
            policy = network.actor_predict(np.expand_dims(state, 0)).numpy()[0]

        actions = game.chooseActionfromPolicy(policy)
        # actions_h = game.heuristic(timeLimit=200, tm_idx= tm_idx)

        game.DoAction(actions, tm_idx)
        # print(game.normal_routing(tm_idx),' ', game.old_mlu,' ',max(game.link_utilization), )
        # f.write(f'action  : {actions} \n')

        f.write(f'{game.normal_routing(tm_idx)} {game.old_mlu} {max(game.link_utilization)}\n')


    f.close()


def main(_):
    # Using cpu for testing
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.get_logger().setLevel('INFO')

    config = get_config(FLAGS) or FLAGS
    env = Environment(config, is_training=False)
    game = CFRRL_Game(config, env)
    network = Network(config, game.state_dims, game.action_dim, game.max_moves)

    step = network.restore_ckpt(FLAGS.ckpt)
    if config.method == 'actor_critic':
        learning_rate = network.lr_schedule(network.actor_optimizer.iterations.numpy()).numpy()
    elif config.method == 'pure_policy':
        learning_rate = network.lr_schedule(network.optimizer.iterations.numpy()).numpy()
    print('\nstep %d, learning rate: %f\n' % (step, learning_rate))

    sim(config, network, game)


if __name__ == '__main__':
    app.run(main)
