import numpy as np
from estimator import ImportanceSampling, ImportanceSamplingUpdate, AIS
from model import StationCounting
import click


@click.command()
@click.option('--station', default='KBUF')
@click.option('--swor', is_flag = True)
@click.option('--cond', is_flag = True) # use cond instead of simp for the variance
@click.option('--var_weighting', default = 'LURE') # uniform / LURE weighting of variance estimates
@click.option('--r', default = 10) # number of rounds between two trainings
@click.option('--m', default = 1) # number of samples each round
@click.option('--t', default = 20) # number of trainings
def main(station, swor, cond, var_weighting, r, m, t):
    model = StationCounting(station=station)
    e = ImportanceSamplingUpdate(var_weighting)
    algo = AIS(model, e, swor, m=int(m), update_retro = r)
    latest_mean = 0
    for round in range(t):
        for s in range(r):
            print(f'Round {round*r+s}')
            algo.step(latest_mean)
            estimate, total_variance= algo.sqrt_mixer(simple_var=not cond)
            if not swor:
                latest_mean = estimate
            print(f'AIS weighting, mean: {estimate}, std: {np.sqrt(total_variance)}')
            estimate, total_variance = algo.sqrt_lure_mixer(simple_var=not cond)
            if swor:
                latest_mean = estimate
            print(f'AIS+LURE weighting, mean: {estimate}, std: {np.sqrt(total_variance)}')
            estimate, total_variance = algo.inv_var_mixer_smoothed(simple_var=not cond)
            print(f'Inverse variance weighting, mean: {estimate}, std: {np.sqrt(total_variance)}')

        algo.train()


if __name__ == '__main__':
    main()