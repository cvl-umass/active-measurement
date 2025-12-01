import numpy as np
from estimator import ImportanceSampling, ImportanceSamplingUpdate, AIS
from model import DetectorCounting
import click

@click.command()
@click.option('--path', default='../data/DSC5214_data.csv')
@click.option('--ckpt_dir', default='../finetune/DSC5214')
@click.option('--swor', is_flag = True)
@click.option('--cond', is_flag = True) # use cond instead of simp for the variance
@click.option('--var_weighting', default = 'LURE') # uniform / LURE weighting of variance estimates
@click.option('--r', default = 10) # number of rounds between two trainings
@click.option('--m', default = 1) # number of samples each round
@click.option('--t', default = 10) # number of trainings
@click.option('--seed', default=0) # random seed
def main(path, ckpt_dir, swor, cond, var_weighting, r, m, t, seed):
    model = DetectorCounting(seed=seed, path=path, output_dir=ckpt_dir)
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