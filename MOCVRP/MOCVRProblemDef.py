import torch
import numpy as np


def get_training_problems(batch_size, problem_size):

    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    if problem_size == 20 or problem_size == 30 or problem_size == 40:
        demand_scaler = 30
    elif problem_size == 50 or problem_size == 60 or problem_size == 70:
        demand_scaler = 40
    elif problem_size == 80 or problem_size == 90 or problem_size == 100:
        demand_scaler = 50
    else:
        raise NotImplementedError

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)

    preference = torch.Tensor(batch_size, 2).uniform_(1e-6, 1)

    return depot_xy, node_xy, node_demand, preference


def get_random_problems(batch_size, problem_size):

    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    if problem_size == 20 or problem_size == 30 or problem_size == 40:
        demand_scaler = 30
    elif problem_size == 50 or problem_size == 60 or problem_size == 70:
        demand_scaler = 40
    elif problem_size == 80 or problem_size == 90 or problem_size == 100:
        demand_scaler = 50
    else:
        raise NotImplementedError

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)

    preference = torch.Tensor(batch_size, 2).uniform_(1e-6, 1)

    return depot_xy, node_xy, node_demand, preference

def augment_xy_data_by_8_fold(xy_data):
    
    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    
    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1-x, y), dim=2)
    dat3 = torch.cat((x, 1-y), dim=2)
    dat4 = torch.cat((1-x, 1-y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1-y, x), dim=2)
    dat7 = torch.cat((y, 1-x), dim=2)
    dat8 = torch.cat((1-y, 1-x), dim=2)

    data_augmented = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)

    return data_augmented

def augment_preference(pref):
    coff = torch.Tensor(7).uniform_(1e-6, 1)
    new_pref = []
    new_pref.append(pref)
    for i in range(len(coff)):
        new_pref.append(coff[i] * pref)

    return torch.cat(new_pref, dim=0)