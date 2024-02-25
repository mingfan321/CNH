import torch
import numpy as np

def get_random_problems(batch_size, problem_size):
    instances = torch.rand(size=(batch_size, problem_size, 6))
    preference = torch.Tensor(batch_size, 3).uniform_(1e-6, 1)
    # preference = torch.zeros(size=(0, 3))
    # for i in range(batch_size):
    #     r = np.random.rand(1)
    #     if r < 0.5:
    #         aa = np.random.randint(0, 3)
    #         weights = torch.zeros(3).cuda()
    #         weights[aa] = 1
    #         weights = weights / torch.sum(weights)
    #         preference = torch.cat((preference, weights[None, :]), 0)
    #     else:
    #         pref = torch.Tensor(1, 3).uniform_(1e-6, 1)
    #         preference = torch.cat((preference, pref), 0)

    problems = {
        'instances': instances,
        'preference': preference
    }
    return problems


def augment_xy_data_by_n_fold_3obj(xy_data, n):

    size = n

    x1 = xy_data[:, :, [0]]
    y1 = xy_data[:, :, [1]]
    x2 = xy_data[:, :, [2]]
    y2 = xy_data[:, :, [3]]
    x3 = xy_data[:, :, [4]]
    y3 = xy_data[:, :, [5]]
    
    dat1 = {}
    dat2 = {}
    dat3 = {}
    
    dat_aug = []
    
    dat1[0] = torch.cat((x1, y1), dim=2)
    dat1[1]= torch.cat((1-x1, y1), dim=2)
    dat1[2] = torch.cat((x1, 1-y1), dim=2)
    dat1[3] = torch.cat((1-x1, 1-y1), dim=2)
    dat1[4]= torch.cat((y1, x1), dim=2)
    dat1[5] = torch.cat((1-y1, x1), dim=2)
    dat1[6] = torch.cat((y1, 1-x1), dim=2)
    dat1[7] = torch.cat((1-y1, 1-x1), dim=2)
    
    dat2[0] = torch.cat((x2, y2), dim=2)
    dat2[1]= torch.cat((1-x2, y2), dim=2)
    dat2[2] = torch.cat((x2, 1-y2), dim=2)
    dat2[3] = torch.cat((1-x2, 1-y2), dim=2)
    dat2[4]= torch.cat((y2, x2), dim=2)
    dat2[5] = torch.cat((1-y2, x2), dim=2)
    dat2[6] = torch.cat((y2, 1-x2), dim=2)
    dat2[7] = torch.cat((1-y2, 1-x2), dim=2)
    
    dat3[0] = torch.cat((x3, y3), dim=2)
    dat3[1]= torch.cat((1-x3, y3), dim=2)
    dat3[2] = torch.cat((x3, 1-y3), dim=2)
    dat3[3] = torch.cat((1-x3, 1-y3), dim=2)
    dat3[4]= torch.cat((y3, x3), dim=2)
    dat3[5] = torch.cat((1-y3, x3), dim=2)
    dat3[6] = torch.cat((y3, 1-x3), dim=2)
    dat3[7] = torch.cat((1-y3, 1-x3), dim=2)
    
    all_idx = [[i, j, k] for i in range(8) for j in range(8) for k in range(8)]
    item_list = list(range(512))
    np.random.shuffle(item_list)
    
    for i in range(size):
        idx = all_idx[item_list[i]]
        dat = torch.cat((dat1[idx[0]], dat2[idx[1]], dat3[idx[2]]), dim=2)
        dat_aug.append(dat)
    aug_problems = torch.cat(dat_aug, dim=0)
   
    return aug_problems

def augment_preference(pref, size):
    coff = torch.Tensor(size - 1).uniform_(1e-6, 1)
    new_pref = []
    new_pref.append(pref)
    for i in range(len(coff)):
        new_pref.append(coff[i] * pref)

    return torch.cat(new_pref, dim=0)



