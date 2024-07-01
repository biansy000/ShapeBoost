import pickle as pk
from shapeboost.beta_decompose.beta_process2 import part_names, joints_name_24
from shapeboost.beta_decompose.beta_process_finer2 import vertice2capsule_finer, get_part_spine_finer
import copy

with open(f'shapeboost/beta_decompose/beta_process_finer2_split2.pkl', 'rb') as f:
    all_info2 = pk.load(f)


with open(f'shapeboost/beta_decompose/beta_process_finer2_split3.pkl', 'rb') as f:
    all_info3 = pk.load(f)


def get_info_split(split_num=2, use_all=False):
    if split_num == 2:
        all_info = copy.deepcopy(all_info2)
    else:
        all_info = copy.deepcopy(all_info3)
    
    finer_part_names = all_info.finer_part_names
    finer_part_seg = all_info.finer_part_seg
    part_names_ids_finer = all_info.part_names_ids_finer

    if not use_all:
        finer_part_names_list = [ 
            [f'{part_name}_{k}' for k in range(split_num)] for part_name in part_names
        ]

        finer_part_names = []
        for item in finer_part_names_list:
            finer_part_names += item
        
        for i, part_name_finer in enumerate(finer_part_names):

            k = part_name_finer.split('_')[:-1]
            k = '_'.join(k)

            part_names_ids_finer.append(joints_name_24.index(k))
    
    vertice2capsule_finer_new = lambda x, y: vertice2capsule_finer(x, y, part_names_finer=finer_part_names, part_seg_finer=finer_part_seg)
    get_part_spine_finer_new = lambda x, y: get_part_spine_finer(x, y, part_names_finer=finer_part_names)
    
    all_info['finer_part_names'] = finer_part_names
    all_info['part_names_ids_finer'] = part_names_ids_finer
    all_info['vertice2capsule_finer'] = vertice2capsule_finer_new
    all_info['get_part_spine_finer'] = get_part_spine_finer_new

    return all_info

