import pickle as pk

with open('extra_files/beta_process_finer.pkl', 'rb') as f:
    part_names_finer, part_seg_finer, part_root_joints_finer, spine_joints_finer, mean_part_width_ratio_dict_finer, part_seg_lens_finer, \
        part_pairs_finer, part_names_ids_finer, mean_part_width_finer0 = pk.load(f)

mean_part_width_ratio_finer = [mean_part_width_ratio_dict_finer[n] for n in part_names_finer]
mean_part_width_finer = mean_part_width_finer0[0]

