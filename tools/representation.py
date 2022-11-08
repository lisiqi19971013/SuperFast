import torch
import torch as th
import tools.event as event


def _split_coordinate(c):
    c = c.float()
    left_c = c.floor()
    right_weight = c - left_c
    left_c = left_c.int()
    right_c = left_c + 1
    return left_c, right_c, right_weight


def _to_lin_idx(t, x, y, W, H, B):
    mask = (0 <= x) & (0 <= y) & (0 <= t) & (x <= W-1) & (y <= H-1) & (t <= B-1)
    lin_idx = x.long() + y.long() * W + t.long() * W * H
    return lin_idx, mask


def to_voxel_grid(event_sequence, nb_of_time_bins=5, remapping_maps=None):
    """Returns voxel grid representation of event steam.

    In voxel grid representation, temporal dimension is
    discretized into "nb_of_time_bins" bins. The events fir
    polarities are interpolated between two near-by bins
    using bilinear interpolation and summed up.

    If event stream is empty, voxel grid will be empty.
    """
    voxel_grid = th.zeros(nb_of_time_bins, event_sequence._image_height, event_sequence._image_width, dtype=th.float32, device='cpu')

    voxel_grid_flat = voxel_grid.flatten()

    # Convert timestamps to [0, nb_of_time_bins] range.
    duration = event_sequence.duration()
    start_timestamp = event_sequence.start_time()
    features = th.from_numpy(event_sequence._features)
    x = features[:, event.X_COLUMN]
    y = features[:, event.Y_COLUMN]
    polarity = features[:, event.POLARITY_COLUMN].float()
    t = (features[:, event.TIMESTAMP_COLUMN] - start_timestamp) * (nb_of_time_bins - 1) / duration
    t = t.float()

    if remapping_maps is not None:
        remapping_maps = th.from_numpy(remapping_maps)
        x, y = remapping_maps[:,y,x]

    # left_t, right_t = t.floor(), t.floor()+1
    # left_x, right_x = x.float().floor(), x.float().floor()+1
    # left_y, right_y = y.float().floor(), y.float().floor()+1
    #
    # for lim_x in [left_x, right_x]:
    #     for lim_y in [left_y, right_y]:
    #         for lim_t in [left_t, right_t]:
    #             mask = (0 <= lim_x) & (0 <= lim_y) & (0 <= lim_t) & (lim_x <= event_sequence._image_width-1) \
    #                    & (lim_y <= event_sequence._image_height-1) & (lim_t <= nb_of_time_bins-1)
    #
    #             # we cast to long here otherwise the mask is not computed correctly
    #             lin_idx = lim_x.long() \
    #                       + lim_y.long() * event_sequence._image_width \
    #                       + lim_t.long() * event_sequence._image_width * event_sequence._image_height
    #
    #             weight = polarity * (1-(lim_x-x).abs()) * (1-(lim_y-y).abs()) * (1-(lim_t-t).abs())
    #             voxel_grid_flat.index_add_(dim=0, index=lin_idx[mask], source=weight[mask].float())

    lim_x = x
    lim_y = y
    lim_t = t
    mask = (0 <= lim_x) & (0 <= lim_y) & (0 <= lim_t) & (lim_x <= event_sequence._image_width-1) \
           & (lim_y <= event_sequence._image_height-1) & (lim_t <= nb_of_time_bins-1)

    lin_idx = lim_x.long() + lim_y.long() * event_sequence._image_width \
              + lim_t.long() * event_sequence._image_width * event_sequence._image_height

    weight = polarity
    voxel_grid_flat.index_add_(dim=0, index=lin_idx[mask], source=weight[mask].float())
    return voxel_grid


def to_count_map(event_sequence, nb_of_time_bins=5):
    ecm = th.zeros(2, event_sequence._image_height, event_sequence._image_width, nb_of_time_bins, dtype=th.float32, device='cpu')
    ecm = ecm.flatten()

    duration = event_sequence.duration()
    # print(duration)
    start_timestamp = event_sequence.start_time()
    features = th.from_numpy(event_sequence._features)
    x = features[:, event.X_COLUMN]
    y = features[:, event.Y_COLUMN]
    polarity = (features[:, event.POLARITY_COLUMN].float() + 1)/2
    t = (features[:, event.TIMESTAMP_COLUMN] - start_timestamp) * (nb_of_time_bins) / (duration+1)

    mask = (0 <= x) & (0 <= y) & (0 <= t) & (x <= event_sequence._image_width-1) \
           & (y <= event_sequence._image_height-1) & (t.long() <= nb_of_time_bins-1)

    lin_idx = t.long() + x.long() * nb_of_time_bins + y.long() * event_sequence._image_width * nb_of_time_bins + \
              polarity.long() * nb_of_time_bins * event_sequence._image_width * event_sequence._image_height
    src = th.ones(lin_idx.shape)
    ecm.put_(lin_idx[mask], src[mask], accumulate=True)
    ecm = ecm.reshape(2, event_sequence._image_height, event_sequence._image_width, nb_of_time_bins)
    return ecm


# if __name__ == '__main__':
#     from model_old.snnModel.dataLoader import Dataset
#     import numpy as np
#     event_folder = '/repository/lisiqi/data/cup/cup1/final1'
#     dvs_image_folder = '/repository/lisiqi/data/cup/cup1/final1/dvs_cut'
#     gt_image_folder = '/repository/lisiqi/data/cup/cup1/final1/frame_cut'
#
#     d = Dataset(event_folder, dvs_image_folder, gt_image_folder)
#     event_sequence = d.storage._events.filter_by_timestamp(50000, 5000)
#     ecm = to_count_map(event_sequence)
#
#     def calEcm(event, shape=[2, 200, 300]):
#         ecm = np.zeros(shape)
#         for e in event:
#             if e[-1] == 1:
#                 ecm[1, e[1], e[0]] += 1
#             else:
#                 ecm[0, e[1], e[0]] += 1
#         return ecm
#
#     e = event_sequence._features
#     e[:,2] = (e[:,2] - event_sequence.start_time()) * 5 / event_sequence.duration()
#     e1 = e[e[:,2] == 0, :]
#     ecm1 = calEcm(e1)
#     print((torch.tensor(ecm1) - ecm[:,:,:,0]).sum())


