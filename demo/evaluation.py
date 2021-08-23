import argparse
from pathlib import Path
import numpy as np

import kapture
from kapture.io.csv import kapture_from_dir
from kapture.algo.pose_operations import pose_transform_distance

from .utils import parse_query_list, parse_submission


def evaluate(kapture_path: Path, poses_path: Path, queries_single: Path, queries_rigs: Path):
    skip_heavy_useless = [kapture.RecordsLidar, kapture.RecordsWifi,
                          kapture.Keypoints, kapture.Descriptors, kapture.GlobalFeatures,
                          kapture.Matches, kapture.Points3d, kapture.Observations]
    kapture_ = kapture_from_dir(kapture_path, skip_list=skip_heavy_useless)
    if kapture_.trajectories is None:
        raise ValueError('The query Kapture does not have ground truth poses.')

    Ts_w2c = parse_submission(poses_path)
    Ts_w2c_gt = kapture_.trajectories
    keys_single = parse_query_list(queries_single)
    keys_rigs = parse_query_list(queries_rigs)
    keys = keys_single + keys_rigs
    is_rig = np.array([False] * len(keys_single) + [True] * len(keys_rigs))

    err_r, err_t = [], []
    for key in keys:
        T_w2c_gt = Ts_w2c_gt[key]
        if key in Ts_w2c:
            dt, dr = pose_transform_distance(Ts_w2c[key].inverse(), T_w2c_gt.inverse())
            dr = np.rad2deg(dr)
        else:
            dr = np.inf
            dt = np.inf
        err_r.append(dr)
        err_t.append(dt)
    err_r = np.stack(err_r)
    err_t = np.stack(err_t)

    threshs = [(1, 0.1), (2, 0.25), (5, 1.)]
    recalls = [
        np.mean((err_r < th_r) & (err_t < th_t)) for th_r, th_t in threshs]
    recalls_single = [
        np.mean((err_r[~is_rig] < th_r) & (err_t[~is_rig] < th_t)) for th_r, th_t in threshs]
    recalls_rigs = [
        np.mean((err_r[is_rig] < th_r) & (err_t[is_rig] < th_t)) for th_r, th_t in threshs]

    results = {'recall': recalls,
               'recall_single': recalls_single,
               'recall_rigs': recalls_rigs,
               'Rt_thresholds': threshs}
    print('Results:', results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=Path, default=Path('data/'))
    parser.add_argument('--name', type=str, default='netvlad+superpoint+superglue')
    parser.add_argument('--output_path', type=Path, default=Path('./outputs/'))
    args = parser.parse_args()
    evaluate(
        args.data_path / 'query',
        args.output_path / args.name / 'results.txt',
        args.data_path / 'queries_single.txt',
        args.data_path / 'queries_rigs.txt')
