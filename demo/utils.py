from pathlib import Path
from typing import List, Tuple, Iterator
import numpy as np
import h5py

import kapture
from kapture import Trajectories, PoseTransform
from kapture.io.csv import kapture_from_dir

from hloc.utils.parsers import names_to_pair


def image_list_from_kapture(kapture_path: Path) -> List[str]:
    skip_heavy_useless = [kapture.Trajectories,
                          kapture.RecordsLidar, kapture.RecordsWifi,
                          kapture.Keypoints, kapture.Descriptors, kapture.GlobalFeatures,
                          kapture.Matches, kapture.Points3d, kapture.Observations]
    kapture_ = kapture_from_dir(kapture_path, skip_list=skip_heavy_useless)
    image_list = [name for _, _, name in kapture.flatten(kapture_.records_camera, is_sorted=True)]
    return image_list


def parse_query_list(path: Path) -> List[Tuple[int, str]]:
    keys = []
    with open(path, 'r') as fid:
        for line in fid:
            line = line.strip('\n')
            if len(line) == 0 or line[0] == '#':
                continue
            timestamp, camera_id = line.split('/')
            keys.append((int(timestamp), camera_id))
    return keys


def parse_submission(path: Path) -> Trajectories:
    poses = Trajectories()
    with open(path, 'r') as fid:
        for line in fid:
            line = line.strip('\n')
            if len(line) == 0 or line[0] == '#':
                continue
            name, qw, qx, qy, qz, tx, ty, tz = line.split(' ')
            pose = PoseTransform(np.array([qw, qx, qy, qz], float), np.array([tx, ty, tz], float))
            timestamp, camera_id = name.split('/')
            poses[(int(timestamp), camera_id)] = pose
    return poses


def write_submission(path: Path, poses: Trajectories):
    with open(path, 'w') as fid:
        for timestamp, camera_id in poses.key_pairs():
            pose = poses[timestamp, camera_id]
            name = f'{timestamp}/{camera_id}'
            data = [name] + np.concatenate((pose.r_raw, pose.t_raw)).astype(str).tolist()
            fid.write(' '.join(data) + '\n')



def get_keypoints(feats_path: Path, keys: Iterator[str]) -> List[np.ndarray]:
    with h5py.File(feats_path, 'r') as fid:
        keypoints = [fid[str(k)]['keypoints'].__array__() for k in keys]
    return keypoints


def get_matches(matches_path: Path, key_pairs: Iterator[Tuple[str]]) -> List[np.ndarray]:
    matches = []
    with h5py.File(matches_path, 'r') as fid:
        for k1, k2 in key_pairs:
            pair = names_to_pair(str(k1), str(k2))
            m = fid[pair]['matches0'].__array__()
            idx = np.where(m != -1)[0]
            m = np.stack([idx, m[idx]], -1)
            matches.append(m)
    return matches


def camera_to_dict(camera: kapture.Camera) -> dict:
    model, w, h, *params = camera.sensor_params
    return {
        'model': model,
        'width': int(w),
        'height': int(h),
        'params': np.array(params, float),
    }
