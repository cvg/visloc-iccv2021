import logging
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from math import ceil
import numpy as np
from tqdm import tqdm

import pycolmap

import kapture
from kapture import Camera, PoseTransform
from kapture.io.csv import kapture_from_dir

from hloc.utils.parsers import parse_retrieval
from hloc.utils.read_write_model import read_model, Image, Point3D

from .utils import parse_query_list, get_keypoints, get_matches, camera_to_dict
from .visualization import plot_pnp_inliers, dump_plot


def localize(paths: Path, config: dict, num_visualize: int = 20) -> kapture.Trajectories:

    skip_heavy_useless = [kapture.Trajectories,
                          kapture.RecordsLidar, kapture.RecordsWifi,
                          kapture.Keypoints, kapture.Descriptors, kapture.GlobalFeatures,
                          kapture.Matches, kapture.Points3d, kapture.Observations]
    kapture_ = kapture_from_dir(paths.kapture_query, skip_list=skip_heavy_useless)
    rig_to_sensors = defaultdict(list)
    for rig_id, sensor_id in kapture_.rigs.key_pairs():
        rig_to_sensors[rig_id].append(sensor_id)

    keys_single = parse_query_list(paths.queries_single)
    keys_rigs = parse_query_list(paths.queries_rigs)
    poses = kapture.Trajectories()

    logging.info('Reading the sparse SfM model...')
    _, sfm_images, sfm_points = read_model(paths.sfm)
    sfm_name_to_id = {im.name: i for i, im in sfm_images.items()}
    pairs = parse_retrieval(paths.pairs_loc)

    logging.info('Localizing single queries...')
    for idx, (ts, camera_id) in enumerate(tqdm(keys_single)):
        name = kapture_.records_camera[ts, camera_id]
        camera = kapture_.sensors[camera_id]
        refs = pairs[name][:config['num_pairs_loc']]
        ref_ids = [sfm_name_to_id[n] for n in refs]
        T_world2cam, ret = estimate_camera_pose(
            name, ref_ids, camera, sfm_images, sfm_points,
            paths.lfeats, paths.matches_loc, config['pnp_reprojection_thresh'])
        if T_world2cam is not None:
            poses[ts, camera_id] = T_world2cam

        if num_visualize > 0 and idx % ceil(len(keys_single)/num_visualize) == 0:
            plot_pnp_inliers(
                paths.images_query / name, ref_ids, ret, sfm_images, sfm_points, paths.images_map)
            dump_plot(paths.viz / f'single_{ts}_{camera_id}.png')

    logging.info('Localizing camera rigs...')
    for idx, (ts, rig_id) in enumerate(tqdm(keys_rigs)):
        assert rig_id in rig_to_sensors, (rig_id, rig_to_sensors.keys())
        camera_ids = rig_to_sensors[rig_id]
        names = [kapture_.records_camera[ts, i] for i in camera_ids]
        cameras = [kapture_.sensors[i] for i in camera_ids]
        T_cams2rig = [kapture_.rigs[rig_id, i].inverse() for i in camera_ids]
        ref_ids = [[sfm_name_to_id[r] for r in pairs[q][:config['num_pairs_loc']]] for q in names]
        T_world2rig, ret = estimate_camera_pose_rig(
            names, ref_ids, cameras, T_cams2rig, sfm_images, sfm_points,
            paths.lfeats, paths.matches_loc, config['pnp_reprojection_thresh_rig'])

        # recover camera poses from the rig pose
        if T_world2rig is not None:
            poses[ts, rig_id] = T_world2rig

    return poses


def estimate_camera_pose(query: str, ref_ids: List[int], camera: Camera,
                         sfm_images: Dict[int, Image], sfm_points: Dict[int, Point3D],
                         query_features: Path, match_file: Path, thresh: float) -> PoseTransform:

    p2d, = get_keypoints(query_features, [query])
    p2d_to_p3d = defaultdict(list)
    p2d_to_p3d_to_dbs = defaultdict(lambda: defaultdict(list))
    num_matches = 0

    refs = [sfm_images[i].name for i in ref_ids]
    all_matches = get_matches(match_file, zip([query]*len(refs), refs))

    for idx, (ref_id, matches) in enumerate(zip(ref_ids, all_matches)):
        p3d_ids = sfm_images[ref_id].point3D_ids
        if len(p3d_ids) == 0:
            logging.warning('No 3D points found for %s.', sfm_images[ref_id].name)
            continue
        matches = matches[p3d_ids[matches[:, 1]] != -1]
        num_matches += len(matches)

        for i, j in matches:
            p3d_id = p3d_ids[j]
            p2d_to_p3d_to_dbs[i][p3d_id].append(idx)
            # avoid duplicate observations
            if p3d_id not in p2d_to_p3d[i]:
                p2d_to_p3d[i].append(p3d_id)

    idxs = list(p2d_to_p3d.keys())
    p2d_idxs = [i for i in idxs for _ in p2d_to_p3d[i]]
    p2d_m = p2d[p2d_idxs]
    p2d_m += 0.5  # COLMAP coordinates

    p3d_ids = [j for i in idxs for j in p2d_to_p3d[i]]
    p3d_m = [sfm_points[j].xyz for j in p3d_ids]
    p3d_m = np.array(p3d_m).reshape(-1, 3)

    # mostly for logging and post-processing
    p3d_matched_dbs = [(j, p2d_to_p3d_to_dbs[i][j])
                       for i in idxs for j in p2d_to_p3d[i]]

    ret = pycolmap.absolute_pose_estimation(p2d_m, p3d_m, camera_to_dict(camera), thresh)

    if ret['success']:
        T_w2cam = PoseTransform(ret['qvec'], ret['tvec'])
    else:
        T_w2cam = None

    ret = {
        **ret, 'p2d_q': p2d_m, 'p3d_r': p3d_m, 'p3d_ids': p3d_ids,
        'num_matches': num_matches, 'p3d_matched_dbs': p3d_matched_dbs}
    return T_w2cam, ret


def estimate_camera_pose_rig(queries: List[str], ref_ids_list: List[List[int]],
                             cameras: List[Camera], T_cams2rig: List[PoseTransform],
                             sfm_images: Dict[int, Image], sfm_points: Dict[int, Point3D],
                             query_features: Path, match_file: Path, thresh: float) -> PoseTransform:
    p2d_m_list = []
    p3d_m_list = []
    p3d_ids_list = []
    p3d_matched_dbs_list = []
    num_matches_list = []
    for query, ref_ids in zip(queries, ref_ids_list):
        p2d, = get_keypoints(query_features, [query])
        p2d_to_p3d = defaultdict(list)
        p2d_to_p3d_to_dbs = defaultdict(lambda: defaultdict(list))
        num_matches = 0

        refs = [sfm_images[i].name for i in ref_ids]
        all_matches = get_matches(match_file, zip([query]*len(refs), refs))

        for idx, (ref_id, matches) in enumerate(zip(ref_ids, all_matches)):
            p3d_ids = sfm_images[ref_id].point3D_ids
            if len(p3d_ids) == 0:
                logging.warning('No 3D points found for %s.', sfm_images[ref_id].name)
                continue
            matches = matches[p3d_ids[matches[:, 1]] != -1]
            num_matches += len(matches)

            for i, j in matches:
                p3d_id = p3d_ids[j]
                p2d_to_p3d_to_dbs[i][p3d_id].append(idx)
                # avoid duplicate observations
                if p3d_id not in p2d_to_p3d[i]:
                    p2d_to_p3d[i].append(p3d_id)

        idxs = list(p2d_to_p3d.keys())
        p2d_idxs = [i for i in idxs for _ in p2d_to_p3d[i]]
        p2d_m = p2d[p2d_idxs]
        p2d_m += 0.5  # COLMAP coordinates

        p3d_ids = [j for i in idxs for j in p2d_to_p3d[i]]
        p3d_m = [sfm_points[j].xyz for j in p3d_ids]
        p3d_m = np.array(p3d_m).reshape(-1, 3)

        # mostly for logging and post-processing
        p3d_matched_dbs = [(j, p2d_to_p3d_to_dbs[i][j])
                           for i in idxs for j in p2d_to_p3d[i]]

        # Save for pose estimation.
        p2d_m_list.append(p2d_m)
        p3d_m_list.append(p3d_m)
        p3d_ids_list.append(p3d_ids)
        p3d_matched_dbs_list.append(p3d_matched_dbs)
        num_matches_list.append(num_matches)

    camera_dicts = [camera_to_dict(camera) for camera in cameras]
    rel_poses = [T.inverse() for T in T_cams2rig]
    qvecs = [p.r_raw for p in rel_poses]
    tvecs = [p.t for p in rel_poses]

    ret = pycolmap.rig_absolute_pose_estimation(
        p2d_m_list, p3d_m_list, camera_dicts, qvecs, tvecs, thresh)

    if ret['success']:
        T_w2rig = PoseTransform(ret['qvec'], ret['tvec'])
    else:
        T_w2rig = None

    ret = {
        **ret, 'p2d_q': p2d_m_list, 'p3d_r': p3d_m_list, 'p3d_ids': p3d_ids_list,
        'num_matches': num_matches_list, 'p3d_matched_dbs': p3d_matched_dbs_list}
    return T_w2rig, ret
