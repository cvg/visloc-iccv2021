from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt

from hloc.utils.viz import plot_images, plot_matches, save_plot, cm_RdGn
from hloc.utils.read_write_model import Image, Point3D
from hloc.utils.io import read_image


def plot_pnp_inliers(query_path: Path, ref_ids: List[int], ret: Dict,
                     sfm_images: Dict[int, Image], sfm_points: Dict[int, Point3D],
                     map_root: Path, num_pairs: int = 2):

    n = len(ref_ids)
    num_inliers = np.zeros(n)
    dbs_kp_q_db = [[] for _ in range(n)]
    inliers_dbs = [[] for _ in range(n)]
    inliers = ret.get('inliers', np.full(len(ret['p2d_q']), False))
    # for each pair of query keypoint and its matched 3D point,
    # we need to find its corresponding keypoint in each database image
    # that observes it. We also count the number of inliers in each.
    for i, (inl, (p3d_id, db_idxs)) in enumerate(zip(inliers, ret['p3d_matched_dbs'])):
        p3d = sfm_points[p3d_id]
        for db_idx in db_idxs:
            num_inliers[db_idx] += inl
            kp_db = p3d.point2D_idxs[p3d.image_ids == ref_ids[db_idx]][0]
            dbs_kp_q_db[db_idx].append((i, kp_db))
            inliers_dbs[db_idx].append(inl)

    idxs = np.argsort(num_inliers)[::-1][:num_pairs]
    qim = read_image(query_path)
    refs = [sfm_images[i].name for i in ref_ids]
    ims = []
    titles = []
    for i in idxs:
        ref = refs[i]
        rim = read_image(map_root / ref)
        ims.extend([qim, rim])
        inls = inliers_dbs[i]
        titles.extend([f'{sum(inls)}/{len(inls)}', Path(ref).name])
    plot_images(ims, titles)

    for i, idx in enumerate(idxs):
        color = cm_RdGn(np.array(inliers_dbs[idx]))
        dbs_kp_q_db_i = dbs_kp_q_db[idx]
        if len(dbs_kp_q_db_i) == 0:
            continue
        idxs_p2d_q, p2d_db = np.array(dbs_kp_q_db_i).T
        plot_matches(
            ret['p2d_q'][idxs_p2d_q],
            sfm_images[ref_ids[idx]].xys[p2d_db],
            color=color.tolist(),
            indices=(i*2, i*2+1), a=0.1)


def dump_plot(path: Path):
    path.parent.mkdir(exist_ok=True, parents=True)
    save_plot(path)
    plt.close()
