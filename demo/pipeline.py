import argparse
from types import SimpleNamespace
from pathlib import Path

from hloc import extract_features, match_features, pairs_from_retrieval, triangulation
from hloc.utils.read_write_model import read_model, write_model

from kapture.io.records import get_record_fullpath
from kapture.converter.colmap.export_colmap import export_colmap
from kapture.io.csv import trajectories_to_file

from .utils import image_list_from_kapture, write_submission
from .localization import localize
from .evaluation import evaluate


CONFIG = {
        'name': 'netvlad+superpoint+superglue',
        'global_features': {
            'model': {'name': 'netvlad'},
            'preprocessing': {'resize_max': 1024},
        },
        'local_features': {
            'model': {
                'name': 'superpoint',
                'nms_radius': 3,
                'max_keypoints': 2048,
            },
            'preprocessing': {
                'grayscale': True,
                'resize_max': 1600,
            },
        },
        'matching': {
            'model': {
                'name': 'superglue',
                'weights': 'outdoor',
                'sinkhorn_iterations': 10,
            },
        },
        'num_pairs_sfm': 10,
        'num_pairs_loc': 10,
        'pnp_reprojection_thresh': 12.0,
        'pnp_reprojection_thresh_rig': 1.0,
}


def run(dataset_path: Path, map_name: str, query_name: str, output_path: Path, config: dict):
    outputs = output_path / config['name']
    outputs.mkdir(parents=True, exist_ok=True)
    paths = SimpleNamespace(
        gfeats='global_features.h5',
        lfeats='local_features.h5',
        matches_sfm='matches_sfm.h5',
        matches_loc='matches_loc.h5',
        pairs_sfm='pairs_sfm.txt',
        pairs_loc='pairs_loc.txt',
        sfm_empty='sfm_empty',
        sfm='sfm',
        viz='viz',
        query_poses='query_poses.txt',
        results='results.txt',
    )
    for k, v in paths.__dict__.items():
        setattr(paths, k, outputs / v)
    paths.kapture_map = dataset_path / map_name
    paths.kapture_query = dataset_path / query_name
    paths.images_map = Path(get_record_fullpath(paths.kapture_map))
    paths.images_query = Path(get_record_fullpath(paths.kapture_query))
    paths.queries_single = dataset_path / 'queries_single.txt'
    paths.queries_rigs = dataset_path / 'queries_rigs.txt'

    images_map = image_list_from_kapture(paths.kapture_map)
    images_query = image_list_from_kapture(paths.kapture_query)

    # MAPPING
    extract_features.main(
        config['global_features'], paths.images_map, feature_path=paths.gfeats,
        image_list=images_map, as_half=True)
    pairs_from_retrieval.main(
        paths.gfeats, paths.pairs_sfm, config['num_pairs_sfm'],
        query_list=images_map, db_list=images_map)

    extract_features.main(
        config['local_features'], paths.images_map, feature_path=paths.lfeats,
        image_list=images_map, as_half=True)
    match_features.main(
        config['matching'], paths.pairs_sfm, paths.lfeats, matches=paths.matches_sfm)

    export_colmap(paths.kapture_map, paths.sfm_empty / 'colmap.db', paths.sfm_empty,
                  force_overwrite_existing=True)
    write_model(*read_model(paths.sfm_empty, ext='.txt'), paths.sfm_empty)
    if not paths.sfm.exists():
        triangulation.main(
            paths.sfm, paths.sfm_empty, paths.images_map,
            paths.pairs_sfm, paths.lfeats, paths.matches_sfm)

    # LOCALIZATION
    extract_features.main(
        config['global_features'], paths.images_query, feature_path=paths.gfeats,
        image_list=images_query, as_half=True)
    pairs_from_retrieval.main(
        paths.gfeats, paths.pairs_loc, config['num_pairs_loc'],
        query_list=images_query, db_list=images_map)

    extract_features.main(
        config['local_features'], paths.images_query, feature_path=paths.lfeats,
        image_list=images_query, as_half=True)
    match_features.main(
        config['matching'], paths.pairs_loc, paths.lfeats, matches=paths.matches_loc)

    query_poses = localize(paths, config)
    trajectories_to_file(paths.query_poses, query_poses)
    write_submission(paths.results, query_poses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_path', type=Path, default=Path('data/'),
                        help='path to the top-level directory of the dataset')
    parser.add_argument('--map_name', type=str, default='mapping',
                        help='name of the Kapture dataset of the map')
    parser.add_argument('--query_name', type=str, default='query',
                        help='name of the Kapture dataset of the queries')
    parser.add_argument('--output_path', type=Path, default=Path('./outputs/'),
                        help='path to the output directory')
    args = parser.parse_args()
    run(**args.__dict__, config=CONFIG)
