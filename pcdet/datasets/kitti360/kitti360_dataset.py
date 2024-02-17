import numpy as np
from ..dataset import DatasetTemplate
import glob
import os

from typing import Iterator, Union, Optional, Tuple
from pathlib import Path

def list_dir_or_file(dir_path: Union[str, Path],
                        list_dir: bool = True,
                        list_file: bool = True,
                        suffix: Optional[Union[str, Tuple[str]]] = None,
                        recursive: bool = False) -> Iterator[str]:
        """
        List directories and/or files within a specified path.

        Args:
            dir_path (str | Path): Path of the directory.
            list_dir (bool): List the directories. Default: True.
            list_file (bool): List the path of files. Default: True.
            suffix (str or tuple[str], optional): File suffix that we are interested in. Default: None.
            recursive (bool): If set to True, recursively scan the directory. Default: False.

        Yields:
            Iterator[str]: A relative path to `dir_path`.
        """

        if not isinstance(dir_path, Path):
            dir_path = Path(dir_path)

        if suffix is not None and not isinstance(suffix, (str, tuple)):
            raise TypeError('`suffix` must be a string or tuple of strings')

        if list_dir and suffix is not None:
            raise TypeError('`list_dir` should be False when `suffix` is not None')

        def _list_contents(current_path: Path):
            for item in current_path.iterdir():
                if item.is_dir():
                    if list_dir:
                        yield str(item.relative_to(dir_path))
                    if recursive:
                        yield from _list_contents(item)
                elif item.is_file() and list_file:
                    if suffix is None or item.suffix in (suffix if isinstance(suffix, tuple) else (suffix,)):
                        yield str(item.relative_to(dir_path))

        return _list_contents(dir_path)

class Kitti360Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

        self.kitti_infos = []
        self.include_kitti_data()

    def include_kitti_data(self):
        if self.logger is not None:
            self.logger.info('Loading KITTI dataset')
        
        self.kitti_infos = list(list_dir_or_file(
            os.path.join(self.root_path, 'data_3d_raw'),
            list_dir=False, recursive=True, suffix='.bin'
        ))


        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(self.kitti_infos)))

#        data_path_pattern = os.path.join(self.root_path, 'data_3d_raw', '*_sync', '*.bin')
#        self.kitti_infos = glob.glob(data_path_pattern, recursive=True)

    def get_lidar(self, lidar_file):
        return self.client.load_to_numpy(str(self.root_path / 'data_3d_raw' / lidar_file), dtype=np.float32).reshape(-1, 4)

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        return len(self.kitti_infos)

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        lidar_path = self.kitti_infos[index]

        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])
        
        path_split = str(lidar_path).split('/')
        input_dict = {
            'frame_id': path_split[-4] + '_' + path_split[-1][:-4],
        }

        if "points" in get_item_list:
            points = self.get_lidar(lidar_path)
            input_dict['points'] = points

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict


