# Copyright 2023 Enjoy Robotics Zrt - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Modifications to this file is to be shared with the code owner.
# Proprietary and confidential
# Owner: Enjoy Robotics Zrt maintainer@enjoyrobotics.com, 2023

import click
import cv2
from cv2 import aruco
import numpy as np
import yaml
import os


class MarkerFactory:

    @staticmethod
    def create_marker(size: int, marker_id: int, margin: int) -> np.ndarray:
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)

        # white background
        img = 255 * np.ones((size, size), dtype=np.uint8)
        img_marker = aruco.drawMarker(aruco_dict, marker_id, size - 2 * margin)

        # add marker centered
        img[margin:-margin, margin:-margin] = img_marker
        img = cv2.flip(img, 1)

        return img


class TileMap:
    _map: np.ndarray

    def __init__(self, tile_size: int):
        self._map = 255 * np.ones((4, 3, tile_size, tile_size), dtype=np.uint8)

    def set_tile(self, pos: tuple, img: np.ndarray) -> None:
        assert np.all(self._map[pos[0], pos[1]].shape == img.shape)
        self._map[pos[0], pos[1]] = img

    def get_map_image(self) -> np.ndarray:
        """Merge the tile map into a single image."""
        img = np.concatenate(self._map, axis=-1)
        img = np.concatenate(img, axis=-2)

        img = img.T

        return img


@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--tile_size', type=int, default=100)
def main(path: str, tile_size: int) -> None:
    margin = int(0.3 * tile_size)

    marker_factory = MarkerFactory()
    tile_map = TileMap(tile_size)

    order = ['left', 'botton', 'front', 'top', 'back', 'right']

    ids = []

    marker_id = 0
    for i in range(4):
        for j in range(3):
            if i != 1 and (j == 0 or j == 2):
                continue

            marker_img = marker_factory.create_marker(
                tile_size, marker_id, margin)
            tile_map.set_tile((i, j), marker_img)
            ids.append(marker_id)

            marker_id += 1

    tile_img = tile_map.get_map_image()

    tile_img_square = np.zeros((tile_size * 4, tile_size * 4))
    tile_img_square[:, (tile_size // 2):(-tile_size // 2)] = tile_img

    cv2.imwrite(os.path.join(path, 'marker_tile.png'), tile_img)
    cv2.imwrite(os.path.join(path, 'marker_tiles_square.png'), tile_img_square)

    marker_config = dict(zip(order, ids))

    config = {}
    config['aruco_dict'] = '4X4_250'
    config['markers'] = marker_config

    with open(os.path.join(path, 'marker_info.yml'), 'w') as yml_file:
        yaml.dump(config, yml_file)


if __name__ == '__main__':
    main()
