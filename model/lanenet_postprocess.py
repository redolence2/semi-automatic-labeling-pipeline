#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-30 上午10:04
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_postprocess.py
# @IDE: PyCharm Community Edition
"""
LaneNet model post process
"""
import os.path as ops
import math

import cv2
import numpy as np
# import loguru
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
import sys

# LOG = loguru.logger


def _morphological_process(image, kernel_size=5):
    """
    morphological process to fill the hole in the binary segmentation result
    :param image:
    :param kernel_size:
    :return:
    """
    if len(image.shape) == 3:
        raise ValueError('Binary segmentation result image should be a single channel image')

    if image.dtype is not np.uint8:
        image = np.array(image, np.uint8)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

    # close operation fille hole
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
    dilate_kernel = np.ones((2,2), dtype=np.uint8)
    closing = cv2.erode(closing, dilate_kernel, 1)

    return closing


def _connect_components_analysis(image):
    """
    connect components analysis to remove the small components
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)


class _LaneFeat(object):
    """

    """
    def __init__(self, feat, coord, class_id=-1):
        """
        lane feat object
        :param feat: lane embeddng feats [feature_1, feature_2, ...]
        :param coord: lane coordinates [x, y]
        :param class_id: lane class id
        """
        self._feat = feat
        self._coord = coord
        self._class_id = class_id

    @property
    def feat(self):
        """

        :return:
        """
        return self._feat

    @feat.setter
    def feat(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float64)

        if value.dtype != np.float32:
            value = np.array(value, dtype=np.float64)

        self._feat = value

    @property
    def coord(self):
        """

        :return:
        """
        return self._coord

    @coord.setter
    def coord(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value)

        if value.dtype != np.int32:
            value = np.array(value, dtype=np.int32)

        self._coord = value

    @property
    def class_id(self):
        """

        :return:
        """
        return self._class_id

    @class_id.setter
    def class_id(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.int64):
            raise ValueError('Class id must be integer')

        self._class_id = value


class _LaneNetCluster(object):
    """
     Instance segmentation result cluster
    """

    def __init__(self, cfg):
        """

        """
        self._color_map = matplotlib.cm.get_cmap('tab20c')
        self._cfg = cfg

    def _embedding_feats_dbscan_cluster(self, embedding_image_feats):
        """
        dbscan cluster
        :param embedding_image_feats:
        :return:
        """
        if not isinstance(embedding_image_feats, list):
            embedding_image_feats = [embedding_image_feats]

        origin_features_list = []
        cluster_nums_list = []
        db_labels_list = []
        unique_labels_list = []
        cluster_center_list = []

        for feat in embedding_image_feats:
            print('---------------start DBSCAN----------------')
            db = DBSCAN(eps=self._cfg.dbscan_eps, min_samples=self._cfg.dbscan_min_sample_num)
            try:
                features = StandardScaler().fit_transform(feat)
                db.fit(features)
                print('-------------try--start DBSCAN----------------')
            except Exception as err:
                # LOG.error(err)
                ret = {
                    'origin_features_list': None,
                    'cluster_nums_list': None,
                    'db_labels_list': None,
                    'unique_labels_list': None,
                    'cluster_center_list': None
                }
                return ret
            db_labels = db.labels_
            unique_labels = np.unique(db_labels)

            num_clusters = len(unique_labels)
            cluster_centers = db.components_

            origin_features_list.append(features)
            cluster_nums_list.append(num_clusters)
            db_labels_list.append(db_labels)
            unique_labels_list.append(unique_labels)
            cluster_center_list.append(cluster_centers)
            

        ret = {
                'origin_features_list': origin_features_list,
                'cluster_nums_list': cluster_nums_list,
                'db_labels_list': db_labels_list,
                'unique_labels_list': unique_labels_list,
                'cluster_center_list': cluster_center_list
            }
        return ret

    @staticmethod
    def _get_lane_embedding_feats(binary_seg_ret, instance_seg_ret):
        """
        get lane embedding features according the binary seg result
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        print(f"instance_seg_ret shape: {instance_seg_ret.shape}")
        if isinstance(binary_seg_ret, list):
            print('into each area mode !-----------------------')
            lane_embedding_feat_list = []
            lane_coordinate_list = []

            for each_connected_area in binary_seg_ret:
                lane_embedding_feat_list.append(instance_seg_ret[each_connected_area])
                lane_coordinate_list.append(np.vstack((each_connected_area[1], each_connected_area[0])).transpose())
                assert lane_embedding_feat_list[-1].shape[0] == lane_coordinate_list[-1].shape[0]

            ret = {
                'lane_embedding_feats_list': lane_embedding_feat_list,
                'lane_coordinates_list': lane_coordinate_list
            }

            print('out each area mode !-----------------------')
            return ret

        idx = np.where(binary_seg_ret == 255)
        lane_embedding_feats = instance_seg_ret[idx]
        lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

        assert lane_embedding_feats.shape[0] == lane_coordinate.shape[0]

        ret = {
            'lane_embedding_feats': lane_embedding_feats,
            'lane_coordinates': lane_coordinate
        }

        return ret

    def apply_lane_feats_cluster(self, binary_seg_result, instance_seg_result):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :return:
        """
        # get embedding feats and coords
        get_lane_embedding_feats_result = self._get_lane_embedding_feats(
            binary_seg_ret=binary_seg_result,
            instance_seg_ret=instance_seg_result
        )

        # dbscan cluster
        dbscan_cluster_result = self._embedding_feats_dbscan_cluster(
            # embedding_image_feats=get_lane_embedding_feats_result['lane_embedding_feats']
            embedding_image_feats=get_lane_embedding_feats_result['lane_embedding_feats_list']
        )

        print(f"instance_seg_result.shape : {instance_seg_result.shape}")
        mask = np.zeros(shape=[instance_seg_result.shape[0], instance_seg_result.shape[1], 3], dtype=np.uint8)
        mask_for_curve_fit = np.zeros(shape=[instance_seg_result.shape[0], instance_seg_result.shape[1], 1], dtype=np.uint8)

        db_labels_list = dbscan_cluster_result['db_labels_list']
        unique_labels_list = dbscan_cluster_result['unique_labels_list']
        coord_list = get_lane_embedding_feats_result['lane_coordinates_list']
        num_cluster_list = dbscan_cluster_result['cluster_nums_list']

        if num_cluster_list is None:
            print(f"db scan fail , no answer!")
            return None, None, None
        
        print(f"num_cluster_list {num_cluster_list}")
        if len(num_cluster_list) == 0:
            print(f"num_cluster_list == 0 , no answer!")
            return None, None, None

        color_num = 0
        for num in num_cluster_list:
            color_num += num

        if color_num <= 0:
            return None, None, None

        if len(db_labels_list) <= 0:
            return None, None, None

        lane_coords = []
        color_cnt = 0
        for p in range(len(db_labels_list)):
            for index, label in enumerate(unique_labels_list[p].tolist()):
                # if label == -1:
                #     continue
                idx = np.where(db_labels_list[p] == label)
                pix_coord_idx = tuple((coord_list[p][idx][:, 1], coord_list[p][idx][:, 0]))
                color = self._color_map(color_cnt/color_num)[0:3]
                color = [i * 255 for i in color]

                mask[pix_coord_idx] = color
                mask_for_curve_fit[pix_coord_idx] = color_cnt + 1

                lane_coords.append(coord_list[p][idx])
                color_cnt += 1

        return mask, mask_for_curve_fit, lane_coords


class LaneNetPostProcessor(object):
    """
    lanenet post process for lane generation
    """
    def __init__(self, cfg):
        """

        :param ipm_remap_file_path: ipm generate file path
        """
        # assert ops.exists(ipm_remap_file_path), '{:s} not exist'.format(ipm_remap_file_path)

        self._cfg = cfg
        self._cluster = _LaneNetCluster(cfg=cfg)
        # self._ipm_remap_file_path = ipm_remap_file_path

        # remap_file_load_ret = self._load_remap_matrix()
        # self._remap_to_ipm_x = remap_file_load_ret['remap_to_ipm_x']
        # self._remap_to_ipm_y = remap_file_load_ret['remap_to_ipm_y']

    # def _load_remap_matrix(self):
    #     """

    #     :return:
    #     """
    #     fs = cv2.FileStorage(self._ipm_remap_file_path, cv2.FILE_STORAGE_READ)

    #     remap_to_ipm_x = fs.getNode('remap_ipm_x').mat()
    #     remap_to_ipm_y = fs.getNode('remap_ipm_y').mat()

    #     ret = {
    #         'remap_to_ipm_x': remap_to_ipm_x,
    #         'remap_to_ipm_y': remap_to_ipm_y,
    #     }

    #     fs.release()

    #     return ret

    def postprocess(self, binary_seg_result, instance_seg_result, src_wh):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :param min_area_threshold:
        :param source_image:
        :param with_lane_fit:
        :param data_source:
        :return:
        """
        # convert binary_seg_result
        binary_seg_result = np.array(binary_seg_result * 255, dtype=np.uint8)

        # apply image morphology operation to fill in the hold and reduce the small area
        morphological_ret = _morphological_process(binary_seg_result, kernel_size=5)

        connect_components_analysis_ret = _connect_components_analysis(image=morphological_ret)

        connected_area_list = []
        labels = connect_components_analysis_ret[1]
        stats = connect_components_analysis_ret[2]
        for index, stat in enumerate(stats):
            if index == 0:
                continue

            if stat[4] <= self._cfg.min_sample_num:
                # idx = np.where(labels == index)
                # morphological_ret[idx] = 0
                continue

            cur_label_idx = np.where(labels == index)
            connected_area_list.append(cur_label_idx)

        print(f"connected_area_list len : {len(connected_area_list)}")
        # apply embedding features cluster
        mask_image, mask_for_curve_fit, imagelane_coords = self._cluster.apply_lane_feats_cluster(
            # binary_seg_result=morphological_ret,
            binary_seg_result=connected_area_list,
            instance_seg_result=instance_seg_result
        )

        if mask_image is None:
            return {
                'mask_image': None,
                'fit_params': None,
                'source_image': None,
                'mask_for_curve_fit': None,
            }
    
        tmp_mask = cv2.resize(mask_image, dsize=(src_wh[0], src_wh[1]), interpolation=cv2.INTER_NEAREST)
        mask_for_curve_fit = cv2.resize(mask_for_curve_fit, dsize=(src_wh[0], src_wh[1]), interpolation=cv2.INTER_NEAREST)
        # source_image = cv2.addWeighted(source_image, 0.6, tmp_mask, 0.4, 0.0, dst=source_image)
        return {
            'mask_image': tmp_mask,
            'fit_params': None,
            # 'source_image': source_image,
            'mask_for_curve_fit': mask_for_curve_fit,
        }
