import copy
import pickle
import os

import numpy as np

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate
from pathlib import Path

from paralleldomain.decoding.helper import decode_dataset
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.utilities.coordinate_system import CoordinateSystem
from paralleldomain.utilities.transformation import Transformation

class PDDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path="/media/Data/parallel_domain/", logger=None):
        """
        Args:
            root_path: path to parallel domain data
            sensor_index: which lidar data to pick #TODO: Pick all LiDARs for a frame and combine
            dataset_cfg: obtained from .YAML in pcdet
            class_names: classes on which detector is trained
            training: if this dataset is for training or not
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.dataset = decode_dataset(dataset_path=root_path, dataset_format="dgp")
        self.class_map = self.dataset.get_scene(self.dataset.scene_names[0]).get_class_map(annotation_type = AnnotationTypes.BoundingBoxes3D)
        self.class_map = {k: f"{self.class_map[k].name}" for k,v in self.class_map.items()}
        print(self.class_map)
        self.sensors = self.dataset.get_scene(self.dataset.scene_names[0]).lidar_names
        self.frame_list = []
        self.create_frame_list(self.dataset)
        self.robot_to_cam = np.array([[0,0,1,0],
                                      [-1,0,0,0],
                                      [0,-1,0,0],
                                      [0,0,0,1]])
        # self.car_classes = [4,5,6,7, 20, 47, 35, 36]
        # self.bike_classes = [13, 1]
        # self.pedesttrians_classes = [22, 14, 2, 18]
        self.valid_classes = ["Bicycle", "Bicyclist", "Bus", "Car", "Caravan/RV", "ConstructionVehicle", "Motorcycle", "Motorcyclist", "OwnCar(EgoCar)", "Pedestrian", "SchoolBus", "Train", "Truck"]
        self.class_map_kitti = {"Bicycle": "Cyclist", "Bicyclist": "Cyclist", "Motorcycle": "Cyclist", "Motorcyclist": "Cyclist", "Bus" : "Car", "Car" : "Car", "Caravan/RV" : "Car", "ConstructionVehicle" : "Car", "OwnCar(EgoCar)": "Car", "SchoolBus": "Car", "Train": "Car", "Truck": "Car", "Pedestrian": "Pedestrian"}
        self.sample_id_list = range(len(self.frame_list))
        self.custom_infos = []
        self.include_data(self.mode)
        self.map_class_to_kitti = self.dataset_cfg.MAP_CLASS_TO_KITTI
    
    def create_frame_list(self, dataset):
        for scene_name in dataset.scene_names:
            scene = dataset.get_scene(scene_name=scene_name)
            for frame_id in scene.frame_ids:
                frame = scene.get_frame(frame_id=frame_id)
                self.frame_list.append(frame)

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, index):
        current_frame = self.frame_list[index]
        final_points = self.get_lidar(index)
        gt_boxes, gt_names = self.get_label(index)

        input_dict = {
            'points': final_points,
            'frame_id': index,
            'gt_names': gt_names,
            'gt_boxes': gt_boxes
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
    
    # def get_lidar(self, current_frame):
    #     lidar_coord_frame = current_frame.get_lidar(lidar_name=self.sensors[1]).pose.inverse.transformation_matrix
    #     lidar_points = []
    #     lidar_points_intense = []
    #     for lidar_name in self.sensors:
    #         cur_lidar_data = current_frame.get_lidar(lidar_name=lidar_name)
    #         cur_pc = cur_lidar_data.point_cloud
    #         cur_lidar_points = cur_pc.xyz_i
    #         lidar_points_intense.append(cur_pc.intensity.reshape(-1,1))
    #         homogenised_pc = cur_pc.xyz_one
    #         lidar_points.append((self.robot_to_cam @ lidar_coord_frame @ cur_lidar_data.pose.transformation_matrix @ (homogenised_pc.T)).T[:,:3])
        
    #     final_points = np.concatenate(lidar_points, axis = 0)
    #     final_points_i = np.concatenate(lidar_points_intense, axis = 0)
    #     final_points_intense = np.zeros((final_points.shape[0], 4))
    #     final_points_intense[:,:3] = final_points
    #     final_points_intense[:,3] = final_points_i.reshape(-1)
        
    #     return final_points_intense
    def get_lidar(self, idx):
        current_frame = self.frame_list[idx]
        lidar_coord_frame = current_frame.get_lidar(lidar_name=self.sensors[1]).pose.inverse.transformation_matrix
        lidar_points = []
        lidar_points_intense = []
        for lidar_name in self.sensors:
            cur_lidar_data = current_frame.get_lidar(lidar_name=lidar_name)
            cur_pc = cur_lidar_data.point_cloud
            cur_lidar_points = cur_pc.xyz_i
            lidar_points_intense.append(cur_pc.intensity.reshape(-1,1))
            homogenised_pc = cur_pc.xyz_one
            lidar_points.append((self.robot_to_cam @ lidar_coord_frame @ cur_lidar_data.pose.transformation_matrix @ (homogenised_pc.T)).T[:,:3])
        
        final_points = np.concatenate(lidar_points, axis = 0)
        final_points_i = np.concatenate(lidar_points_intense, axis = 0)
        final_points_intense = np.zeros((final_points.shape[0], 4))
        final_points_intense[:,:3] = final_points
        final_points_intense[:,3] = final_points_i.reshape(-1)
        
        return final_points_intense
    
    # def get_label(self, current_frame):
    #     lidar_coord_frame = current_frame.get_lidar(lidar_name=self.sensors[1]).pose.inverse.transformation_matrix
    #     instance_dict = {}
    #     annotations = []
    #     annotations_name = []
    #     for lidar_name in self.sensors:
    #         cur_lidar_data = current_frame.get_lidar(lidar_name=lidar_name)
    #         boxes3d = cur_lidar_data.get_annotations(annotation_type = AnnotationTypes.BoundingBoxes3D)
    #         for i, box in enumerate(boxes3d.boxes):
    #             if box.instance_id in instance_dict:
    #                 continue
    #             transformed_box_pose = Transformation.from_transformation_matrix(self.robot_to_cam @ lidar_coord_frame @ cur_lidar_data.pose.transformation_matrix @ box.pose.transformation_matrix)
    #             box_annotation = [transformed_box_pose.translation[0], transformed_box_pose.translation[1], transformed_box_pose.translation[2], box.width, box.length, box.height, transformed_box_pose.as_euler_angles("xyz")[2] - np.pi/2]
    #             if box_annotation[0] > 0:
    #                 annotations.append(box_annotation)
    #                 annotations_name.append(self.class_map[box.class_id])
        
    #     return np.array(annotations, dtype = np.float32), np.array(annotations_name)

    def get_label(self, idx):
        current_frame = self.frame_list[idx]
        lidar_coord_frame = current_frame.get_lidar(lidar_name=self.sensors[1]).pose.inverse.transformation_matrix
        instance_dict = {}
        annotations = []
        annotations_name = []
        for lidar_name in self.sensors:
            cur_lidar_data = current_frame.get_lidar(lidar_name=lidar_name)
            boxes3d = cur_lidar_data.get_annotations(annotation_type = AnnotationTypes.BoundingBoxes3D)
            for i, box in enumerate(boxes3d.boxes):
                if box.instance_id in instance_dict:
                    continue
                transformed_box_pose = Transformation.from_transformation_matrix(self.robot_to_cam @ lidar_coord_frame @ cur_lidar_data.pose.transformation_matrix @ box.pose.transformation_matrix)
                box_annotation = [transformed_box_pose.translation[0], transformed_box_pose.translation[1], transformed_box_pose.translation[2], box.width, box.length, box.height, transformed_box_pose.as_euler_angles("xyz")[2] - np.pi/2]
                if box_annotation[0] > 0:
                    if self.class_map[box.class_id] in self.valid_classes:
                        annotations.append(box_annotation)
                        annotations_name.append(self.class_map_kitti[self.class_map[box.class_id]])
        
        return np.array(annotations, dtype = np.float32), np.array(annotations_name)
    
    def get_infos(self, class_names, num_workers=4, has_label=True, sample_id_list=None, num_features=4):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': num_features, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            if has_label:
                annotations = {}
                gt_boxes_lidar, name = self.get_label(sample_idx)
                annotations['name'] = name
                annotations['gt_boxes_lidar'] = gt_boxes_lidar[:, :7]
                info['annos'] = annotations

            return info

        # sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        
        # create a thread pool to improve the velocity
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, self.sample_id_list)
        return list(infos)
    
    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('custom_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]

        # Output the num of all classes in database
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)
    
    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.custom_infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos, map_name_to_kitti):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.custom_infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos, self.map_class_to_kitti)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict
    
    def include_data(self, mode):
        self.logger.info('Loading Custom dataset.')
        custom_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = Path(self.root_path) / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                custom_infos.extend(infos)

        self.custom_infos.extend(custom_infos)
        self.logger.info('Total samples for CUSTOM dataset: %d' % (len(custom_infos)))
    
    def set_split(self, split): 
        self.split = split

def create_custom_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = PDDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = 'train', 'val'
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    train_filename = Path(save_path) / ('custom_infos_%s.pkl' % train_split)
    val_filename = Path(save_path) / ('custom_infos_%s.pkl' % val_split)

    print('------------------------Start to generate data infos------------------------')

    dataset.set_split(train_split)
    custom_infos_train = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(custom_infos_train, f)
    print('Custom info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    custom_infos_val = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(custom_infos_val, f)
    print('Custom info train file is saved to %s' % val_filename)

    print('------------------------Start create groundtruth database for data augmentation------------------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)
    print('------------------------Data preparation done------------------------')


if __name__ == '__main__':
    import sys

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_custom_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_custom_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path="/media/Data/parallel_domain/",
            save_path="/media/Data/parallel_domain/",
        )
