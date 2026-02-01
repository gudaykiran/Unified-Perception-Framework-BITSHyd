"""
Unified Perception Framework - Main Model
Author: Your Name, NIT Warangal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .camera_branch import CameraBranch
from .lidar_branch import LiDARBranch
from .fusion_module import SensorFusionModule
from .panoptic_head import PanopticSegmentationHead

class UnifiedPerceptionModel(nn.Module):
    """
    Complete unified perception model combining:
    Camera + LiDAR + Fusion + Panoptic Segmentation
    """
    
    def __init__(self, num_classes=19, feature_dim=256, fusion_type='cross_attention'):
        super(UnifiedPerceptionModel, self).__init__()
        
        # Initialize components
        self.camera_branch = CameraBranch(num_classes, feature_dim)
        self.lidar_branch = LiDARBranch(feature_dim)
        self.fusion_module = SensorFusionModule(feature_dim, fusion_type)
        self.panoptic_head = PanopticSegmentationHead(feature_dim, num_classes)
        
    def forward(self, rgb_batch, lidar_list, return_features=False):
        """Complete forward pass"""
        
        # Extract features from both modalities
        camera_output = self.camera_branch(rgb_batch, return_features=True)
        camera_features = camera_output['features']
        
        # Align LiDAR features to camera resolution
        target_size = (camera_features.shape[2], camera_features.shape[3])
        lidar_output = self.lidar_branch(lidar_list, target_size=target_size)
        lidar_features = lidar_output['features']
        
        # Fuse multimodal features
        fused_features = self.fusion_module(camera_features, lidar_features)
        
        # Generate panoptic predictions
        output_size = (rgb_batch.shape[2], rgb_batch.shape[3])
        panoptic_output = self.panoptic_head(fused_features, target_size=output_size)
        
        result = {
            'semantic': panoptic_output['semantic'],
            'center': panoptic_output['center'],
            'offset': panoptic_output['offset'],
            'camera_semantic': camera_output['segmentation']
        }
        
        if return_features:
            result.update({
                'camera_features': camera_features,
                'lidar_features': lidar_features,
                'fused_features': fused_features
            })
            
        return result
