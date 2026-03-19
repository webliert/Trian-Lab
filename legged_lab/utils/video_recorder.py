# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""视频录制配置模块 / Video Recording Configuration Module

该模块定义了人形机器人训练过程中间隔录制视频的配置类和工具函数
This module defines configuration classes and utility functions for interval video recording
during humanoid robot training.
"""

import os
from dataclasses import dataclass
from typing import Optional

from isaaclab.utils import configclass


@configclass
class VideoRecorderCameraCfg:
    """视频录制相机配置 / Video Recorder Camera Configuration"""
    # 相机位置 [x, y, z] 世界坐标
    position: tuple = (2.0, -2.0, 1.8)  # 左前方位置
    # 相机观察目标点 [x, y, z]
    look_at: tuple = (0.0, 0.0, 0.5)  # 对准机器人中心
    # 相机名称
    name: str = "main_camera"
    # Prim路径
    prim_path: str = "/World/recording_camera"


@configclass
class VideoRecorderCfg:
    """视频录制器配置 / Video Recorder Configuration
    
    定义训练过程中视频录制的参数
    Defines parameters for video recording during training.
    """
    # 是否启用视频录制
    enable: bool = False
    
    # 录制间隔（步数）
    interval: int = 500
    
    # 每次录制的帧数
    num_frames: int = 500
    
    # 输出目录
    output_dir: str = "./humanoid_videos"
    
    # 视频文件名前缀
    filename_prefix: str = "training_video"
    
    # 视频参数
    fps: int = 30
    width: int = 1280
    height: int = 720
    codec: str = "libx264"  # H.264 codec
    
    # 相机配置
    camera: VideoRecorderCameraCfg = VideoRecorderCameraCfg()


class VideoRecorderManager:
    """视频录制管理器 / Video Recorder Manager
    
    用于在训练过程中间隔录制视频的管理器
    Manager for recording videos at intervals during training.
    """
    
    def __init__(
        self,
        cfg: VideoRecorderCfg,
        sim,
        log_dir: str,
        device: str = "cuda:0"
    ):
        """初始化视频录制管理器 / Initialize Video Recorder Manager
        
        Args:
            cfg: 视频录制配置
            sim: Isaac Lab仿真上下文
            log_dir: 日志目录
            device: 设备
        """
        self.cfg = cfg
        self.sim = sim
        self.log_dir = log_dir
        self.device = device
        
        # 创建输出目录
        self.output_dir = os.path.join(log_dir, cfg.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 帧缓冲区
        self.frame_buffer = []
        self.is_recording = False
        self.recording_start_step = 0
        
        # 相机和渲染相关
        self.camera = None
        self.render_product = None
        
        # 初始化相机
        self._setup_camera()
    
    def _setup_camera(self):
        """设置录制相机 / Setup Recording Camera"""
        try:
            import omni.usd
            from pxr import UsdGeom, Gf, Sdf
            
            stage = omni.usd.get_context().get_stage()
            
            # 创建相机Prim
            camera_path = self.cfg.camera.prim_path
            if not stage.GetPrimAtPath(camera_path):
                UsdGeom.Camera.Define(stage, camera_path)
            
            # 设置相机参数
            camera_prim = stage.GetPrimAtPath(camera_path)
            camera_api = UsdGeom.Camera(camera_prim)
            
            # 设置相机位置和朝向
            camera_api.CreateTranslateAttr(Gf.Vec3d(*self.cfg.camera.position))
            
            # 设置焦距和其他参数
            camera_api.CreateFocalLengthAttr(50.0)  # 50mm 焦距
            camera_api.CreateHorizontalApertureAttr(36.0)  # 感光元件宽度 mm
            camera_api.CreateVerticalApertureAttr(20.25)  # 感光元件高度 mm
            
            print(f"[INFO] Video recording camera created at: {camera_path}")
            print(f"[INFO] Camera position: {self.cfg.camera.position}")
            print(f"[INFO] Camera look_at: {self.cfg.camera.look_at}")
            
        except Exception as e:
            print(f"[WARN] Failed to setup recording camera: {e}")
    
    def should_start_recording(self, current_step: int) -> bool:
        """检查是否应该开始录制 / Check if should start recording
        
        Args:
            current_step: 当前步数
            
        Returns:
            是否应该开始录制
        """
        if not self.cfg.enable:
            return False
        
        # 每隔interval步录制一次
        return current_step > 0 and current_step % self.cfg.interval == 0
    
    def should_stop_recording(self, current_step: int) -> bool:
        """检查是否应该停止录制 / Check if should stop recording
        
        Args:
            current_step: 当前步数
            
        Returns:
            是否应该停止录制
        """
        if not self.is_recording:
            return False
        
        frames_recorded = current_step - self.recording_start_step
        return frames_recorded >= self.cfg.num_frames
    
    def start_recording(self, current_step: int):
        """开始录制 / Start Recording
        
        Args:
            current_step: 当前步数
        """
        self.is_recording = True
        self.recording_start_step = current_step
        self.frame_buffer = []
        print(f"[INFO] Started video recording at step {current_step}")
    
    def stop_recording(self, current_step: int):
        """停止录制并保存视频 / Stop Recording and Save Video
        
        Args:
            current_step: 当前步数
        """
        if not self.frame_buffer:
            print(f"[WARN] No frames captured for video")
            self.is_recording = False
            return
        
        # 生成文件名
        filename = f"{self.cfg.filename_prefix}_step{self.recording_start_step}.mp4"
        filepath = os.path.join(self.output_dir, filename)
        
        # 保存视频
        self._save_video(filepath)
        
        print(f"[INFO] Saved video to: {filepath}")
        print(f"[INFO] Total frames: {len(self.frame_buffer)}")
        
        self.is_recording = False
        self.frame_buffer = []
    
    def capture_frame(self):
        """捕获当前帧 / Capture Current Frame
        
        在每个仿真步中调用此方法捕获帧
        Call this method to capture frame at each simulation step
        """
        if not self.is_recording:
            return
        
        try:
            # 渲染场景
            self.sim.render()
            
            # 获取渲染图像
            # 注意：这里需要使用Isaac Sim的渲染API
            frame = self._grab_frame()
            
            if frame is not None:
                self.frame_buffer.append(frame)
                
        except Exception as e:
            print(f"[WARN] Failed to capture frame: {e}")
    
    def _grab_frame(self):
        """获取渲染帧 / Grab Rendered Frame
        
        从渲染器获取当前帧
        Get current frame from renderer
        
        Note: 此方法需要在Isaac Lab环境中实际实现
        目前返回None，视频录制功能需要与环境类的render调用配合
        """
        # 实际实现需要使用Isaac Sim的渲染API
        # 这里返回None，依赖外部环境的render调用
        return None
    
    def _save_video(self, filepath: str):
        """保存视频 / Save Video
        
        将帧缓冲区保存为视频文件
        Save frame buffer as video file
        
        Args:
            filepath: 输出文件路径
        """
        if not self.frame_buffer:
            return
        
        try:
            import cv2
            
            # 获取帧的形状
            frame = self.frame_buffer[0]
            if frame is None:
                print(f"[WARN] No valid frames to save")
                return
                
            height, width = frame.shape[:2]
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
            out = cv2.VideoWriter(
                filepath,
                fourcc,
                self.cfg.fps,
                (width, height)
            )
            
            # 写入帧
            for frame in self.frame_buffer:
                if frame is not None:
                    # BGR to RGB if needed
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame
                    out.write(frame_rgb)
            
            out.release()
            print(f"[INFO] Video saved successfully: {filepath}")
            
        except ImportError:
            # 如果没有cv2，保存为图片序列
            self._save_as_images(filepath.replace('.mp4', ''))
        except Exception as e:
            print(f"[ERROR] Failed to save video: {e}")
    
    def _save_as_images(self, output_dir: str):
        """保存为图片序列 / Save as Image Sequence
        
        当cv2不可用时，保存为图片序列
        Save as image sequence when cv2 is not available
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for i, frame in enumerate(self.frame_buffer):
            if frame is not None:
                filepath = os.path.join(output_dir, f"frame_{i:04d}.png")
                try:
                    import cv2
                    cv2.imwrite(filepath, frame)
                except:
                    pass
        
        print(f"[INFO] Saved {len(self.frame_buffer)} frames as images to: {output_dir}")
    
    def shutdown(self):
        """关闭视频录制器 / Shutdown Video Recorder"""
        if self.is_recording and self.frame_buffer:
            # 如果还在录制中，保存当前视频
            self.stop_recording(0)
        
        print("[INFO] Video recorder shutdown")


def create_video_recorder_manager(
    cfg: VideoRecorderCfg,
    sim,
    log_dir: str,
    device: str = "cuda:0"
) -> VideoRecorderManager:
    """创建视频录制管理器 / Create Video Recorder Manager
    
    Args:
        cfg: 视频录制配置
        sim: Isaac Lab仿真上下文
        log_dir: 日志目录
        device: 设备
        
    Returns:
        VideoRecorderManager实例
    """
    return VideoRecorderManager(cfg, sim, log_dir, device)
