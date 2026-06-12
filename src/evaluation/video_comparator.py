"""video_comparator.py: Module to compare two videos and compute similarity metrics.

This module provides functionality to compare two video files using various similarity metrics such as
Structural Similarity Index (SSIM) and Semantic content score.
"""

__author__      = "Ravidu Silva"

import typing
import threading
from contextlib import nullcontext

import cv2
from fastdtw import fastdtw
import numpy as np
import clip
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import euclidean, cdist

import config


class VideoComparator:
    """Class to compare two videos and compute similarity metrics."""

    def __init__(self, embedding_clip_model=config.SupportedModels.VIDEO_COMPARATOR_EMBEDDING_CLIP_MODEL, device="cuda", fps: int = 5):
        """
        Initialize the VideoComparator with a pre-trained model for semantic encoding.

        Args:
            embedding_clip_model (str): The name of the CLIP model to use for semantic encoding.
            device (str): The device to run the model on ('cuda' or 'cpu').
            fps (int): Frames per second to sample from the videos for comparison.
        """
        self.lock = threading.Lock()

        self.fps = fps
        self.device = device

        # Check is the model is available in CLIP
        if embedding_clip_model not in clip.available_models():
            raise ValueError(f"Model '{embedding_clip_model}' is not available in CLIP. "
                             f"Available models: {clip.available_models()}")

        print("Loading CLIP model for video semantic encoding...")
        self.embedding_model, self.preprocess = clip.load(embedding_clip_model, device=self.device)
        self.embedding_model.float()
        self.embedding_model.eval()
        print("CLIP model loaded: ", embedding_clip_model)



    def calculate_video_similarity(self, video_path_ref: str, video_path_test: str) -> dict:
        """Calculate similarity metrics between two videos.

        Args:
            video_path_ref (str): Path to the reference video.
            video_path_test (str): Path to the test video.

        Returns:
            dict: A dictionary containing SSIM and Semantic scores.
        """
        with self.lock:
            # Sample frames from both videos
            frames_ref_gray, frames_ref_rgb = self.sample_frames(video_path_ref)
            frames_test_gray, frames_test_rgb = self.sample_frames(video_path_test)

            # Compute per-frame SSIM matrix
            ssim_matrix = self.per_frame_ssim(frames_ref_gray, frames_test_gray)

            # Compute DTW-based SSIM score
            ssim_score = self.dtw_ssim_score(ssim_matrix)

            # Encode frames to get semantic embeddings
            embeddings_ref = self.encode_frames(frames_ref_rgb)
            embeddings_test = self.encode_frames(frames_test_rgb)

            # Compute semantic similarity score
            visual_semantic_score = self.visual_semantic_similarity(embeddings_ref, embeddings_test)

            return {
                "ssim_score": ssim_score,
                "visual_semantic_score": visual_semantic_score
            }

    
    def sample_frames(self, video_path: str) -> typing.Union[list[np.ndarray], list[np.ndarray]]:
        """Sample frames from the video at the specified FPS.

        Args:
            video_path (str): Path to the video file.

        Returns:
            (list[np.ndarray], list[np.ndarray]): (grayscale, RGB) List of sampled frames as numpy arrays.
        """
        # Check if it's an image file
        if video_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Load single image
            frame = cv2.imread(video_path)
            if frame is None:
                raise ValueError(f"Could not load image file: {video_path}")
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            return [gray_frame], [rgb_frame]
    
        # Process as video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Invalid video FPS for file: {video_path}")
        
        frame_interval = max(1, int(video_fps / self.fps))

        gray_frames = []
        rgb_frames = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                rgb_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_count += 1

        cap.release()
        
        if len(gray_frames) == 0:
            raise ValueError(f"No frames extracted from video: {video_path}")
        
        return gray_frames, rgb_frames
    
    def per_frame_ssim(self, frames_ref: list[np.ndarray], frames_test: list[np.ndarray]) -> list[float]:
        """Compute SSIM for each pair combination of frames from two videos.

        Args:
            frames_ref (list[np.ndarray]): List of frames from the reference video.
            frames_test (list[np.ndarray]): List of frames from the test video.

        Returns:
            list[float]: List of SSIM scores for each frame pair.
        """
        framesA = frames_ref
        framesB = frames_test
        ssim_matrix = np.zeros((len(framesA), len(framesB)), dtype=np.float32)

        for i, frameA in enumerate(framesA):
            for j, frameB in enumerate(framesB):
                # Check if frames are of the same size
                if frameA.shape != frameB.shape:
                    # Resize frameB to match frameA's size
                    frameB = cv2.resize(frameB, (frameA.shape[1], frameA.shape[0]), interpolation=cv2.INTER_AREA)
                score = ssim(frameA, frameB, data_range=255, full=False)
                ssim_matrix[i, j] = score

        return ssim_matrix

    def dtw_ssim_score(self, ssim_matrix: np.ndarray) -> float:
        """Compute the DTW-based SSIM score between two videos.

        Args:
            ssim_matrix (np.ndarray): Matrix of SSIM scores between frames of two videos.

        Returns:
            float: DTW-based SSIM score.
        """
        seqA = ssim_matrix.max(axis=1).reshape(-1, 1)
        seqB = ssim_matrix.max(axis=0).reshape(-1, 1)

        dist, path = fastdtw(seqA, seqB, dist=euclidean)

        # Normalize distance to [0, 1] and convert to similarity score
        norm = max(len(seqA), len(seqB))
        norm_dist = dist / norm

        # Convert normalized distance to similarity score
        score = np.exp(-5 * norm_dist)
        return float(score)
    
    def encode_frames(self, frames: list[np.ndarray]) -> torch.Tensor:
        """Encode frames using the CLIP model to obtain semantic embeddings.

        Args:
            frames (list[np.ndarray]): List of frames as numpy arrays.

        Returns:
            torch.Tensor: Tensor of shape (num_frames, embedding_dim) containing frame embeddings.
        """
        img_feats = []
        for f in frames:
            img = self.preprocess(Image.fromarray(f)).unsqueeze(0).to(self.device).float()
            autocast_context = (
                torch.cuda.amp.autocast(enabled=False)
                if str(self.device).startswith("cuda")
                else nullcontext()
            )
            with torch.no_grad(), autocast_context:
                img_feat = self.embedding_model.encode_image(img)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            img_feats.append(img_feat.cpu().numpy()[0])
        return np.vstack(img_feats)

    
    def visual_semantic_similarity(self, embeddings_ref: torch.Tensor, embeddings_test: torch.Tensor) -> float:
        """Compute visual semantic similarity between two videos using DTW on CLIP embeddings.

        Args:
            embeddings_ref (torch.Tensor): Embeddings of the reference video frames.
            embeddings_test (torch.Tensor): Embeddings of the test video frames.

        Returns:
            float: Semantic similarity score.
        """
        # DTW alignment
        cost, _ = fastdtw(embeddings_ref, embeddings_test, dist=lambda x, y: 1 - np.dot(x, y))
        
        # Normalize cost to [0,1]
        norm_cost = cost / max(len(embeddings_ref), len(embeddings_test))
        score = np.exp(-norm_cost)

        return float(score)
