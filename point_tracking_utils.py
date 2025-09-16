import imageio
from collections import defaultdict
import os
import cv2

# cotracker visualization
from cotracker.utils.visualizer import Visualizer as CoTrackerVisualizer

# interpolate cotracker points
from scipy.interpolate import RegularGridInterpolator

# detectron for gripper detection
from detectron2.config import get_cfg
from Gripper_detector.inference import BatchPredictor as GripperPredictor
from detectron2.utils.visualizer import Visualizer as DetectronVisualizer

import torch
import numpy as np
import time
# from google import genai

# cluster into points based on the cotracker output
from sklearn.cluster import KMeans

from base64 import b64encode
import uuid


def video_to_frames(video_frames, traj_splits):
    """
    Splits a list of video frames into segments based on frame indices in traj_splits.

    Args:
        video_frames (list): List of video frames.
        traj_splits (list): List of frame indices to split the video.

    Returns:
        list: A list of lists, where each inner list contains frames for a segment.
    """
    segment_frames = []
    frame_idx = traj_splits[0]
    for i in range(1, len(traj_splits)):
        segment_frames.append(video_frames[frame_idx : traj_splits[i] + 1])
        frame_idx = traj_splits[i] + 1
    return segment_frames



def setup_predictors(device, debug: bool):
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(
        device
    )
    cfg = get_cfg()
    cfg.merge_from_file("Gripper_detector/config/det_config.yaml")
    cfg.MODEL.WEIGHTS = "detectron_gripper.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    gripper_predictor = GripperPredictor(cfg).to(device)
    gripper_predictor.input_format = "RGB"
    return cotracker, gripper_predictor

def draw_gripper_onto_video(video_frames, gripper_positions):
    # Draw mean positions onto each frame
    for i, frame in enumerate(video_frames):
        if gripper_positions[i] is not None:
            mean_x, mean_y = gripper_positions[i]
            cv2.circle(
                frame,
                (int(mean_x), int(mean_y)),
                radius=5,
                color=(0, 255, 0),
                thickness=-1,
            )


def save_video(masked_video: np.ndarray, output_path: str, fps=30):
    """
    Saves the masked video to disk.

    Args:
        masked_video: Numpy array of shape [T, H, W, C], where T is the number of frames,
                      C is the number of channels, H and W are the height and width.
        output_path: Path to save the video (e.g., "masked_video.mp4").
        fps: Frames per second for the output video (default: 30).

    Returns:
        None. Saves the video to disk.
    """
    # T, C, H, W = masked_video.shape
    T, H, W, C = masked_video.shape

    video_writer = imageio.get_writer(output_path, fps=fps)

    # Write frames to the video file
    for frame in masked_video.astype(np.uint8):
        video_writer.append_data(frame)

    video_writer.close()
    print(f"Video saved to {output_path}")


def upscale_image(image, processor_sr, model_sr, device):
    """Upscale image using Swin2SR"""
    inputs = processor_sr(image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model_sr(**inputs)
    output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.moveaxis(output, source=0, destination=-1)
    output = (output * 255.0).round().astype(np.uint8)
    return output


# helper functions to track movement, check for when points stock, etc.
def track_significant_movement(
    pred_tracks,
    image_width,
    image_height,
    movement_threshold=0.05,
):
    """
    Identifies points with significant relative movement across frames
    Args:
        pred_tracks: [T, n_points, 2] array of absolute coordinates
        image_width: original video width in pixels
        image_height: original video height in pixels
        movement_threshold: relative movement threshold (0-1 scale, default 0.05 = 5% of frame size)
    Returns:
        List of significant point indices
    """
    T, n_points, _ = pred_tracks.shape
    significant_points = []

    # Convert absolute coordinates to normalized [0,1] space
    normalized_tracks = pred_tracks.copy()
    normalized_tracks[..., 0] /= image_width  # X normalization
    normalized_tracks[..., 1] /= image_height  # Y normalization

    for point_idx in range(n_points):
        # Calculate path length in normalized space
        for t in range(0, T):
            if np.any(
                np.linalg.norm(
                    normalized_tracks[t + 1 :, point_idx]
                    - normalized_tracks[t - 1, point_idx],
                    axis=1,
                )
                > movement_threshold
            ):
                significant_points.append(point_idx)
                break


    return significant_points


def determine_stopping_points(
    pred_tracks,
    significant_points,
    image_width,
    image_height,
    stabil_thresh=0.001,
    future_n_frames=5,
):
    """
    Determines the stopping points for significant points across the entire video.

    Args:
        pred_tracks: [T, n_points, 2] array of absolute coordinates
        significant_points: List of indices of significant points
        image_width: original video width in pixels
        image_height: original video height in pixels
        stabil_thresh: threshold for considering a point as stopped (default: 0.05)
        future_n_frames: number of future frames to check for stability (default: 5)

    Returns:
        Dictionary with point indices as keys and list of stopping frame indices as values
    """
    stopping_points = {pid: [] for pid in significant_points}

    normalized_tracks = pred_tracks.copy()
    normalized_tracks[:, :, 0] /= image_width
    normalized_tracks[:, :, 1] /= image_height

    for pid in significant_points:
        for t in range(len(pred_tracks) - future_n_frames):
            future_displacements = np.linalg.norm(
                normalized_tracks[t, pid] - normalized_tracks[t + future_n_frames, pid]
            )
            if future_displacements < stabil_thresh:
                stopping_points[pid].append(t)

    # last point is always a stopping point
    for pid in significant_points:
        stopping_points[pid].append(len(pred_tracks) - 1)

    return stopping_points


def mask_video_with_crops(video_frames, significant_points, stopping_points, crop_ratio=0.1):
    """
    Masks out everything in the original video except for square areas around significant points.

    Args:
        video_frames: Tensor of shape [T, C, H, W], where T is the number of frames,
                      C is the number of channels, H and W are the height and width.
        significant_points: Tensor of shape [T, N, 2], where N is the number of significant points,
                            containing x-y coordinates for each point.
        stopping_points: Tensor of shape [T, N, 2], where N is the number of significant points,
                            containing the stopping points for each point.
        crop_ratio: Float between 0 and 1, representing the fraction of the smaller image dimension
                    to use for the crop size.

    Returns:
        A tensor of masked video frames with shape [T, C, H, W].
        A tensor of masks with shape [T, H, W].
    """
    T, C, H, W = video_frames.shape

    # Calculate crop size based on the smaller dimension of the image
    crop_size = int(min(H, W) * crop_ratio)
    half_crop = crop_size // 2

    # Initialize a tensor to store masked video frames
    masked_video = torch.zeros_like(video_frames)

    # Initialize all masks
    masks = torch.zeros((T, H, W), dtype=torch.float32)

    def mask_point(points, t, i):
        # Get coordinates for this point, and make sure it's within the frame
        x_center = int(points[t, i, 0])
        y_center = int(points[t, i, 1])

        x_center, y_center = max(0, x_center), max(0, y_center)
        x_center, y_center = min(W - 1, x_center), min(H - 1, y_center)

        # Define cropping boundaries
        x_min = max(0, x_center - half_crop)
        x_max = min(W, x_center + half_crop)
        y_min = max(0, y_center - half_crop)
        y_max = min(H, y_center + half_crop)

        # Set mask values to 1 in the crop area
        mask[y_min:y_max, x_min:x_max] = 1.0


    for t in range(T):
        # Create a mask for this frame
        mask = torch.zeros((H, W), dtype=torch.float32)

        for i in range(significant_points.shape[1]):
            mask_point(significant_points, t, i)
        for i in range(stopping_points.shape[1]):
            mask_point(stopping_points, t, i)

        # Apply mask to all channels of this frame
        masked_video[t] = video_frames[t] * mask.unsqueeze(
            0
        )  # Broadcast mask to all channels
        masks[t] = mask

    return masked_video, masks


def get_points_in_gripper_box(
    frames_detections, pred_tracks, inclusion_threshold=0.5
) -> np.ndarray:
    """
    Identifies points that appear in the detection box with a given inclusion_threshold.

    Args:
        frames_detections: List of dictionaries containing detectron2 instances for each frame.
        pred_tracks: [T, N, 2] array of absolute coordinates.
        inclusion_threshold: top percentile of points to include out of points which are in the detection box at any point.

    Returns:
        List of point indices that appear in the detection box at least half of the time.
    """
    T, N, _ = pred_tracks.shape
    points_in_box = defaultdict(int)  # starts at 0

    for i, detections in enumerate(frames_detections):
        instances = detections["instances"].to("cpu")
        scores = instances.scores.numpy()
        sorted_indices = np.argsort(scores)[::-1]
        boxes = instances.pred_boxes.tensor.numpy()
        if len(boxes) > 0:
            boxes = boxes[sorted_indices]
            mle_box = boxes[0]

            box_xmin, box_ymin, box_xmax, box_ymax = mle_box

            frame_points = pred_tracks[i]  # shape (N, 2)
            valid = (
                (frame_points[:, 0] > box_xmin)
                & (frame_points[:, 0] < box_xmax)
                & (frame_points[:, 1] > box_ymin)
                & (frame_points[:, 1] < box_ymax)
            )
            valid_ids = torch.nonzero(valid).flatten()
            for pid in valid_ids:
                points_in_box[pid.item()] += 1

    # Find points that appear in the detection box at the specified inclusion_threshold using percentile
    # get the top percentile of points that appear in the box
    points_in_box_values = np.array(sorted([count for count in points_in_box.values()]))
    if len(points_in_box_values) == 0:
        return []
    inclusion_minimum_count = np.percentile(
        points_in_box_values, q=inclusion_threshold * 100
    )
    points_in_gripper_box = np.array(
        [
            pid
            for pid, count in points_in_box.items()
            if count >= inclusion_minimum_count
        ]
    )
    return points_in_gripper_box


def predict_gripper_with_cotracker(
    frames: np.ndarray,
    gripper_predictor: GripperPredictor,
    significant_points: torch.tensor,
    batch_size: int,
):
    with torch.inference_mode():
        batched_frames = [
            frames[i : i + batch_size] for i in range(0, len(frames), batch_size)
        ]
        frames_detections = []
        for batch in batched_frames:
            frames_detections.extend(gripper_predictor(batch))

    points_in_gripper_box_idxs = get_points_in_gripper_box(
        frames_detections, significant_points
    )

    if len(points_in_gripper_box_idxs) == 0:
        return None, None

    points_in_gripper_box = significant_points[:, points_in_gripper_box_idxs]  # T N 2

    gripper_box_mean_x_y = (
        torch.mean(points_in_gripper_box, dim=1).squeeze(0).cpu().numpy()
    )
    return gripper_box_mean_x_y, frames_detections


def predict_trajectory_splits(
    video: np.ndarray,
    stopping_points_dict: dict,
    min_num_frames_in_split=5,
    min_split_distance=30,
) -> np.ndarray:
    # get subtrajectory split points based on the stopping_points_dict
    stopping_times = []
    for i, pid in enumerate(stopping_points_dict):
        stopping_times.extend(stopping_points_dict[pid])

    # count the number of stopping points at each timestep
    stopping_times = np.array(stopping_times)
    unique, counts = np.unique(stopping_times, return_counts=True)
    unique_to_counts = dict(zip(unique.tolist(), counts.tolist()))
    idx_to_moved_points = [
        unique_to_counts[i] if i in unique_to_counts else 0 for i in range(len(video))
    ]

    kmeans = KMeans(n_clusters=2, random_state=0).fit(
        np.array(idx_to_moved_points).reshape(-1, 1)
    )
    label_of_stopped_points = np.argmax(kmeans.cluster_centers_)

    # figure out cluster assignemtns
    cluster_assignments = kmeans.predict(np.array(idx_to_moved_points).reshape(-1, 1))

    # check the frames where the cluster assignments are different and what the number of moved points are
    prev_start = 0
    currently_in_stopped_point_cluster = False
    traj_splits = []
    for i, assignment in enumerate(cluster_assignments):
        if (
            assignment != label_of_stopped_points and currently_in_stopped_point_cluster
        ) or i == len(cluster_assignments) - 1:
            # we have changed to non-stopped points, mark the end of the segment
            if i - prev_start > min_num_frames_in_split:
                traj_splits.append((prev_start, i))
        elif (
            assignment == label_of_stopped_points
            and not currently_in_stopped_point_cluster
        ):
            # we have changed to stopped points, mark the start of the segment
            prev_start = i
        currently_in_stopped_point_cluster = assignment == label_of_stopped_points

    # get mean of traj splits
    traj_splits = [round((x1 + x2) / 2) for x1, x2 in traj_splits]

    # ensure last frame is included
    if len(traj_splits) == 0 or traj_splits[-1] != len(video) - 1:
        traj_splits.append(len(video) - 1)

    # add first frame
    traj_splits = sorted(set([0] + traj_splits))

    # merge splits that are too close together
    traj_splits = [traj_splits[0]] + [
        traj_splits[i]
        for i in range(1, len(traj_splits))
        if traj_splits[i] - traj_splits[i - 1] > min_split_distance
    ]
    # if there are no splits, then just use the first and last frame
    if len(traj_splits) <= 1:
        traj_splits = [0, len(video) - 1]

    # if last frame is still not included then make the last traj_split the last frame
    if traj_splits[-1] != len(video) - 1:
        traj_splits[-1] = len(video) - 1


    return np.array(traj_splits)


def track_objects_in_video(
    cotracker: torch.nn.Module,
    gripper_detector: GripperPredictor,
    video: np.ndarray,
    grid_size: int,
    movement_threshold: float,
    stopping_threshold: float,
    save_dir: str,
    device: torch.device,
    detectron_batch_size: int,
    visualize=False,
    visualize_detailed_outputs=False,
    crop_ratio=0.1,
    vid_max_length=100,
    future_n_frames=5,
    tracking_start_point="middle",
):
    """Given a video path and images of the video, track objects in the video.

    Args:
        - cotracker: The torch pre-trained cotracker model.
        - gripper_detector: The torch pre-trained gripper detector model.
        - video: video frame sequence as an np array.
        - grid_size: The grid size for the cotracker.
        - movement_threshold: The threshold for significant movement.
        - stopping_threshold: The threshold for stopping movement.
        - save_dir: Directory to save the tracked video.
        - device: torch device
        - detectron_batch_size: Batch size for the gripper detector.
        - visualize: Whether to visualize the object tracking with matplotlib.
        - visualize_detailed_outputs: Whether to visualize detailed outputs of the point and gripper tracking.
        - crop_ratio: Mask cropping ratio
        - vid_max_length: Maximum length of the video. Downsamples to this if longer.
        - future_n_frames: The number of frames to look ahead for stopping points.
        - tracking_start_point: The point to start tracking the objects. Can be "start", "middle", or "end".
    """
    assert video.shape == (
        len(video),
        video[0].shape[0],
        video[0].shape[1],
        3,
    ), "Video must be a 4D array with shape (T, H, W, 3)."

    # Store original video for later use
    original_video = video.copy()

    print(f"Original video shape: {video.shape}")

    downsampled = False
    if len(video) > vid_max_length:
        print(f"Downsampling video from {len(video)} to {vid_max_length} frames.")
        downsampled = True
        # linspace to get the indices of the frames to sample
        indices = np.linspace(0, len(video) - 1, vid_max_length, dtype=int)

        # make sure there are no duplicates
        downsampled_indices = np.array(sorted(list(set(indices))))
        full_length_video = video
        video = video[downsampled_indices]

    print(f"Downsampled video shape: {video.shape}")
    with torch.inference_mode():
        video = (
            torch.tensor(video).permute(0, 3, 1, 2)[None].float().to(device)
        )  # B T 3 H W
        query_idx = video.shape[1] // 2
        if tracking_start_point == "start":
            query_idx = 0
        elif tracking_start_point == "end":
            query_idx = video.shape[1] - 1
        pred_tracks, pred_visibility = cotracker(
            video,
            grid_size=grid_size,
            grid_query_frame=query_idx,
            backward_tracking=True,
        )  # B T N 2,  0 B T N 1
    video = video.cpu()[0]
    print(f"Post cotracker video shape: {video.shape}")
    pred_tracks = pred_tracks[0].cpu()  # T x N x 2
    pred_visibility = pred_visibility[0].cpu()

    # now linearly interpolate the cotracker points to fill in the from
    if downsampled:
        T, N = pred_tracks.shape[:2]
        lin_interpolater = RegularGridInterpolator(
            (downsampled_indices,),
            pred_tracks.numpy(),
            bounds_error=True,  # raise an error if out of bounds; shouldn't happen at all
            method="linear",
        )
        pred_tracks = torch.from_numpy(
            lin_interpolater(np.arange(len(full_length_video))).reshape(
                len(full_length_video), N, 2
            )
        )
        # also do the same to visiblitity using nearest interpolation as it's a binary value
        nearest_interpolator = RegularGridInterpolator(
            (downsampled_indices,),
            pred_visibility.numpy(),
            bounds_error=True,  # raise an error if out of bounds; shouldn't happen at all
            method="nearest",
        )
        pred_visibility = torch.from_numpy(
            nearest_interpolator(np.arange(len(full_length_video))).reshape(
                len(full_length_video), N, 1
            )
        )
        video = torch.from_numpy(full_length_video).permute(0, 3, 1, 2).float()
    print(f"Post linearly interpolated video shape: {video.shape}")

    significant_point_idxs = track_significant_movement(
        pred_tracks.numpy(),
        video.shape[-1],
        video.shape[-2],
        movement_threshold=movement_threshold,
    )
    print(f"Found {len(significant_point_idxs)} significant points.")
    stopping_points_dict = determine_stopping_points(
        pred_tracks.numpy(),
        significant_point_idxs,
        video.shape[-1],
        video.shape[-2],
        stabil_thresh=stopping_threshold,
        future_n_frames=future_n_frames,
    )
    significant_points = pred_tracks[:, significant_point_idxs]
    if significant_points.shape[1] == 0:
        print("No significant points found. Skipping trajectory analysis.")
        return None
        
    # significant_points_visibility = pred_visibility[:, significant_point_idxs]

    stopping_points_tensor = torch.zeros_like(significant_points)
    # make a tensor of the points at the next stopping point location
    for i, pid in enumerate(significant_point_idxs):
        prev_timestep = 0
        for j, timestep in enumerate(stopping_points_dict[pid]):
            stopping_points_tensor[prev_timestep : timestep + 1, i] = (
                significant_points[timestep, i]
            )
            prev_timestep = timestep

    # combine significant points and stopping points
    #combined_points = torch.cat([significant_points, stopping_points_tensor], dim=1)

    # get the trajectory split points
    traj_splits = predict_trajectory_splits(video, stopping_points_dict)
    
    # get the remaining movement for all of the significant points
    point_movements_across_whole_video = np.linalg.norm(
        np.diff(significant_points.numpy(), axis=0), axis=2
    )
    # Calculate remaining movements per trajectory split
    remaining_point_movements_until_next_subtraj = np.zeros_like(
        point_movements_across_whole_video
    )
    # Iterate through trajectory splits
    assert len(traj_splits) >= 2, f"There must be at least two trajectory start/end points. If not, check the trajectory split logic. Traj splits: {traj_splits}. Video length: {len(video)}"
    # Iterate through trajectory splits to track significant points from next split
    significant_point_idxs_per_split = []
    for i in range(0, len(traj_splits) - 1):
        start_idx = traj_splits[i]
        if i +1 == len(traj_splits) - 1:
            end_idx = len(video) # ensures that the last index is included since otherwise they overlap
        else:
            end_idx = traj_splits[i+1]
        # Get significant points for this split
        curr_significant_points = track_significant_movement(
            significant_points[start_idx:end_idx].numpy(),
            video.shape[-1],
            video.shape[-2], 
            movement_threshold=movement_threshold,
        )
        significant_point_idxs_per_split.append(curr_significant_points)

    masked_traj_split_videos = []
    traj_split_masks = []
    for i in range(len(traj_splits) - 2):
        start_idx = traj_splits[i]
        end_idx = traj_splits[i+1]
        
        # Get movements for this split
        split_movements = point_movements_across_whole_video[start_idx:end_idx]

        # Calculate cumsum for just this split
        split_remaining_movements = np.cumsum(split_movements, axis=0)

        # Store in the full array
        remaining_point_movements_until_next_subtraj[start_idx:end_idx] = (
            split_remaining_movements
        )

        # significant points in this split and next split will be combined to form the significant points for this split for masking
        # to include receptacles
        curr_significant_points_idxs = significant_point_idxs_per_split[i] 
        #next_significant_points_idxs = significant_point_idxs_per_split[i+1] 
        curr_significant_points = significant_points[start_idx:end_idx, curr_significant_points_idxs]

        curr_stopping_points = stopping_points_tensor[start_idx:end_idx, curr_significant_points_idxs]

        # mask the video with the crops
        curr_masked_video, curr_mask = mask_video_with_crops(
            video[start_idx:end_idx],
            curr_significant_points,
            curr_stopping_points,
            crop_ratio=crop_ratio,
        )
        masked_traj_split_videos.append(curr_masked_video)
        traj_split_masks.append(curr_mask)

    # do the last split
    start_idx = traj_splits[-2]
    end_idx = len(video)
    split_movements = point_movements_across_whole_video[start_idx:end_idx]
    split_remaining_movements = np.cumsum(split_movements, axis=0)
    remaining_point_movements_until_next_subtraj[start_idx:end_idx] = split_remaining_movements

    # get the significant points for the last split
    curr_significant_points_idxs = significant_point_idxs_per_split[-1]
    curr_stopping_points = stopping_points_tensor[start_idx:end_idx, curr_significant_points_idxs]

    # mask the video with the crops
    curr_masked_video, curr_mask = mask_video_with_crops(
        video[start_idx:end_idx],
        significant_points[start_idx:end_idx, curr_significant_points_idxs],
        curr_stopping_points,
        crop_ratio=crop_ratio,
    )
    masked_traj_split_videos.append(curr_masked_video)
    traj_split_masks.append(curr_mask)

    remaining_point_movements_until_next_subtraj = np.concatenate(
        [
            remaining_point_movements_until_next_subtraj,
            np.zeros((1, remaining_point_movements_until_next_subtraj.shape[1])),
        ]
    )
    masked_video = torch.cat(masked_traj_split_videos, dim=0)
    masks = torch.cat(traj_split_masks, dim=0)

    # convert the masked_video into a numpy video
    masked_video = masked_video.permute(0, 2, 3, 1).numpy()
    print(f"Post masked video shape: {masked_video.shape}")
    # get gripper positions
    gripper_positions, gripper_detections = predict_gripper_with_cotracker(
        masked_video,
        gripper_detector,
        significant_points,
        batch_size=detectron_batch_size,
    )

    if gripper_positions is not None:
        # draw the gripper onto the video
        draw_gripper_onto_video(masked_video, gripper_positions)
    else:
        print("No gripper detected in the video.")

    if visualize:
        # save the tracked video
        os.makedirs(f"{save_dir}/", exist_ok=True)
        save_video(masked_video, f"{save_dir}/masked_video.mp4", fps=20)
    if visualize_detailed_outputs:
        # save a debug video with the visualized everything
        os.makedirs(f"{save_dir}/debug/", exist_ok=True)
        cotracker_vis = CoTrackerVisualizer(
            save_dir=f"{save_dir}/debug/", pad_value=20, linewidth=0.5
        )
        cotracker_vis.visualize(
            video[None],
            pred_tracks[None],
            pred_visibility[None],
            filename="cotracker_points",
        )

        # detectron video
        detectron_frames = []
        if gripper_detections is not None:
            for i, frame in enumerate(masked_video):
                detectron_vis = DetectronVisualizer(img_rgb=frame)
                det_results = detectron_vis.draw_instance_predictions(
                    gripper_detections[i]["instances"].to("cpu")
                )
                detectron_frames.append(det_results.get_image())

            detectron_frames = np.asarray(detectron_frames)

            save_video(
                detectron_frames, f"{save_dir}/debug/detectron_video.mp4", fps=20
            )

        trajectory_split_frames = video.permute(0, 2, 3, 1).numpy()[traj_splits]
        print(f"Trajectory split frames shape: {trajectory_split_frames.shape}")
        save_video(
            trajectory_split_frames,
            f"{save_dir}/debug/trajectory_split_frames.mp4",
            fps=1,
        )

    if gripper_positions is not None:
        # gripper positions, significant points, stopping points all need to be set to a min of 0 and a max of the image shapes
        gripper_positions = np.clip(gripper_positions, 0, masked_video.shape[1:3]).round().astype(np.uint16)
        significant_points = np.clip(significant_points.numpy(), 0, masked_video.shape[1:3]).round().astype(np.uint16)
        stopping_points_tensor = np.clip(stopping_points_tensor.numpy(), 0, masked_video.shape[1:3]).round().astype(np.uint16)
        return (
            gripper_positions,  #  (T, 2)
            significant_points,  #  (T, N, 2)
            stopping_points_tensor,  # (T, N, 2)
            remaining_point_movements_until_next_subtraj,  # (T, N)
            masks.numpy().astype(np.uint8),  # (T, *img_size)
            traj_splits.astype(np.uint16),  # (N,)
        )
    else:
        return None

def add_to_h5_if_not_exists(
    h5_file, key, data, overwrite_existing_data=False, **add_kwargs
):
    if key not in h5_file or (key in h5_file and overwrite_existing_data):
        if key in h5_file:
            del h5_file[key]
        h5_file.create_dataset(key, data=data, **add_kwargs)
    else:
        print(f"Key {key} already exists in the h5 file. Skipping.")


def add_all_keys_to_h5(
    save_h5,
    gripper_positions,
    significant_points,
    stopped_points,
    movement_across_subtrajectory,
    masked_frames,
    traj_splits,
    overwrite_existing_data=False,
):
    add_to_h5_if_not_exists(
        save_h5,
        "gripper_positions",
        gripper_positions,
        dtype=np.uint16,
        overwrite_existing_data=overwrite_existing_data,
    )
    add_to_h5_if_not_exists(
        save_h5,
        "significant_points",
        significant_points,
        dtype=np.uint16,
        overwrite_existing_data=overwrite_existing_data,
    )
    add_to_h5_if_not_exists(
        save_h5,
        "stopped_points",
        stopped_points,
        dtype=np.uint16,
        overwrite_existing_data=overwrite_existing_data,
    )
    add_to_h5_if_not_exists(
        save_h5,
        "movement_across_subtrajectory",
        movement_across_subtrajectory,
        overwrite_existing_data=overwrite_existing_data,
    )
    add_to_h5_if_not_exists(
        save_h5,
        "masked_frames",
        masked_frames,
        compression="gzip",
        dtype=np.uint8,
        compression_opts=9,
        overwrite_existing_data=overwrite_existing_data,
    )
    add_to_h5_if_not_exists(
        save_h5,
        "traj_splits_indices",
        traj_splits,
        dtype=np.uint16,
        overwrite_existing_data=overwrite_existing_data,
    )