from oxe_configs import OXE_DATASET_CONFIGS
from tqdm import tqdm
import json
import os
import torch
import numpy as np
import h5py
import imageio

from point_tracking_utils import (
    setup_predictors,
    track_objects_in_video,
    add_all_keys_to_h5,
)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# plot
DETECTRON_BATCH_SIZE = 64

# prevent TFDS from taking up all GPU memory
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow_datasets as tfds


from tap import Tap

POSSIBLE_LANG_INSTRUCTION_KEYS = [
    "natural_language_instruction",
    "language_instruction",
    "instruction",
]

MAX_LANGTABLE_EPISODES = 100_000 # for language table, we only want to label the first 100k episodes b/c it's way too many

DATASET_TO_ARGS = {
    "droid": {
        "grid_size": 30,  # objects are small
    },
    "bridge_v2": {
        "movement_threshold": 0.10, # double b/c BRIDGE movement is stable
        "future_n_frames": 3,  # need to look less far ahead because robot moves fast
    },
    "jaco_play": {
        "movement_threshold": 0.02,  # lower movement threshold because picks end early, we want to detect the picked object even if just tiny amt of movement
        "stopping_threshold": 0.005,
    },
    "stanford_hydra_dataset_converted_externally_to_rlds": {
        "grid_size": 30, # objets small relative to camera pos
    },
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {
        "future_n_frames": 20,  # robot stops moving every so often, so need to have future_n_frames larger
    },
    "toto": {
        "grid_size": 30,
        "tracking_start_point": "end",  # pouring objects so start tracking from end to get poured objects
    },
    "berkeley_fanuc_manipulation": {
        "movement_threshold": 0.02, # press the stapler and similar tasks are small movements, so lower threshold
    },
}

DATASET_TRANSFORMS = (
    # Datasets used for OpenVLA: https://openvla.github.io/
    "fractal20220817_data 0.1.0 resize_and_jpeg_encode",
    "bridge_v2 0.1.0 resize_and_jpeg_encode",
    "kuka 0.1.0 resize_and_jpeg_encode,filter_success",
    "taco_play 0.1.0 resize_and_jpeg_encode",
    "jaco_play 0.1.0 resize_and_jpeg_encode",
    "berkeley_cable_routing 0.1.0 resize_and_jpeg_encode",
    "roboturk 0.1.0 resize_and_jpeg_encode",
    "viola 0.1.0 resize_and_jpeg_encode",
    "berkeley_autolab_ur5 0.1.0 resize_and_jpeg_encode,flip_wrist_image_channels",
    "toto 0.1.0 resize_and_jpeg_encode",
    "language_table 0.1.0 resize_and_jpeg_encode",
    "stanford_hydra_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode,flip_wrist_image_channels,flip_image_channels",
    "austin_buds_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode",
    "nyu_franka_play_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode",
    "furniture_bench_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode",
    "ucsd_kitchen_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode",
    "austin_sailor_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode",
    "austin_sirius_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode",
    "bc_z 0.1.0 resize_and_jpeg_encode",
    "dlr_edan_shared_control_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode",
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode",
    "utaustin_mutex 0.1.0 resize_and_jpeg_encode,flip_wrist_image_channels,flip_image_channels",
    "berkeley_fanuc_manipulation 0.1.0 resize_and_jpeg_encode,flip_wrist_image_channels,flip_image_channels",
    "cmu_stretch 0.1.0 resize_and_jpeg_encode",
    "dobbe 0.0.1 resize_and_jpeg_encode",
    "fmb 0.0.1 resize_and_jpeg_encode",
    "droid 1.0.0 resize_and_jpeg_encode",
)


class Args(Tap):
    tfds_path: str  # Path to the TFDS files.
    # task: str = None  # The task description.
    debug: bool = False  # Whether to use a smaller model for debugging.
    save_dir: str  # Directory to save the tracked video.
    input_size: tuple = None  # (480, 480)  # The size of the input images to Molmo.
    visualize: bool = False  # Whether to visualize the object tracking with matplotlib.
    specific_tasks: tuple = None  # Specific tasks in OXE to track objects for.
    device: str = "cuda:0"  # The device to use for inference.
    grid_size: int = 20  # The grid size for cotracker free-form point tracking.
    movement_threshold: float = 0.05  # The default threshold for movement (relative to image size) for each tracked point to be considered significant. Will be ignored if the dataset_to_args dict contains the dataset.
    stopping_threshold: float = 0.005  # The default threshold for movement (relative to image size) for each tracked point to be considered stopped. Will be ignored if the dataset_to_args dict contains the dataset.
    future_n_frames: int = 5  # The number of frames to look ahead for stopping points.
    vid_max_length: int = (
        100  # The maximum length of the video to track. Will downsample to this if >.
    )
    tracking_start_point: str = (
        "middle"  # one of ["middle", "start", "end"], where to start cotracking.
    )
    overwrite_existing_data: bool = False

if __name__ == "__main__":
    # TODO: argparse the stuff and aligin with some file to parse and process all mp4 files with their associated task descriptions
    # video_path = "videos/real/real_multiview_vid.mp4"
    # task = "put the m and m's in the the mug"
    # image_size = (480, 480)

    args = Args().parse_args()

    # setup the video predictors
    torch_device = torch.device(args.device)
    cotracker, gripper_predictor = setup_predictors(torch_device, args.debug)

    dataset_names = [x.split()[0] for x in DATASET_TRANSFORMS]

    default_args = {
        "grid_size": args.grid_size,
        "movement_threshold": args.movement_threshold,
        "stopping_threshold": args.stopping_threshold,
        "future_n_frames": args.future_n_frames,
        "tracking_start_point": args.tracking_start_point,
    }

    if args.specific_tasks is not None:
        # overwrite the dataset_names with the specific tasks
        dataset_names = args.specific_tasks

    for dataset_name in tqdm(dataset_names):
        dataset = tfds.load(dataset_name, data_dir=args.tfds_path, split="train")
        n_samples = 10 if args.debug else 1000000000000000000
        if dataset_name == "language_table":
            n_samples = MAX_LANGTABLE_EPISODES
        valid_samples = 0
        img_key_to_name = OXE_DATASET_CONFIGS[dataset_name][
            "image_obs_keys"
        ]  # dict mapping img_keys to the names of the images in OXE
        point_tracking_args = {}
        if dataset_name in DATASET_TO_ARGS:
            point_tracking_args = DATASET_TO_ARGS[dataset_name]
        # Only update keys that don't already exist in point_tracking_args
        point_tracking_args.update(
            {k: v for k, v in default_args.items() if k not in point_tracking_args}
        )

        # if save_dir is not provided, save in the same directory as the video
        this_dataset_save_dir = os.path.join(args.save_dir, dataset_name)
        # keep track of which files have been saved
        finished_video_path_json_path = os.path.join(
            this_dataset_save_dir, "oxe_finished_videos.json"
        )

        if os.path.exists(finished_video_path_json_path):
            try:
                with open(finished_video_path_json_path, "r") as f:
                    finished_video_paths = json.load(f)
            except json.JSONDecodeError:
                print(
                    f"Warning: Could not decode JSON from {finished_video_path_json_path}. Starting with empty dict."
                )
                finished_video_paths = {}
        else:
            finished_video_paths = {}

        os.makedirs(this_dataset_save_dir, exist_ok=True)

        # remove the keys which have None values
        img_key_to_name = {k: v for k, v in img_key_to_name.items() if v is not None}

        # don't use wrist for now for masking #TODO: change this later
        if "wrist" in img_key_to_name:
            img_key_to_name.pop("wrist")

        # check finished_video_paths for the first episode that doesn't exist
        if os.path.exists(
            os.path.join(args.save_dir, dataset_name, "dataset_movement_and_masks.h5")
        ):
            with h5py.File(
                os.path.join(
                    args.save_dir, dataset_name, "dataset_movement_and_masks.h5"
                ),
                "r",
            ) as save_h5:
                for i in range(dataset.cardinality().numpy()):
                    this_episode_name = f"{dataset_name}_ep{i}"
                    if this_episode_name not in finished_video_paths:
                        # check if each key img_key_to_name is in finished_video_paths[this_episode_name] and
                        # check if each of the elements has an item
                        if this_episode_name not in save_h5:
                            break
                        for key in img_key_to_name.keys():
                            if (
                                key not in save_h5[this_episode_name]
                                or len(save_h5[this_episode_name][key]) == 0
                            ):
                                missing_key = True
                                break
                        # missing a key, so we must relabel this episode by breaking out of this loop
                        if missing_key:
                            break

            num_episodes_to_skip = i
            dataset = dataset.skip(num_episodes_to_skip)
        else:
            num_episodes_to_skip = 0

        for i, episode in enumerate(dataset):
            i += num_episodes_to_skip
            # print progress
            print(
                f"------------------- Processing episode {i+1} out of {min(dataset.cardinality().numpy(), n_samples)} of dataset {dataset_name} -------------------"
            )
            # skip if we have already saved the video
            this_episode_name = f"{dataset_name}_ep{i}"
            if (
                this_episode_name in finished_video_paths
            ):
                valid_samples += 1
                continue

            # actually store the images
            img_key_to_episode_images = {k: [] for k in img_key_to_name.keys()}

            if valid_samples >= n_samples:
                break
            # task is the language instruction
            task = None
            for _, step in enumerate(episode["steps"]):
                # skip data loading if no lang
                if task is None or task == "":
                    for key in POSSIBLE_LANG_INSTRUCTION_KEYS:
                        if key in step["observation"]:
                            if dataset_name == "language_table":
                                task = step["observation"][key].numpy()
                                task = bytes(task[np.where(task != 0)].tolist()).decode(
                                    "utf-8"
                                )
                            else:
                                task = step["observation"][key].numpy().decode()
                            break
                        elif key in step:
                            task = step[key].numpy().decode()
                            break
                for img_key, img_name in img_key_to_name.items():
                    # extract video
                    img_key_to_episode_images[img_key].append(
                        step["observation"][img_name].numpy()
                    )

            if task is None or task == "":
                # if the language instruction is None for some reason, skip the episode as it's weird
                print(
                    f"Skipping episode {i + 1} of dataset {dataset_name} as the task is None or empty."
                )
                print(
                    f"Keys in step: {step.keys()} and keys in observation: {step['observation'].keys()}"
                )
                finished_video_paths[this_episode_name] = f"{dataset_name}_ep{i}"
                with open(finished_video_path_json_path, "w") as f:
                    json.dump(finished_video_paths, f)
                continue

            task_name_converted_for_saving = task.replace(" ", "_")

            valid_samples += 1

            with h5py.File(
                os.path.join(
                    args.save_dir, dataset_name, "dataset_movement_and_masks.h5"
                ),
                "a",
            ) as save_h5:
                # extract video from the episode
                for img_key in img_key_to_episode_images.keys():
                    img_key_to_episode_images[img_key] = np.stack(
                        img_key_to_episode_images[img_key]
                    )
                    if np.all(img_key_to_episode_images[img_key] == 0):
                        print(f"Skipping {img_key} as all images are 0.")
                        continue

                    # get the video
                    video_frames = img_key_to_episode_images[img_key]
                    episode_key = (
                        f"episode_{i}"  # for h5 for legacy reasons, not ep_{i}
                    )

                    # process the video
                    point_tracking_args = {
                        "grid_size": args.grid_size,
                        "movement_threshold": args.movement_threshold,
                        "stopping_threshold": args.stopping_threshold,
                        "future_n_frames": args.future_n_frames,
                        "tracking_start_point": args.tracking_start_point,
                    }

                    ret_tuple = track_objects_in_video(
                        cotracker,
                        gripper_predictor,
                        video_frames,
                        grid_size=point_tracking_args["grid_size"],
                        movement_threshold=point_tracking_args["movement_threshold"],
                        stopping_threshold=point_tracking_args["stopping_threshold"],
                        save_dir=os.path.join(
                            args.save_dir, dataset_name, f"{i}_{task}", img_key
                        ),
                        device=torch_device,
                        visualize=args.visualize,
                        detectron_batch_size=DETECTRON_BATCH_SIZE,
                        visualize_detailed_outputs=args.visualize,
                        vid_max_length=args.vid_max_length,
                        tracking_start_point=point_tracking_args[
                            "tracking_start_point"
                        ],
                    )

                    if ret_tuple is None:
                        continue
                    else:
                        (
                            gripper_positions,
                            significant_points,
                            stopped_points,
                            movement_across_video,
                            masked_frames,
                            traj_splits,
                        ) = ret_tuple
                        if dataset_name not in save_h5:
                            save_h5.create_group(dataset_name)
                        if episode_key not in save_h5[dataset_name]:
                            save_h5[dataset_name].create_group(episode_key)
                        if img_key not in save_h5[dataset_name][episode_key]:
                            save_h5[dataset_name][episode_key].create_group(img_key)

                        add_all_keys_to_h5(
                            save_h5[dataset_name][episode_key][img_key],
                            gripper_positions=gripper_positions,
                            significant_points=significant_points,
                            stopped_points=stopped_points,
                            movement_across_subtrajectory=movement_across_video,
                            masked_frames=masked_frames,
                            traj_splits=traj_splits,
                            overwrite_existing_data=args.overwrite_existing_data,
                        )

            # mark it as done by adding it to the finished_video_paths with the task name
            finished_video_paths[this_episode_name] = f"{dataset_name}_ep{i}"
            with open(finished_video_path_json_path, "w") as f:
                json.dump(finished_video_paths, f)
