import h5py
import numpy as np
import json
import os
import torch
import imageio
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from point_tracking_utils import (
    setup_predictors,
    track_objects_in_video,
    add_all_keys_to_h5,
)

# plot
DETECTRON_BATCH_SIZE = 64
# TODO: detect errors when none

from tap import Tap

img_key_to_name = {
    "primary": "agentview_rgb",
    "secondary": None,
}


class Args(Tap):
    # task: str = None  # The task description.
    debug: bool = False  # Whether to use a smaller model for debugging.
    save_dir: str  # Directory to save the tracked video.
    input_size: tuple = None  # (480, 480)  # The size of the input images to Molmo.
    visualize: bool = False  # Whether to visualize the object tracking with matplotlib.
    dataset_location: str = "/home/jeszhang/LIBERO/libero/datasets"  # Specific tasks in OXE to track objects for.
    which_dataset: str = "libero_90_openvla_processed" # The dataset to use for the task. If None processes all
    device: str = "cuda:0"  # The device to use for inference.
    grid_size: int = 20  # The grid size for cotracker free-form point tracking.
    movement_threshold: float = 0.05  # The default threshold for movement (relative to image size) for each tracked point to be considered significant. Will be ignored if the dataset_to_args dict contains the dataset.
    stopping_threshold: float = 0.005  # The default threshold for movement (relative to image size) for each tracked point to be considered stopped. Will be ignored if the dataset_to_args dict contains the dataset.
    vid_max_length: int = (
        100  # The maximum length of the video to track. Will downsample to this if >.
    )
    overwrite_existing_data: bool = False


if __name__ == "__main__":
    args = Args().parse_args()

    # setup the video predictors
    torch_device = torch.device(args.device)
    cotracker, gripper_predictor = setup_predictors(torch_device, args.debug)

    # keep track of which files have been saved
    finished_video_path_json_path = os.path.join(args.save_dir, "finished_videos.json")

    if os.path.exists(finished_video_path_json_path):
        with open(finished_video_path_json_path, "r") as f:
            finished_video_paths = json.load(f)
    else:
        finished_video_paths = {}
    # iterate through each LIBERO dataset
    for dataset in os.listdir(args.dataset_location):
        # collect all the h5 files in the dataset location
        dataset_path = os.path.join(args.dataset_location, dataset)
        if not os.path.isdir(dataset_path):
            continue
        elif args.which_dataset is not None and args.which_dataset not in dataset_path:
            continue
        h5_dataset_locations = []
        for file in os.listdir(dataset_path):
            if file.endswith(".hdf5"):
                h5_dataset_locations.append(os.path.join(dataset_path, file))
        print(f"Found {len(h5_dataset_locations)} h5 files in {dataset}")
        print(f"Processing dataset {dataset}")
        h5_num = 0
        # load dataset
        valid_samples = 0
        # if save_dir is not provided, save in the same directory as the video
        this_dataset_save_dir = os.path.join(args.save_dir, dataset)
        for h5_path in h5_dataset_locations:
            h5_num += 1
            with h5py.File(h5_path, "r") as dataset_h5:
                dataset_name = os.path.basename(h5_path).split(".")[0]

                os.makedirs(this_dataset_save_dir, exist_ok=True)
                n_samples = 5 if args.debug else 1000000000000000000
                point_tracking_args = {
                    "grid_size": args.grid_size,
                    "movement_threshold": args.movement_threshold,
                    "stopping_threshold": args.stopping_threshold,
                }

                # remove the keys which have None values
                img_key_to_name = {
                    k: v for k, v in img_key_to_name.items() if v is not None
                }

                for i, episode_key in enumerate(dataset_h5["data"].keys()):
                    # print progress
                    print(
                        f"------------------- Processing task {h5_num} episode {i+1} of dataset {dataset} -------------------"
                    )

                    # we're going to make an equivalent dataset that simply adds the masks
                    # skip if we have already saved the video
                    this_episode_name = f"{dataset}_{dataset_name}_ep{i}"

                    episode = dataset_h5["data"][episode_key]
                    # episode consists of actions, actions_abs, obs, states

                    if valid_samples >= n_samples:
                        break

                    # generate masked images first
                    masked_images = []

                    # actually store the images
                    img_key_to_episode_images = {
                        k: dataset_h5["data"][episode_key]["obs"][img_key_to_name[k]]
                        for k in img_key_to_name.keys()
                    }

                    task = None  # no names for this, all block stacking for now anyway
                    if "problem_info" in dataset_h5["data"].attrs:
                        task = json.loads(dataset_h5["data"].attrs["problem_info"])[
                            "language_instruction"
                        ] 
                    else:
                        # problem info is lost in openvla's no-op processed libero data
                        task = os.path.basename(h5_path).split(".")[0]

                    valid_samples += 1

                    # extract video from the episode
                    for img_key in img_key_to_episode_images.keys():
                        # get the video
                        video_frames = np.asarray(img_key_to_episode_images[img_key])
                        # flip the video...
                        video_frames = video_frames[:, ::-1, :, :].copy()
                        # process the video 
                        ret_tuple = track_objects_in_video(
                            cotracker,
                            gripper_predictor,
                            video_frames,
                            grid_size=point_tracking_args["grid_size"],
                            movement_threshold=point_tracking_args[
                                "movement_threshold"
                            ],
                            stopping_threshold=point_tracking_args[
                                "stopping_threshold"
                            ],
                            save_dir=os.path.join(
                                args.save_dir, dataset, f"{i}_{task}", img_key
                            ),
                            device=torch_device,
                            visualize=args.visualize,
                            detectron_batch_size=DETECTRON_BATCH_SIZE,
                            visualize_detailed_outputs=args.visualize,
                            vid_max_length=args.vid_max_length,
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
                            with h5py.File(
                                os.path.join(
                                    args.save_dir, "dataset_movement_and_masks.h5"
                                ),
                                "a",
                            ) as save_h5:
                                if dataset_name not in save_h5:
                                    save_h5.create_group(dataset_name)
                                if episode_key not in save_h5[dataset_name]:
                                    save_h5[dataset_name].create_group(episode_key)
                                if img_key not in save_h5[dataset_name][episode_key]:
                                    save_h5[dataset_name][episode_key].create_group(
                                        img_key
                                    )
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
                    finished_video_paths[this_episode_name] = task
                    with open(finished_video_path_json_path, "w") as f:
                        json.dump(finished_video_paths, f)
