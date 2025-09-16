# Installation
```bash
conda create -n point_tracking python=3.12 -y
conda activate point_tracking

git clone git@github.com:jesbu1/point_tracking.git
git submodule update --init --recursive

pip install -r requirements.txt
pip install "detectron2 @ git+https://github.com/facebookresearch/detectron2@9604f5995cc628619f0e4fd913453b4d7d61db3f"
gdown --fuzzy https://drive.google.com/file/d/1qQV-yPZHqW9Z_eKR_U0aTnkcKpQ1em9V/view # download the detectron gripper weights from LLARVA
mv model_final.pth detectron_gripper.pth
```

For co-tracker
```bash
pip install git+https://github.com/facebookresearch/co-tracker
```

To download openx, make sure the `rlds_dataset_mod` submodule is installed, check the paths are correct in `rlds_dataset_mod/prepare_open_x.sh` and run the script after:
```bash
cd rlds_dataset_mod 
conda env create -f environment_ubuntu.yml
conda activate rlds_env
bash prepare_open_x.sh
cd ..
```

However, this downloads an old version of BRIDGE.
To download the latest version, run the following script and put the path to the saved directory the same as the path to the saved directory for the openx datasets.
```bash
bash download_bridge_v2.sh [/path/to/save/tensorflow_datasets]
```

# Labeling Data with Points and Masks

To label OXE datasets, run the `run_openx_processing.sh` script.
See [README_processing_scripts.md](README_processing_scripts.md) for instructions.

The script now processes datasets sequentially instead of running multiple iterations of the same dataset:
- **SAVE_DIR required**: `./run_openx_processing.sh /path/to/save/directory` processes all datasets
- **With dataset index**: `./run_openx_processing.sh /path/to/save/directory 3` processes dataset at index 3 (see `run_openx_processing.sh` for the list of datasets)
- **With dataset name**: `./run_openx_processing.sh /path/to/save/directory bridge_v2` processes that specific dataset (see `run_openx_processing.sh` for the list of datasets)

To label LIBERO data, first download the processed OpenVLA-style-processed LIBERO data here ([LIBERO](https://huggingface.co/datasets/jesbu1/libero_90_openvla_processed)).
Then run the following script:
```bash
python label_videos_libero.py --save_dir /path/to/save/directory --dataset_location [LOCATION_OF_FOLDERS_CONTAINING_LIBERO_HDF5_FILES] --which_dataset [path/to/libero_90_openvla_processed]
```

# Q/A: How does the script keep track of which videos have been labeled?

The scripts keep track of processed videos in separate JSON files:

- `finished_videos.json`: Tracks videos that have been fully processed with tracking
- `labeled_videos.json`: Tracks videos that have been labeled with trajectory segments

### HDF5 File Structure

The processed data is saved in HDF5 files with the following structure:

```
dataset_movement_and_masks.h5
├── {dataset_name}/
│   ├── {episode_key}/
│   │   ├── {img_key}/
│   │   │   ├── gripper_positions: uint16 array
│   │   │   ├── significant_points: uint16 array
│   │   │   ├── stopped_points: uint16 array
│   │   │   ├── movement_across_subtrajectory: float array
│   │   │   ├── masked_frames: uint8 array (compressed with gzip)
│   │   │   └── traj_splits_indices: uint16 array
```

Where:
- `dataset_name`: Name of the dataset (e.g., "oxe")
- `episode_key`: Unique identifier for each episode in the dataset
- `img_key`: Camera view identifier (e.g., "primary", "secondary")
- `gripper_positions`: Binary mask indicating gripper positions
- `significant_points`: Binary mask indicating points with significant movement
- `stopped_points`: Binary mask indicating points that have stopped moving
- `movement_across_subtrajectory`: Array tracking movement across video frames
- `masked_frames`: Video frames with object masks applied (compressed)
- `traj_splits_indices`: Indices indicating trajectory splits

The data is organized hierarchically to maintain the relationship between datasets, episodes, and different camera views. All binary masks are stored as uint8 arrays for efficiency, and the masked frames are compressed using gzip compression to reduce file size. The trajectory labels are stored as variable-length strings to accommodate captions of different lengths.

Note: LIBERO is not like this structure.
It is missing `dataset_name` but is otherwise the same.
