# breast_ICSDA
## Pre-requisties:
* Python (3.7.5), h5py (2.10.0), matplotlib (3.1.1), numpy (1.17.3), opencv-python (4.1.1.26), openslide-python (1.1.1), openslides (3.4.1), pandas (0.25.3), pillow (6.2.1), PyTorch (1.3.1), scikit-learn (0.22.1), scipy (1.3.1), tensorflow (1.14.0), tensorboardx (1.9), torchvision (0.4.2), smooth-topk.
## H&E-stained image Segmentation and Patching
The first step focuses on segmenting the tissue and excluding any holes. The segmentation of specific slides can be adjusted by tuning the individual parameters
```bash
DATA_DIRECTORY/
	├── slide_1.svs
	├── slide_2.svs
	└── ...
```
``` shell
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --seg --patch --stitch 
```

The above command will segment every slide in DATA_DIRECTORY using default parameters, extract all patches within the segemnted tissue regions, create a stitched reconstruction for each slide using its extracted patches (optional) and generate the following folder structure at the specified RESULTS_DIRECTORY:
```bash
RESULTS_DIRECTORY/
	├── masks
    		├── slide_1.png
    		├── slide_2.png
    		└── ...
	├── patches
    		├── slide_1.h5
    		├── slide_2.h5
    		└── ...
	├── stitches
    		├── slide_1.png
    		├── slide_2.png
    		└── ...
	└── process_list_autogen.csv
```
The **masks** folder contains the segmentation results (one image per slide).
The **patches** folder contains arrays of extracted tissue patches from each slide (one .h5 file per slide, where each entry corresponds to the coordinates of the top-left corner of a patch)
The **stitches** folder contains downsampled visualizations of stitched tissue patches (one image per slide) (Optional, not used for downstream tasks)
The auto-generated csv file **process_list_autogen.csv** contains a list of all slides processed, along with their segmentation/patching parameters used.
## Weakly-Supervised Learning using Slide-Level Labels 
```bash
python extract_features_fp.py --data_h5_dir DIR_TO_COORDS --data_slide_dir DATA_DIRECTORY --csv_path CSV_FILE_NAME --feat_dir FEATURES_DIRECTORY --batch_size  --slide_ext .svs
```

The above command expects the coordinates .h5 files to be stored under DIR_TO_COORDS and a batch size of 256 to extract 256-dim features from each tissue patch for each slide and produce the following folder structure:
```bash
FEATURES_DIRECTORY/
    ├── h5_files
            ├── slide_1.h5
            ├── slide_2.h5
            └── ...
    └── pt_files
            ├── slide_1.pt
            ├── slide_2.pt
            └── ...
```
## Datasets
The data used for training and testing are expected to be organized as follows:
```bash
DATA_ROOT_DIR/
    ├──DATASET_1_DATA_DIR/
        ├── h5_files
                ├── slide_1.h5
                ├── slide_2.h5
                └── ...
        └── pt_files
                ├── slide_1.pt
                ├── slide_2.pt
                └── ...
        └── ...
```
Namely, each dataset is expected to be a subfolder (e.g. DATASET_1_DATA_DIR) under DATA_ROOT_DIR, and the features extracted for each slide in the dataset is stored as a .pt file sitting under the **pt_files** folder of this subfolder.
Datasets are also expected to be prepared in a csv format containing at least 3 columns: **case_id**, **slide_id**, and 1 or more labels columns for the slide-level labels. Each **case_id** is a unique identifier for a patient, while the **slide_id** is a unique identifier for a slide that correspond to the name of an extracted feature .pt file. 
Dataset objects used for actual training/testing can be constructed using the **Generic_WSI_Classification_Dataset** Class (defined in **datasets/dataset_generic_tt_mtl.py**). Examples of such dataset objects passed to the models can be found in both **main.py** and **eval.py**. 

For training, look under main.py:
```python 
if args.task == 'Recurrence_metastasis_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '...',
                            data_dir="...",
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'0':0, '1':1},
                            patient_strat=False)
```
## Splits
``` shell
python create_splits_seq.py --task Recurrence_metastasis_vs_normal --seed 1 --label_frac 0.75 --k 1
```
## Training
``` shell
python main_tt_mtl.py --drop_out --early_stopping --lr 2e-3 --k 1  --split_dir /media/zw/Elements1/lvyp/splits/Recurrence_metastasis_vs_normal_100_mtl_5 --exp_code metastasis_vs_normal_balance_22  --task Recurrence_metastasis_vs_normal  --log_data  --data_root_dir /media/zw/Elements1/lvyp/data_features
```
## Heatmap Visualization
Heatmap visualization can be computed in bulk via **create_heatmaps.py** by filling out the config file and storing it in **/heatmaps/configs** and then running **create_heatmaps.py** with the --config NAME_OF_CONFIG_FILE flag. 
To run the demo (raw results are saved in **heatmaps/heatmap_raw_results** and final results are saved in **heatmaps/heatmap_production_results**):
``` shell
CUDA_VISIBLE_DEVICES=0,1 python create_heatmaps.py --config config_template.yaml
```
