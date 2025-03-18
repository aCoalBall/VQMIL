VQMIL
======

## Installation and Dependencies

```
conda create --name vqmil python=3.12
conda activate vqmil
pip3 install -r requirements.txt
```

## Train on Camelyon16

After downloading [Camelyon16](https://camelyon16.grand-challenge.org/Data/) dataset, please refer to the guidelines in [CLAM](https://github.com/mahmoodlab/CLAM) to extract embedding vectors using the following command.

```
python create_patches_fp.py --source $YOUR_WSI_DIR --save_dir $TILES_DIR --patch_level 1 --patch_size 224 --step_size 224 --seg --patch --stitch --use_ostu

python extract_features_fp.py --data_h5_dir $TILES_DIR --data_slide_dir $YOUR_WSI_DIR --csv_path $INFO_CSV --feat_dir $EMBEDDING_DIR --batch_size 512 --slide_ext .tif
```
Then you can get a directory with .pt files of embedding vectors.
Sampling pesudo bags using this directory
```
python main.py --task sampling --feature_dir $PT_FILES
```
Training
```
python main.py --task vqmil --dataset camelyon16 \
    --feature_dir $PT_FILES \
    --pseudo_feature_dir $PSEUDO_PT_FILES
```
