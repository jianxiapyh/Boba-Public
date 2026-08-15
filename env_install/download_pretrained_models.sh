# Download the checkpoints for GroundedSAM2
mkdir data_process/groundedSAM_checkpoints
cd data_process/groundedSAM_checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../..

# Download the checkpoints for superglue
mkdir data_process/models/weights
cd data_process/models/weights
wget https://github.com/magicleap/SuperGluePretrainedNetwork/raw/refs/heads/master/models/weights/superglue_indoor.pth
wget https://github.com/magicleap/SuperGluePretrainedNetwork/raw/refs/heads/master/models/weights/superglue_outdoor.pth
wget https://github.com/magicleap/SuperGluePretrainedNetwork/raw/refs/heads/master/models/weights/superpoint_v1.pth
cd ../../..
