**Data Download:** 
Please follow the instructions [here](https://www.notion.so/aminas-sftp-24179379a0224abcaf572e275c275aea?pvs=4). 
Download the following folder and files.
- /volume1/guestusers/KRAFTON/lrs3/
- /volume1/guestusers/KRAFTON/lrs3_2d_pretrain_b_1024.csv
- /volume1/guestusers/KRAFTON/lrs3_2d_pretrain_b_128.csv
- /volume1/guestusers/KRAFTON/hic-mae7/

Put /volume1/guestusers/KRAFTON/lrs3/ folder in your local ${DATA_ROOT} directory.  

Put /volume1/guestusers/KRAFTON/lrs3_2d_pretrain_b_1024.csv, /volume1/guestusers/KRAFTON/lrs3_2d_pretrain_b_128.csv files in ./preprocess directory.  

After downloading, the folder structures will be as below.  
```
    ${DATA_ROOT}
    ├── lrs3        # LRS3 Dataset
    │   ├── 00j9bKdiOjk
    │   │   ├── 00001_00
    │   │   ├── 00002_00
    │   │   ├── ...
    │   │   └── ...             
    │   ├── 04SEzifEsGg                                        
    │   ├── ...          
    └── └── ...                 
     

    AVRep
    ├── preprocess        
        ├── lrs3_2d_pretrain_b_1024.csv
        ├── lrs3_2d_pretrain_b_128.csv
        ├── ... 
        └── ...
```

**Prepare the environment:**
```bash
conda env create -f environment.yaml
conda activate hicmae
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```
만약, 다음 환경으로 에러가 계속 발생하면  
- /volume1/guestusers/KRAFTON/hic-mae7

환경을 복사해서 사용해주세요!

```bash
conda create -n hicmae --clone hic-mae7
```

**Run script:** 

1. LRS3 dataset
```bash
sh ./scripts/voxceleb2/audio_visual/hicmae_pretrain_base/pretrain_base_cy_syncnet_lrs3_1024.sh ${DATA_ROOT}
```

2. Voxceleb2 dataset
```bash
sh ./scripts/voxceleb2/audio_visual/hicmae_pretrain_base/pretrain_base_cy_syncnet_lrs3_128.sh ${DATA_ROOT}
```
