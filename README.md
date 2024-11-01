**Data Download:**  
Download the following folder and files.
- lrs3/
- lrs3_2d_pretrain_b_1024.csv
- lrs3_2d_pretrain_b_128.csv
- hic-mae7/

Put lrs3/ folder in your local ${DATA_ROOT} directory.  

Put lrs3_2d_pretrain_b_1024.csv, lrs3_2d_pretrain_b_128.csv files in ./preprocess directory.  

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
- hic-mae7/

환경을 복사해서 사용해주세요!

```bash
conda create -n hicmae --clone hic-mae7
```

**Run script:** 

1. lrs3_1024
```bash
sh ./scripts/voxceleb2/audio_visual/hicmae_pretrain_base/pretrain_base_cy_syncnet_lrs3_1024.sh ${DATA_ROOT}
```

2. lrs3_128
```bash
sh ./scripts/voxceleb2/audio_visual/hicmae_pretrain_base/pretrain_base_cy_syncnet_lrs3_128.sh ${DATA_ROOT}
```
