# Work in progress...

Codes and configurations for our model, trained weights (see below) have been uploaded.

Some codes (calculating 95HD, loss functions with SVLS, loading pipelines, etc.) which we directly code them in source codes of mmsegmentation package are being put together now, and will be released soon. The project is built upon [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) and we also borrow some codes from [vit-pytorch](https://github.com/lucidrains/vit-pytorch), deep thanks for their codes.

# Experiments
## Data acquisition
The Synapse dataset can be downloaded in [https://github.com/Beckschen/TransUNet](https://github.com/Beckschen/TransUNet).

The CPS dataset can be downloaded in [https://github.com/DengPingFan/PraNet](https://github.com/DengPingFan/PraNet).

The PanNuke dataset can be downloaded in [https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke).
## File format
```
├── data
    ├── SynapseDataset
    │   ├── images
    │   │   ├── training
    │   │   └── validation
    │   ├── annotations
    │       ├── training
    │       └── validation
    ├── CPSDataset
    │   ├── images
    │   │   ├── training
    │   │   └── validation
    │   │   └── validation_respectively
    │   │      └── CVC-ClinicDB
    │   │      └── CVC-ColonDB
    │   │      └── ETIS-LaribPolypDB
    │   │      └── Kvasir
    │   ├── annotations
    │       ├── training
    │       └── validation
    │       └── validation_respectively
    │          └── CVC-ClinicDB
    │          └── CVC-ColonDB
    │          └── ETIS-LaribPolypDB
    │          └── Kvasir
    ├── CLTSDataset
    │   ├── images
    │   │   ├── training
    │   │   └── validation
    │   ├── annotations
    │       ├── training
    │       └── validation
    ├── PanNukeDataset
        ├── Folds123
        │   ├── images
        │   │   ├── training
        │   │   └── validation
        │   ├── annotations
        │       ├── training
        │       └── validation
        ├── Folds132
        │   ├── images
        │   │   ├── training
        │   │   └── validation
        │   ├── annotations
        │       ├── training
        │       └── validation
        ├── Folds231
            ├── images
            │   ├── training
            │   └── validation
            ├── annotations
                ├── training
                └── validation
```
## Results
### SynapseDataset
![image](https://github.com/BerenChou/SGBTransNet/blob/main/results/Synapae_Results.jpg)
### CPSDataset
![image](https://github.com/BerenChou/SGBTransNet/blob/main/results/CVC-ClinicDB_Results.jpg)
![image](https://github.com/BerenChou/SGBTransNet/blob/main/results/Kvasir_Results.jpg)
![image](https://github.com/BerenChou/SGBTransNet/blob/main/results/CVC-ColonDB_Results.jpg)
![image](https://github.com/BerenChou/SGBTransNet/blob/main/results/ETIS_Results.jpg)
## Trained model
Trained weights of SGBTransNet for the Synapse dataset: [Google Drive](https://drive.google.com/file/d/1VR-3Nyz1yq2foorOZY-dxmCvhcyz-R9S/view?usp=sharing).

Trained weights of SGBTransNet for the CPS dataset: [Google Drive](https://drive.google.com/file/d/1jdrLCooxc03hhsAD5t9JBkhYTZFDE9fm/view?usp=sharing).
