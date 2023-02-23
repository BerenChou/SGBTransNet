WIP.

Configuration files and some python files (including dataset class files, loading pipelines, metric files, loss functions, etc.) which we directly place them in the mmseg package are being put together. We release the codes of our model first. The project is built upon [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) and we also borrow some codes from [vit-pytorch](https://github.com/lucidrains/vit-pytorch), deep thanks for their codes.

Trained weights of SGBTransNet for the Synapse dataset: [Google Drive](https://drive.google.com/file/d/1VR-3Nyz1yq2foorOZY-dxmCvhcyz-R9S/view?usp=sharing). More weights and testing screenshots are coming in no time.

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
