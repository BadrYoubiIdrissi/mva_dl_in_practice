# mva_dl_in_practice TP 1 code

This code contains the code for the TP1 of advanced deep learning in MVA

The code is structured modularily and can run multiple experiments easily. For example the following command works

```
python train.py -m hydra/launcher=ruche 
                   model=multitask 
                   datamodule=usps_colored 
                   model.layers_color=[],[8],[8,8] 
                   model.layers_digit=[],[8] 
                   logger.name=layer_sharing
```
Or 
```
python train.py -m hydra/launcher=ruche 
                   datamodule=usps 
                   model=custom_cnn 
                   logger.name=usps_optimizer_effect 
                   datamodule.batch_size=500 
                   model.lr=0.001,0.005,0.01,0.1
```
