

## Datasets 
- Split CIFAR10
- Split CIFAR100
- Split Mini-ImageNet

  
### Data preparation
- CIFAR10 & CIFAR100 will be downloaded during the first run
- Mini-ImageNet: Download from https://www.kaggle.com/whitemoon/miniimagenet/download , and place it in datasets/mini_imagenet/


## Run commands
Detailed descriptions of options can be found in [general_main.py](general_main.py)

### Sample commands to run algorithms on Split-CIFAR10
```shell
#ER
python general_main.py --data cifar10 --num_tasks 5 --cl_type nc --agent ER --retrieve random --update random --mem_size 1000

#MIR
python general_main.py --data cifar10 --num_tasks 5 --cl_type nc --agent ER --retrieve MIR --update random --mem_size 1000

#GSS
python general_main.py --data cifar10 --num_tasks 5 --cl_type nc --agent ER --retrieve random --update GSS --mem_size 1000 --epoch 10

#ASER
python general_main.py --data cifar10 --num_tasks 5 --cl_type nc --agent ER --retrieve ASER --update ASER --mem_size 1000 --n_smp_cls 9.0 --epoch 10

#SCR
python general_main.py --data cifar10 --num_tasks 5 --cl_type nc --agent SCR --retrieve random --update random --mem_size 1000

#DVC
python general_main.py --data cifar10 --num_tasks 5 --cl_type nc --agent ER_DVC --retrieve MGI --update random --mem_size 1000 --dl_weight 2.0

#PCR
python general_main.py --data cifar10 --num_tasks 5 --cl_type nc --agent PCR --retrieve random --update random --mem_size 1000
```

### Sample command to add CASP
```shell
#ER + CASP
python general_main.py --data cifar10 --num_tasks 5 --cl_type nc --agent ER --retrieve random --update random --mem_size 1000 --CASP True --CASP_Epoch 4
```

## Repo Structure & Description
    ├──agents                       #Files for different algorithms
        ├──base.py                      #Abstract class for algorithms
        ├──exp_replay.py                #File for ER, MIR, GSS and ASER
        ├──exp_replay_dvc.py            #File for DVC
        ├──pcr.py                       #File for PCR
        ├──scr.py                       #File for SCR
    
    ├──continuum                    #Files for create the data stream objects
        ├──dataset_scripts              #Files for processing each specific dataset
            ├──dataset_base.py              #Abstract class for dataset
            ├──cifar10.py                   #File for CIFAR10
            ├──cifar100,py                  #File for CIFAR100
            ├──mini_imagenet.py             #File for Mini_ImageNet
        ├──continuum.py             
        ├──data_utils.py
    
    ├──models                       #Files for backbone models
        ├──resnet.py                    #Files for ResNet
    
    ├──utils                        #Files for utilities
        ├──buffer                       #Files related to buffer
            ├──aser_retrieve.py             #File for ASER retrieval
            ├──aser_update.py               #File for ASER update
            ├──aser_utils.py                #File for utilities for ASER
            ├──buffer.py                    #Abstract class for buffer
            ├──buffer_utils.py              #General utilities for all the buffer files
            ├──gss_greedy_update.py         #File for GSS update
            ├──mir_retrieve.py              #File for MIR retrieval
            ├──random_retrieve.py           #File for random retrieval
            ├──reservoir_update.py          #File for random update
    
        ├──name_match.py                #Match name strings to objects 
        ├──setup_elements.py            #Set up and initialize basic elements
        ├──utils.py                     #File for general utilities
        ├──loss.py                      #Contrastive loss

    ├──CASP.py                          #CASP codes (Our Method)
    ├──general_main.py                  #Detailed descriptions of options
    ├──loss.py                          #DVC's losses




## Acknowledgments
- [online-continual-learning](https://github.com/RaptorMai/online-continual-learning)
- [DVC](https://github.com/YananGu/DVC)
- [PCR](https://github.com/FelixHuiweiLin/PCR)


