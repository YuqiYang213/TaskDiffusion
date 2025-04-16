# Yuqi Yang

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.custom_collate import collate_mil

from easydict import EasyDict as edict
import copy
def edict2dict(inp):
    if type(inp) == dict:
        return {k: edict2dict(v) for k, v in inp.items()}
    elif type(inp) == edict:
        return {k: edict2dict(v) for k, v in inp.items()}
    else:
        return inp

def get_backbone(p):
    """ Return the backbone """

    if  p['backbone'] == 'TaskDiffusion_vitL':
        from models.transformers.TaskDiffusion_baseline import TaskDiffusion_vit_large_patch16_384
        backbone = TaskDiffusion_vit_large_patch16_384(p=p, pretrained=True, drop_path_rate=0.15, img_size=p.TRAIN.SCALE)
        backbone_channels = p.final_embed_dim
        p.backbone_channels = backbone_channels
        p.spatial_dim = [[p.TRAIN.SCALE[0]//16, p.TRAIN.SCALE[1]//16] for _ in range(4)]
    
    elif p['backbone'] == 'TaskDiffusion_vitB':
        from models.transformers.TaskDiffusion_baseline import TaskDiffusion_vit_base_patch16_384
        backbone = TaskDiffusion_vit_base_patch16_384(p=p, pretrained=True, drop_path_rate=0.15, img_size=p.TRAIN.SCALE)
        backbone_channels = p.final_embed_dim
        p.backbone_channels = backbone_channels
        p.spatial_dim = [[p.TRAIN.SCALE[0]//16, p.TRAIN.SCALE[1]//16] for _ in range(4)]


    else:
        raise NotImplementedError

    return backbone, backbone_channels


def get_head(p, backbone_channels, task):
    """ Return the decoder head """
    
    if p['head'] == 'conv':
        from models.transformers.MLoRE import ConvHead_up
        if p.TASKS.NUM_CLASSES[task] > 0:
            return ConvHead_up(backbone_channels, p.TASKS.NUM_CLASSES[task])
        else:
            return ConvHead_up(backbone_channels, p.TASKS.NUM_OUTPUTS[task])

    elif p['head'] == 'deconv':
        from models.transformers.MLoRE import DEConvHead
        if p.TASKS.NUM_CLASSES[task] > 0:
            return DEConvHead(backbone_channels, p.TASKS.NUM_CLASSES[task])
        else:
            return DEConvHead(backbone_channels, p.TASKS.NUM_OUTPUTS[task])

    elif p['head'] == 'TaskDiffusion_head':
        from models.transformers.TaskDiffusion_head import Crosstask_diffusion_decoder
        return Crosstask_diffusion_decoder(p = p,
                   bit_scale=p.bit_scales,
                   timesteps=p.timesteps,
                   randsteps=p.randsteps,
                   embed_dim=backbone_channels, 
                   module_layer=p.module_layer, 
                   with_auxiliary_head=p.with_aux,
                   num_layer=4)
    
    elif p['head'] == 'TaskDiffusion_mlore_head':
        from models.transformers.TaskDiffusion_mlore_head import Crosstask_diffusion_decoder_mlore
        return Crosstask_diffusion_decoder_mlore(p = p,
                   bit_scale=p.bit_scales,
                   timesteps=p.timesteps,
                   randsteps=p.randsteps,
                   embed_dim=backbone_channels, 
                   module_layer=p.module_layer, 
                   with_auxiliary_head=p.with_aux,
                   num_layer=4)

    else:
        raise NotImplementedError

def get_model(p):
    """ Return the model """

    backbone, backbone_channels = get_backbone(p)
    
    if p['model'] == 'TaskDiffusion':
        from models.TaskDiffusion_wrapper import TaskDiffusionWrapper
        feat_channels = backbone_channels
        heads = get_head(p, feat_channels, None)
        aux_heads = None
        model = TaskDiffusionWrapper(p, backbone, heads, aux_heads=aux_heads)
    
    elif p['model'] == 'TaskDiffusion_mlore':
        from models.TaskDiffusion_mlore_wrapper import TaskDiffusionMLoREWrapper
        feat_channels = backbone_channels
        heads = get_head(p, feat_channels, None)
        aux_heads = None
        model = TaskDiffusionMLoREWrapper(p, backbone, heads, aux_heads=aux_heads)

    else:
        raise NotImplementedError('Unknown model {}'.format(p['model']))
    return model


"""
    Transformations, datasets and dataloaders
"""
def get_transformations(p):
    """ Return transformations for training and evaluationg """
    from data import transforms
    import torchvision

    # Training transformations
    if p['train_db_name'] == 'NYUD' or p['train_db_name'] == 'PASCALContext':
        train_transforms = torchvision.transforms.Compose([ # from ATRC
            transforms.RandomScaling(scale_factors=[0.5, 2.0], discrete=False),
            transforms.RandomCrop(size=p.TRAIN.SCALE, cat_max_ratio=0.75),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.PhotoMetricDistortion(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=p.TRAIN.SCALE),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])

        # Testing 
        valid_transforms = torchvision.transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=p.TEST.SCALE),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])
        return train_transforms, valid_transforms

    else:
        return None, None


def get_train_dataset(p, transforms=None):
    """ Return the train dataset """

    db_name = p['train_db_name']
    print('Preparing train dataset for db: {}'.format(db_name))

    if db_name == 'PASCALContext':
        from data.pascal_context import PASCALContext
        database = PASCALContext(p.db_paths['PASCALContext'], download=False, split=['train'], transform=transforms, retname=True,
                                          do_semseg='semseg' in p.TASKS.NAMES,
                                          do_edge='edge' in p.TASKS.NAMES,
                                          do_normals='normals' in p.TASKS.NAMES,
                                          do_sal='sal' in p.TASKS.NAMES,
                                          do_human_parts='human_parts' in p.TASKS.NAMES,
                                          overfit=False)

    if db_name == 'NYUD':
        from data.nyud import NYUD_MT
        database = NYUD_MT(p.db_paths['NYUD_MT'], download=False, split='train', transform=transforms, do_edge='edge' in p.TASKS.NAMES, 
                                    do_semseg='semseg' in p.TASKS.NAMES, 
                                    do_normals='normals' in p.TASKS.NAMES, 
                                    do_depth='depth' in p.TASKS.NAMES, overfit=False)

    return database


def get_train_dataloader(p, dataset, sampler):
    """ Return the train dataloader """
    collate = collate_mil
    trainloader = DataLoader(dataset, batch_size=p['trBatch'], drop_last=True,
                             num_workers=p['nworkers'], collate_fn=collate, pin_memory=True, sampler=sampler)
    return trainloader


def get_test_dataset(p, transforms=None):
    """ Return the test dataset """

    db_name = p['val_db_name']
    print('Preparing test dataset for db: {}'.format(db_name))

    if db_name == 'PASCALContext':
        from data.pascal_context import PASCALContext
        database = PASCALContext(p.db_paths['PASCALContext'], download=False, split=['val'], transform=transforms, retname=True,
                                      do_semseg='semseg' in p.TASKS.NAMES,
                                      do_edge='edge' in p.TASKS.NAMES,
                                      do_normals='normals' in p.TASKS.NAMES,
                                      do_sal='sal' in p.TASKS.NAMES,
                                      do_human_parts='human_parts' in p.TASKS.NAMES,
                                    overfit=False)
    
    elif db_name == 'NYUD':
        from data.nyud import NYUD_MT
        database = NYUD_MT(p.db_paths['NYUD_MT'], download=False, split='val', transform=transforms, do_edge='edge' in p.TASKS.NAMES, 
                                    do_semseg='semseg' in p.TASKS.NAMES, 
                                    do_normals='normals' in p.TASKS.NAMES, 
                                    do_depth='depth' in p.TASKS.NAMES)

    else:
        raise NotImplemented("test_db_name: Choose among PASCALContext and NYUD")

    return database


def get_test_dataloader(p, dataset):
    """ Return the validation dataloader """
    collate = collate_mil
    testloader = DataLoader(dataset, batch_size=p['valBatch'], shuffle=False, drop_last=False,
                            num_workers=p['nworkers'], pin_memory=True, collate_fn=collate)
    return testloader


""" 
    Loss functions 
"""
def get_loss(p, task=None):
    """ Return loss function for a specific task """

    if task == 'edge':
        from losses.loss_functions import BalancedBinaryCrossEntropyLoss
        criterion = BalancedBinaryCrossEntropyLoss(pos_weight=p['edge_w'], ignore_index=p.ignore_index)

    elif task == 'semseg' or task == 'human_parts':
        from losses.loss_functions import CrossEntropyLoss
        criterion = CrossEntropyLoss(ignore_index=p.ignore_index)

    elif task == 'normals':
        from losses.loss_functions import L1Loss
        criterion = L1Loss(normalize=True, ignore_index=p.ignore_index)

    elif task == 'sal':
        from losses.loss_functions import CrossEntropyLoss
        criterion = CrossEntropyLoss(balanced=True, ignore_index=p.ignore_index) 

    elif task == 'depth':
        from losses.loss_functions import L1Loss
        criterion = L1Loss(ignore_invalid_area=p.ignore_invalid_area_depth, ignore_index=0)

    else:
        criterion = None

    return criterion


def get_criterion(p):
    from losses.loss_schemes import MultiTaskLoss
    loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in p.TASKS.NAMES})
    loss_weights = p['loss_kwargs']['loss_weights']
    return MultiTaskLoss(p, p.TASKS.NAMES, loss_ft, loss_weights)


"""
    Optimizers and schedulers
"""
def get_optimizer(p, model):
    """ Return optimizer for a given model and setup """

    print('Optimizer uses a single parameter group - (Default)')
    params = model.parameters()

    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])
    
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    # get scheduler
    if p.scheduler == 'poly':
        from utils.train_utils import PolynomialLR
        scheduler = PolynomialLR(optimizer, p.max_iter, gamma=0.9, min_lr=0)
    elif p.scheduler == 'step':
        scheduler = torch.optim.MultiStepLR(optimizer, milestones=p.scheduler_kwargs.milestones, gamma=p.scheduler_kwargs.lr_decay_rate)

    return scheduler, optimizer