import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR


def build_optimizer(model, config):
    name = config.TRAINER.OPTIMIZER
    lr = config.TRAINER.TRUE_LR

    is_finetune = config.TRAINER.IS_FINETUNE

    if is_finetune:
        special_params_names = config.TRAINER.SPECIAL_PARAMS_NAMES
        base_params_lr_ratio = config.TRAINER.BASE_PARAMS_LR_RATIO

        def is_special_param(param_name: str) -> bool:
            for s in special_params_names:
                if param_name.startswith(s) and \
                        (len(param_name) == len(s) or param_name[len(s)] == '.'):
                    return True
            return False

        special_params = list(filter(lambda kv: is_special_param(kv[0]), model.named_parameters()))
        base_params = list(filter(lambda kv: not is_special_param(kv[0]), model.named_parameters()))
        special_params = [kv[1] for kv in special_params]
        base_params = [kv[1] for kv in base_params]
        params = [
            {'params': base_params, 'lr': base_params_lr_ratio * lr},
            {'params': special_params, 'lr': lr},
        ]
    else:
        params = [{'params': model.parameters()}]

    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=config.TRAINER.ADAM_DECAY)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=config.TRAINER.ADAMW_DECAY)
    else:
        raise ValueError(f"TRAINER.OPTIMIZER = {name} is not a valid optimizer!")


def build_scheduler(config, optimizer):
    """
    Returns:
        scheduler (dict):{
            'scheduler': lr_scheduler,
            'interval': 'step',  # or 'epoch'
            'monitor': 'val_f1', (optional)
            'frequency': x, (optional)
        }
    """
    scheduler = {'interval': config.TRAINER.SCHEDULER_INTERVAL}
    name = config.TRAINER.SCHEDULER

    if name == 'MultiStepLR':
        scheduler.update(
            {'scheduler': MultiStepLR(optimizer, config.TRAINER.MSLR_MILESTONES, gamma=config.TRAINER.MSLR_GAMMA)})
    elif name == 'CosineAnnealing':
        scheduler.update(
            {'scheduler': CosineAnnealingLR(optimizer, config.TRAINER.COSA_TMAX)})
    elif name == 'ExponentialLR':
        scheduler.update(
            {'scheduler': ExponentialLR(optimizer, config.TRAINER.ELR_GAMMA)})
    else:
        raise NotImplementedError()

    return scheduler
