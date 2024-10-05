import torch.optim as optim

def get_scheduler(
    lr_scheduler: str, 
    optimizer: optim.Optimizer, 
    kwargs: dict
    ) -> optim.lr_scheduler._LRScheduler:
    
    if lr_scheduler == 'stepLR':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            **kwargs
            # step_size=10
            # gamma=0.1
        )
    elif lr_scheduler == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **kwargs
            # mode='min',
            # factor=scheduler_gamma,
            # patience=steps_per_epoch,
            # verbose=True
        )
    elif lr_scheduler == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            **kwargs
            # T_max=100, 
            # eta_min=0.001
        )
    elif lr_scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            **kwargs 
            # T_0=50, 
            # T_mult=2, 
            # eta_min=0.001
        )
    else:
        raise ValueError(f"Unsupported scheduler: {lr_scheduler}")
    
    return scheduler