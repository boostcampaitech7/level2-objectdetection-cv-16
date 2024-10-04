import torch.optim as optim

def get_optimizer(optimizer_name: str, model, kwargs: dict) -> optim.Optimizer:
    optimizer_name = optimizer_name.lower()
    
    params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_name == 'adam':
        return optim.Adam(params, **kwargs)
    elif optimizer_name == 'sgd':
        return optim.SGD(params, **kwargs)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(params, **kwargs)
    elif optimizer_name == 'nadam':
        return optim.NAdam(params, **kwargs)
    elif optimizer_name == 'radam':
        return optim.RAdam(params, **kwargs)
    elif optimizer_name == 'adagrad':
        return optim.Adagrad(params, **kwargs)
    elif optimizer_name == 'adadelta':
        return optim.Adadelta(params, **kwargs)
    elif optimizer_name == 'adamw':
        return optim.AdamW(params, **kwargs) # 가중치감소를 직접 적용 -> L2 정규화 강화
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
"""
optimizer_name = 'adam'  (adam, sgd, rmsprop)
learning_rate = 1e-3
optimizer = get_optimizer(model, optimizer_name, lr=learning_rate)
"""