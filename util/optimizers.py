import torch.optim as optim

def get_optimizer(optimizer_name: str, model, kwargs: dict) -> optim.Optimizer:
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), **kwargs)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), **kwargs)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model.parameters(), **kwargs)
    elif optimizer_name == 'nadam':
        return optim.NAdam(model.parameters(), **kwargs)
    elif optimizer_name == 'radam':
        return optim.RAdam(model.parameters(), **kwargs)
    elif optimizer_name == 'adagrad':
        return optim.Adagrad(model.parameters(), **kwargs)
    elif optimizer_name == 'adadelta':
        return optim.Adadelta(model.parameters(), **kwargs)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), **kwargs) # 가중치감소를 직접 적용 -> L2 정규화 강화
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
"""
optimizer_name = 'adam'  (adam, sgd, rmsprop)
learning_rate = 1e-3
optimizer = get_optimizer(model, optimizer_name, lr=learning_rate)
"""