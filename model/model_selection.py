import torch
import torch.nn as nn

class ModelSelector:
    """
    사용할 모델 유형을 선택하는 클래스.
    """
    def __init__(
        self, 
        model_type: str, 
        num_classes: int, 
        **kwargs
    ):
        # 모델 유형을 소문자로 변환
        model_type = model_type.lower()
        
        if model_type == '':
            pass
        elif model_type == '':
            pass
        else:
            raise ValueError("Unknown model type specified.")

    def get_model(self) -> nn.Module:

        # 생성된 모델 객체 반환
        return self.model