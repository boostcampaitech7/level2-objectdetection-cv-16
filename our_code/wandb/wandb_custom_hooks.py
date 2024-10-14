from mmengine.hooks import Hook
from mmengine.registry import HOOKS
import wandb
import os

@HOOKS.register_module()
class WandbInitHook(Hook):
    def __init__(self, project="Project2", name="experiment", **kwargs):
        self.project = project
        self.name = name
        self.kwargs = kwargs
        self.last_logged_iter = -1  # 마지막으로 로깅한 iteration 추적


    def before_run(self, runner):
        os.environ["WANDB_API_KEY"] = "2a631ea744b03506a1330798e0724d1d917a821f"
        wandb.login()
        wandb.init(project=self.project, name=self.name, **self.kwargs)


    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        # 1000의 배수인 iteration에서만 로깅
        if runner.iter % 1000 == 0 and runner.iter != self.last_logged_iter:
            log_dict = {
                "iteration": runner.iter,
            }
        
            # Loss 가져오기
            if outputs and 'loss' in outputs:
                log_dict["train/iter_loss"] = outputs['loss'].item()
        
            # Learning rate 가져오기
            if hasattr(runner.optim_wrapper, 'get_lr'):
                log_dict["learning_rate"] = runner.optim_wrapper.get_lr()
        
            wandb.log(log_dict, step=runner.iter)
            self.last_logged_iter = runner.iter  # 마지막으로 로깅한 iteration 업데이트


    def after_train_epoch(self, runner):
        log_dict = {
            "epoch": runner.epoch,
        }
    
        # Learning rate 가져오기
        if hasattr(runner.optim_wrapper, 'get_lr'):
            log_dict["learning_rate"] = runner.optim_wrapper.get_lr()
    
        # runner.message_hub에서 손실 값을 가져옴
        if hasattr(runner, 'message_hub'):
            train_loss = runner.message_hub.get_scalar('train/loss').current()
            if train_loss is not None:
                log_dict["train/epoch_loss"] = train_loss
    
        wandb.log(log_dict, step=runner.epoch)


    def after_val_epoch(self, runner, metrics=None):
        log_dict = {}
    
        if metrics is not None:
            for metric_name in ['bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_75']:
                if metric_name in metrics:
                    log_dict[f'val/{metric_name}'] = metrics[metric_name]
        elif hasattr(runner, 'message_hub'):
            for metric_name in ['bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_75']:
                metric_value = runner.message_hub.get_scalar(f'val/{metric_name}').current()
                if metric_value is not None:
                    log_dict[f'val/{metric_name}'] = metric_value
    
        wandb.log(log_dict, step=runner.epoch)


    def after_run(self, runner):
        wandb.finish()
        
"""
학습 손실 (에폭 단위)
검증 메트릭 (mAP, recall 등)
학습률
반복 단위의 손실 (100번째 반복마다)


"""