# By Hanrong Ye
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch.nn as nn
import torch.nn.functional as F

INTERPOLATE_MODE = 'bilinear'

class TaskDiffusionWrapper(nn.Module):
    def __init__(self, p, backbone, heads, aux_heads=None):
        super(TaskDiffusionWrapper, self).__init__()
        self.tasks = p.TASKS.NAMES

        self.backbone = backbone
        self.heads = heads 
        self.aux_heads = aux_heads


    def forward(self, x, ground_truths, need_info=False, if_eval=True, aux_forward=False):
        img_size = x.size()[-2:]
        out = {}

        target_size = img_size

        task_features, info = self.backbone(x) 
        
        if aux_forward:
            out_real, out_aux = self.heads(task_features, ground_truths, if_eval)
            for t in self.tasks:
                out_real[t] = F.interpolate(out_real[t], target_size, mode=INTERPOLATE_MODE)
                out_aux[t] = F.interpolate(out_aux[t], target_size, mode=INTERPOLATE_MODE)
            out = [out_real, out_aux]
        else:
            out_real = self.heads(task_features, ground_truths, if_eval)
            # print(out_real)
            for t in self.tasks:
                out_real[t] = F.interpolate(out_real[t], target_size, mode=INTERPOLATE_MODE)
            out = out_real
        if need_info:
            return out, info
        else:
            return out
