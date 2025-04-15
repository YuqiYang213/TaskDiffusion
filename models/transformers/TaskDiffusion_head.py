import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import expm1
from einops import rearrange, reduce, repeat
import math


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))


def alpha_cosine_log_snr(t, ns=0.0002, ds=0.00025):
    # not sure if this accounts for beta being clipped to 0.999 in discrete version
    return -log((torch.cos((t + ns) / (1 + ds) * math.pi * 0.5) ** -2) - 1, eps=1e-5)


def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

class Fuse_withtime(nn.Module):
    def __init__(self, embed_dim, time_dim, num_task):
        super().__init__()
        self.num_task = num_task
        self.conv_1 = nn.Conv2d(embed_dim * num_task, embed_dim // 4, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(embed_dim // 4)
        self.act = nn.GELU()
        self.conv_2 = nn.Conv2d(embed_dim // 4, num_task * num_task, kernel_size=1)

        self.time_mlp = nn.Sequential(  # [2, 1024]
            nn.SiLU(),
            nn.Linear(time_dim, embed_dim  // 2)  # [2, 512]
        )
    def forward(self, x, t):
        # x shape: b, num_task * c , h, w
        x_ori = x
        b, _, h, w = x.shape
        x = self.conv_1(x)
        x = self.norm(x)
        time_emb = self.time_mlp(t)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        scale, shift = time_emb.chunk(2, dim=1)
        x = x * (scale + 1) + shift
        x = self.act(x)
        x = self.conv_2(x)

        x = F.softmax(x.reshape(b, self.num_task, self.num_task, 1, h, w), dim=1)
        x = x_ori.reshape(b, self.num_task, 1, -1, h, w) * x
        return (torch.sum(x, dim=1) + x_ori.reshape(b, self.num_task, -1, h, w))/ 2

class Conv_MLP_withtime_res(nn.Module):
    def __init__(self, embed_dim, time_dim):
        super().__init__()
        self.conv_1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(embed_dim)
        self.act = nn.GELU()

        self.time_mlp = nn.Sequential(  # [2, 1024]
            nn.SiLU(),
            nn.Linear(time_dim, embed_dim * 2)  # [2, 512]
        )
    def forward(self, x, t):
        # print(x.shape)
        x_ori = x
        x = self.conv_1(x)
        x = self.norm(x)
        time_emb = self.time_mlp(t)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        scale, shift = time_emb.chunk(2, dim=1)
        x = x * (scale + 1) + shift
        x = self.act(x)
        x = self.conv_2(x)
        return x + x_ori


class Conv_MLP_withtime(nn.Module):
    def __init__(self, embed_dim, out_dim, time_dim):
        super().__init__()
        self.conv_1 = nn.Conv2d(embed_dim, out_dim, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(out_dim, out_dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = nn.GELU()

        self.time_mlp = nn.Sequential(  # [2, 1024]
            nn.SiLU(),
            nn.Linear(time_dim, out_dim * 2)  # [2, 512]
        )
    def forward(self, x, t):
        # print(x.shape)
        x = self.conv_1(x)
        x = self.norm(x)
        time_emb = self.time_mlp(t)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        scale, shift = time_emb.chunk(2, dim=1)
        x = x * (scale + 1) + shift
        x = self.act(x)
        x = self.conv_2(x)
        return x

class Transform_withtime(nn.Module):
    def __init__(self, embed_dim, output_dim, num_layer=4):
        super().__init__()
        self.conv_transforms = nn.ModuleList()
        for i in range(num_layer):
            self.conv_transforms.append(nn.Conv2d(
                    embed_dim + output_dim,
                    embed_dim,
                    1,
                    padding=0,
                ))
        self.num_layer = num_layer
        self.embed_dim = embed_dim
    def forward(self, x, gt):
        b, c_l, h, w = x.shape
        assert(c_l == self.embed_dim * self.num_layer)
        x = rearrange(x, 'b (l c) h w -> b l c h w', l=self.num_layer, c=self.embed_dim)
        x_out = []
        for i in range(self.num_layer):
            x_i = x[:, i]
            feat = torch.cat([x_i, gt], dim=1)
            x_out.append(self.conv_transforms[i](feat))
        return torch.cat(x_out, dim=1)


class Mask_withtime(nn.Module):
    def __init__(self, embed_dim, time_dim, num_layer):
        super().__init__()
        self.conv_1 = nn.Conv2d(embed_dim * num_layer, embed_dim, kernel_size=1)
        self.conv_2 = nn.Conv2d(embed_dim, num_layer, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(embed_dim)
        self.act = nn.GELU()

        self.time_mlp = nn.Sequential(  # [2, 1024]
            nn.SiLU(),
            nn.Linear(time_dim, embed_dim * 2)  # [2, 512]
        )
    def forward(self, x, t):
        # print(x.shape)
        x = self.conv_1(x)
        x = self.norm(x)
        time_emb = self.time_mlp(t)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        scale, shift = time_emb.chunk(2, dim=1)
        x = x * (scale + 1) + shift
        x = self.act(x)
        x = self.conv_2(x)
        return x

class Conv_withtime(nn.Module):
    def __init__(self, embed_dim, out_dim, time_dim, kernel_size):
        super().__init__()
        self.conv_1 = nn.Conv2d(embed_dim, out_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.norm = nn.BatchNorm2d(out_dim)

        self.time_mlp = nn.Sequential(  # [2, 1024]
            nn.SiLU(),
            nn.Linear(time_dim, out_dim * 2)  # [2, 512]
        )
    def forward(self, x, t):
        # print(x.shape)
        x = self.conv_1(x)
        x = self.norm(x)
        time_emb = self.time_mlp(t)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        scale, shift = time_emb.chunk(2, dim=1)
        x = x * (scale + 1) + shift
        return x

class FPN_sharediff_withtime(nn.Module):
    def __init__(self, embed_dim, time_dim, num_layer=4, num_conv_layer=2):
        super().__init__()
        self.fpn_conv = nn.ModuleList()
        # self.task_mask = Mask_withtime(embed_dim, time_dim, num_layer)
        for i in range(num_layer):
            layers_cur = nn.ModuleList()
            for j in range(num_conv_layer):
                layers_cur.append(Conv_MLP_withtime(embed_dim, embed_dim, time_dim))
            self.fpn_conv.append(layers_cur)
        self.embed_dim = embed_dim
        self.num_layer = num_layer
        self.num_conv_layer = num_conv_layer
    def forward(self, x, t):
        b, c_l, h, w = x.shape
        assert(c_l == self.embed_dim * self.num_layer)
        x = rearrange(x, 'b (l c) h w -> b l c h w', l=self.num_layer, c=self.embed_dim)
        x_out = []
        for i in range(self.num_layer):
            x_cur = x[:, i]
            for j in range(self.num_conv_layer):
                x_cur = self.fpn_conv[i][j](x_cur, t)
            x_out.append(x_cur)
        return x_out
    
class FPN_withtime(nn.Module):
    def __init__(self, embed_dim, time_dim, tasks, num_layer=4, num_conv_layer=2):
        super().__init__()
        self.share_fpn = FPN_sharediff_withtime(embed_dim, time_dim, num_layer=4, num_conv_layer=num_conv_layer)
        self.tasks = tasks
        self.num_layer = num_layer
        self.embed_dim = embed_dim
        self.mask_task = nn.ModuleList()
        self.feat_task = nn.ModuleList()
        self.specific_head = nn.ModuleList()
        self.fpn_mask = nn.ModuleDict()
        for i in range(num_layer):
            self.mask_task.append(Conv_withtime(embed_dim, len(tasks) ** 2, time_dim, kernel_size=3))
            self.feat_task.append(nn.ModuleDict())
            self.specific_head.append(nn.ModuleDict())
            for task in tasks:
                self.feat_task[i][task] = nn.Sequential(nn.Conv2d(embed_dim , embed_dim, kernel_size=3, padding=1),
                                                nn.BatchNorm2d(embed_dim), nn.GELU(),  
                                                nn.Conv2d(embed_dim, embed_dim, kernel_size=1), 

                                                nn.Conv2d(embed_dim , embed_dim, kernel_size=3, padding=1),
                                                nn.BatchNorm2d(embed_dim), nn.GELU(),  
                                                nn.Conv2d(embed_dim, embed_dim, kernel_size=1), 
                                                )
                self.specific_head[i][task] = Conv_withtime(embed_dim, embed_dim, time_dim, kernel_size=1)
        for task in tasks:
            self.fpn_mask[task] = Mask_withtime(embed_dim, time_dim, num_layer=4)
    def forward(self, x, t, mask=None, x_ori=None):
        if mask is None:
            assert(x_ori is not None)
            b, c_l, h, w = x_ori.shape
            assert(c_l == self.embed_dim * self.num_layer)
            x_ori = rearrange(x_ori, 'b (l c) h w -> b l c h w', l=self.num_layer, c=self.embed_dim)
            mask = []
            for i in range(self.num_layer):
                mask.append({})
                for task in self.tasks:
                    mask[i][task] = self.feat_task[i][task](x_ori[:, i])
        assert(len(mask) == self.num_layer)
        share_fpn_feature = self.share_fpn(x, t) # list, every element is B, C, H, W
        # share_fpn_mask = []
        for i in range(self.num_layer):
            share_fpn_feature[i] = self.mask_task[i](share_fpn_feature[i], t)
            share_fpn_feature[i] =  rearrange(share_fpn_feature[i], 'b (l m) h w -> b l m h w', l=len(self.tasks), m=len(self.tasks))
        x_last = {}
        for idx, task in enumerate(self.tasks):
            x_layers = []
            for i in range(self.num_layer):
                share_task_now = 0
                for idx_2,  task_2 in enumerate(self.tasks):
                    share_task_now += torch.sigmoid(share_fpn_feature[i][:, idx, idx_2].unsqueeze(1)) * mask[i][task_2]
                task_spe_feature = share_task_now
                task_spe_feature = self.specific_head[i][task](task_spe_feature, t)
                x_layers.append(task_spe_feature)
            x_mask = self.fpn_mask[task](torch.cat(x_layers, dim=1), t)
            x_mask = torch.softmax(x_mask, dim=1)
            x_last[task] = 0
            for i in range(self.num_layer):
                x_last[task] = x_last[task] + x_mask[:, i].unsqueeze(1) * x_layers[i]
        return x_last, mask
    

class Conv_MLP(nn.Module):
    def __init__(self, embed_dim_in, embed_dim):
        super().__init__()
        self.conv_1 = nn.Conv2d(embed_dim_in, embed_dim, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.conv_3 = nn.Conv2d(embed_dim_in, embed_dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x_ori = self.conv_3(x)
        x = self.conv_1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv_2(x)
        return x + x_ori

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class Crosstask_diffusion_decoder(nn.Module):
    """Encoder Decoder segmentors.
    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 p, 
                 bit_scale=0.1,
                 timesteps=1,
                 randsteps=1,
                 time_difference=1,
                 learned_sinusoidal_dim=16,
                 sample_range=(0, 0.999),
                 noise_schedule='cosine',
                 diffusion='ddim',
                 accumulation=False,
                 embed_dim=384,
                 encode_module=Conv_MLP_withtime_res,
                 module_layer=4,
                 with_auxiliary_head=False,
                 num_layer=4,
                 ):
        super(Crosstask_diffusion_decoder, self).__init__()

        self.bit_scale = bit_scale
        self.timesteps = timesteps
        self.randsteps = randsteps
        self.diffusion = diffusion
        self.time_difference = time_difference
        self.sample_range = sample_range
        self.use_gt = False
        self.accumulation = accumulation
        self.with_auxiliary_head = with_auxiliary_head
        self.p = p
        all_tasks = p.TASKS.NAMES
        all_classes = p.TASKS.NUM_CLASSES
        all_outputs = p.TASKS.NUM_OUTPUTS
        if with_auxiliary_head:
            self.aux_heads = nn.ModuleDict()
            for task in all_tasks:
                self.aux_heads[task] = nn.Sequential(Conv_MLP(embed_dim * num_layer, embed_dim), 
                                          nn.Conv2d(embed_dim, all_classes[task], kernel_size=3, padding=1) if all_classes[task] > 0 else nn.Conv2d(embed_dim, all_outputs[task], kernel_size=3, padding=1))
        self.embed_dim = embed_dim
        self.embedding_tables = nn.ModuleDict()
        self.invalid_embedding_tables = nn.ModuleDict()
        self.pred_heads = nn.ModuleDict()
        for task in all_tasks:
            if all_classes[task] > 0:
                self.embedding_tables[task] = nn.Embedding(all_classes[task] + 1, embed_dim // 4)
                self.pred_heads[task] = nn.Conv2d(embed_dim, all_classes[task], kernel_size=3, padding=1)
            else:
                self.embedding_tables[task] = nn.Conv2d(all_outputs[task], embed_dim // 4, kernel_size=1)
                self.invalid_embedding_tables[task] = nn.Embedding(1, embed_dim // 4)
                self.pred_heads[task] = nn.Conv2d(embed_dim, all_outputs[task], kernel_size=3, padding=1)
        if 'sal' in all_tasks:
            self.embed_all = nn.Conv2d((embed_dim // 4) * (len(all_tasks) - 1), embed_dim, kernel_size=1)
        else:
            self.embed_all = nn.Conv2d((embed_dim // 4) * len(all_tasks), embed_dim, kernel_size=1)

        print(f" timesteps: {timesteps},"
              f" randsteps: {randsteps},"
              f" sample_range: {sample_range},"
              f" diffusion: {diffusion}")

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')
        
        if 'sal' in all_tasks:
            self.mask_shape= (embed_dim // 4) * 5
        else:
            self.mask_shape= embed_dim
        self.transforms = Transform_withtime(embed_dim, self.mask_shape, num_layer)

        # time embeddings
        time_dim = embed_dim   # 1024
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(  # [2,]
            sinu_pos_emb,  # [2, 17]
            nn.Linear(fourier_dim, time_dim),  # [2, 1024]
            nn.GELU(),
            nn.Linear(time_dim, time_dim)  # [2, 1024]
        )

        self.encoders = nn.ModuleList()
        self.fuse_module = nn.ModuleList()
        # add fpn layer
        self.fpn_encoder = FPN_withtime(embed_dim, time_dim, all_tasks)
        for i in range(module_layer):
            self.encoders.append(nn.ModuleDict())
            for task in all_tasks:
                self.encoders[-1][task] = Conv_MLP_withtime_res(embed_dim, time_dim)
    
    def forward(self, x, gt, if_eval=True):
        if if_eval:
            return self.forward_test(x)
        else:
            return self.forward_train(x, gt)

    def forward_test(self, x):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        # assert(len(x) == 1)
        # x = x[0]
        if self.diffusion == "ddim":
            out = self.ddim_sample(x)
        else:
            raise NotImplementedError
        return out

    def forward_train(self, x, gt_semantic_seg):
        """Forward function for training.
        Args:
            img (Tensor): Input images.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        # backbone & neck
        batch, c, h, w, device, = *x['semseg'].shape, x['semseg'].device
        all_task = self.p.TASKS.NAMES
        all_classes = self.p.TASKS.NUM_CLASSES
        all_outputs = self.p.TASKS.NUM_OUTPUTS
        gt_down_all = []
        gt_down_split = []
        for task in all_task:
            if all_classes[task] > 0:
                gt_down = F.interpolate(gt_semantic_seg[task].float(), size=(h, w), mode="nearest")
                gt_down = gt_down.to(torch.long)
                gt_down[gt_down == 255] = all_classes[task]


                # print(gt_down.shape)
                gt_down = self.embedding_tables[task](gt_down).squeeze(1).permute(0, 3, 1, 2)
                # gt_down = (torch.sigmoid(gt_down) * 2 - 1) * self.bit_scale[task]
                if task == 'sal':
                    gt_down_split.append(gt_down)
                else:
                    gt_down_all.append(gt_down)
            else:
                gt_down = gt_semantic_seg[task]
                mask = torch.zeros(size=gt_down.shape).cuda().to(torch.long)

                mask_embed = self.invalid_embedding_tables[task](torch.zeros(size=(batch, 1, h, w)).cuda().to(torch.long)).squeeze(1).permute(0, 3, 1, 2)
                mask[gt_down == 255] = 1
                mask = torch.mean(mask.float(), dim=1).unsqueeze(1)
                gt_down = F.interpolate(gt_down.float(), size=(h, w), mode="bilinear")
                mask_down = F.interpolate(mask, size=(h, w), mode="bilinear")
                mask_embed_down = F.interpolate(mask_embed.float(), size=(h, w), mode="bilinear")
                gt_down = self.embedding_tables[task](gt_down)

                gt_down = mask_down * mask_embed_down + (1 - mask_down) * gt_down

                gt_down_all.append(gt_down)
        gt_down_all = torch.cat(gt_down_all, dim=1)
        # fuse the feature together
        gt_down_all = self.embed_all(gt_down_all)
        gt_down_split.append(gt_down_all)
        gt_down_all = torch.cat(gt_down_split, dim=1)
        gt_down_all = (torch.sigmoid(gt_down_all) * 2 - 1) * self.bit_scale['all']

        # sample time
        times = torch.zeros((batch,), device=device).float().uniform_(self.sample_range[0],
                                                                      self.sample_range[1])  # [bs]

        # random noise
        noise_level = self.log_snr(times)
        padded_noise_level = self.right_pad_dims_to(x['semseg'], noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)

        noise = torch.randn_like(gt_down_all)
        noised_gt = alpha * gt_down_all + sigma * noise

        # conditional input
        feat = self.transforms(x['semseg'], noised_gt) # feature for all tasks are the same

        input_times = self.time_mlp(noise_level)
        logits, feat_aux = self._decode_head_forward(feat, input_times, x_ori=x['semseg'])
        if self.with_auxiliary_head:
            logits_aux = {}
            for task in all_task:
                x_task_fpn = []
                for i in range(4):
                    x_task_fpn.append(feat_aux[i][task])
                x_task_fpn = F.interpolate(torch.cat(x_task_fpn, dim=1), scale_factor=4, mode='bilinear')
                logits_aux[task] = self.aux_heads[task](x_task_fpn)
            return [logits, logits_aux]
        return logits

    def _decode_head_forward(self, x, t, x_ori=None, mask=None, if_up=True):
        """Run forward function and calculate loss for decode head in
        inference."""
        # x shape:b, c, h//16, w//16
        l_all = len(self.encoders)
        all_task = self.p.TASKS.NAMES
        x, mask = self.fpn_encoder(x, t, x_ori=x_ori, mask=mask)
        for il, layer in enumerate(self.encoders):
            x_all = []
            for idx, task in enumerate(all_task):
                x_all.append(x[task])
            for idx, task in enumerate(all_task):
                x_all[idx] = x_all[idx].unsqueeze(1)
            x_all_fused = torch.cat(x_all, dim=1)
            for idx, task in enumerate(all_task):
                x[task] = x_all_fused[:, idx]
                if if_up and il == l_all - 1:
                    x[task] = F.interpolate(x[task], scale_factor=4, mode='bilinear')
                x[task] = layer[task](x[task], t)
        logits = {}
        for idx, task in enumerate(all_task):
            logits[task] = self.pred_heads[task](x[task])
        return logits, mask

    def right_pad_dims_to(self, x, t):
        padding_dims = x.ndim - t.ndim
        if padding_dims <= 0:
            return t
        return t.view(*t.shape, *((1,) * padding_dims))

    def _get_sampling_timesteps(self, batch, *, device):
        times = []
        for step in range(self.timesteps):
            t_now = 1 - (step / self.timesteps) * (1 - self.sample_range[0])
            t_next = max(1 - (step + 1 + self.time_difference) / self.timesteps * (1 - self.sample_range[0]),
                         self.sample_range[0])
            time = torch.tensor([t_now, t_next], device=device)
            time = repeat(time, 't -> t b', b=batch)
            times.append(time)
        return times

    def ddim_sample(self, x):
        b, c, h, w, device = *x['semseg'].shape, x['semseg'].device
        time_pairs = self._get_sampling_timesteps(b, device=device)
        all_task = self.p.TASKS.NAMES
        all_classes = self.p.TASKS.NUM_CLASSES
        for task in all_task:
            x[task] = repeat(x[task], 'b c h w -> (r b) c h w', r=self.randsteps)
        mask_t = torch.randn((self.randsteps * b, self.mask_shape, h, w), device=device)
        mask_t_truesize = mask_t
        mask_cur = None
        for idx, (times_now, times_next) in enumerate(time_pairs):
            feat = self.transforms(x['semseg'], mask_t)    # the feature for different tasks are the same 
            log_snr = self.log_snr(times_now)
            log_snr_next = self.log_snr(times_next)

            padded_log_snr = self.right_pad_dims_to(mask_t, log_snr)
            padded_log_snr_next = self.right_pad_dims_to(mask_t, log_snr_next)
            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            input_times = self.time_mlp(log_snr)
            if idx == (len(time_pairs) - 1):
                mask_t_truesize = F.interpolate(mask_t, scale_factor=4, mode='bilinear')
            mask_logits, mask_cur = self._decode_head_forward(feat, input_times, x_ori=x['semseg'], mask=mask_cur, if_up=(idx == (len(time_pairs) - 1)))  # a dictionary
            mask_pred_all = []
            gt_down_split = []
            for task in all_task:
                if all_classes[task] > 0:
                    if task != 'sal':
                        mask_pred = torch.argmax(mask_logits[task], dim=1)
                    else:
                        mask_pred = torch.where(torch.softmax(mask_logits[task], dim=1)[:, 1] > 0.85, 1, 0)
                    mask_pred = self.embedding_tables[task](mask_pred).permute(0, 3, 1, 2)
                    if task == 'sal':
                        gt_down_split.append(mask_pred)
                    else:
                        mask_pred_all.append(mask_pred)
                else:
                    if task == 'depth':
                        mask_pred = mask_logits[task].clamp_(0, 80)
                    elif task == 'edge':
                        mask_pred = torch.sigmoid(mask_logits[task])
                    elif task == 'normals':
                        mask_pred = F.normalize(mask_logits[task], p = 2, dim = 1)
                    mask_pred = self.embedding_tables[task](mask_pred)
                    mask_pred_all.append(mask_pred)
            mask_pred_all = torch.cat(mask_pred_all, dim=1)
            # fuse the embedding together
            mask_pred_all = self.embed_all(mask_pred_all)
            gt_down_split.append(mask_pred_all)
            mask_pred_all = torch.cat(gt_down_split, dim=1)
            mask_pred_all = (torch.sigmoid(mask_pred_all) * 2 - 1) * self.bit_scale['all']
            pred_noise = (mask_t_truesize - alpha * mask_pred_all) / sigma.clamp(min=1e-8)
            mask_t = mask_pred_all * alpha_next + pred_noise * sigma_next
            mask_t_truesize = mask_t
            mask_t = F.interpolate(mask_t, size=(h, w), mode='bilinear')
            
        for task in all_task:
            mask_logits[task] = rearrange(mask_logits[task], '(r b) c h w -> r b c h w', r=self.randsteps)
            mask_logits[task] = mask_logits[task].mean(dim=0)
        return mask_logits