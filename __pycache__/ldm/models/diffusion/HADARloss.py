import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ldm.modules.HADARNet.semantic_segmentor import SemanticSegmentor
from ldm.modules.HADARNet.emissivity_prior_loss import EmissivityPriorLoss


class HADARloss():
    def __init__(self, vnums=4, loss_type='MSE', use_semantic=False,
                 seg_pretrained=True, seg_encoder='mit_b2',
                 prior_weight=0.15, prior_conf_thresh=0.70):
        self.vnums = vnums
        if loss_type == 'L1':
            self.loss = nn.L1Loss()
        elif loss_type == 'MSE':
            self.loss = nn.MSELoss()
        else:
            pass

        self.use_semantic = use_semantic
        if use_semantic:
            self.seg_model = SemanticSegmentor(
                encoder_name=seg_encoder,
                pretrained=seg_pretrained,
            )
            self.seg_model.eval()
            for param in self.seg_model.parameters():
                param.requires_grad = False

            self.emissivity_prior_loss = EmissivityPriorLoss(
                confidence_threshold=prior_conf_thresh,
                weight=prior_weight,
            )
            print(f"[HADARloss] Semantic guidance enabled. "
                  f"Seg encoder: {seg_encoder}, "
                  f"Prior weight: {prior_weight}, "
                  f"Confidence threshold: {prior_conf_thresh}")
        else:
            self.seg_model = None
            self.emissivity_prior_loss = None

    def loss_rec(self, preds, x):
        x_mean = torch.mean(x, dim=1).unsqueeze(1)
        rec_img = self.rec(preds, x)
        loss = self.loss(rec_img, x_mean)
        return loss

    def loss_rec_with_semantic(self, preds, x, rgb_input):
        """Reconstruction loss + semantic emissivity prior."""
        x_mean = torch.mean(x, dim=1).unsqueeze(1)
        rec_img = self.rec(preds, x)
        loss = self.loss(rec_img, x_mean)

        if self.use_semantic and rgb_input is not None:
            seg_logits = self.seg_model(rgb_input)
            e = self.rec_e(preds)
            loss += self.emissivity_prior_loss(e, seg_logits)

        return loss

    def rec_e(self, preds):
        e = preds[:,0,:,:].unsqueeze(1)
        return e

    def rec_T(self, preds):
        T = preds[:,1,:,:].unsqueeze(1)
        return T

    def rec_env(self, preds, x_mean):
        b, _, h, w = preds.shape
        V = preds[:,2:2+self.vnums,:,:]
        h_split_nums = int(math.sqrt(self.vnums))
        w_split_nums = self.vnums // h_split_nums
        assert h_split_nums * w_split_nums == self.vnums
        x_beta = F.avg_pool2d(x_mean, (h // h_split_nums, w // w_split_nums)).reshape(b, 1, self.vnums)
        v_pred = V.reshape(b, self.vnums, h*w)
        env = torch.matmul(x_beta, v_pred)
        env = env.view(b, 1, h, w)
        return env

    def rec(self, preds, x):
        x_mean = torch.mean(x, dim=1)
        e = self.rec_e(preds)
        T = self.rec_T(preds)
        env = self.rec_env(preds, x_mean)
        rec_img = torch.mul(e, T) + torch.mul(1-e, env)
        return rec_img