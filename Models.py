from core.encoder import Encoder
from core.correlation_lookup import Lookup
from core.motion_encoder import Motion_Encoder
from core.MemAttention import MemAttention
from core.GRU import RAFT_GRU
from core.flow_predictor import Flow_Predictor
from core.upsample_flow import Upsample_Flow
from Loss import CascadingL1Loss
from Loss import PSNRLoss

from utils.consts import device, H, W, batch_sz, norm_fn, hidden_dim, GRU_iterations, loss_cascade_ratio, dropout

import torch
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self, inTrain: bool):
        super(BaseModel, self).__init__()

        self.curr_flow = torch.zeros((batch_sz, 2, H // 8, W // 8)).to(device)

        self.inTrain = inTrain

        self.siamese_encoder = Encoder(dropout=dropout)
        self.context_encoder = Encoder(dropout=dropout)
        self.motion_lookup = Lookup()
        self.motion_encoder = Motion_Encoder()
        self.mem_buffer = MemAttention()
        self.GRU = RAFT_GRU()
        self.flow_predictor = Flow_Predictor()
        self.upsampler = Upsample_Flow(norm_fn=norm_fn)

    def forward(self, x):
        return x


# class FlowModel(BaseModel):
#
#     def __init__(self, inTrain, dataset, epochs, optimizer):
#
#         super(FlowModel, self).__init__(dataset, epochs, optimizer)
#
#     def forward_loss_normal(self):
#         H = self.H
#         W = self.W
#         for epoch in range(self.epochs):
#             frame_pairs = self.slide_2_frames()
#             pair = next(frame_pairs)
#             flow_gt = pass # GET FROM DATASET!!!
#
#             curr_frame_idx = 0
#             while pair and curr_frame_idx < self.cutoff:
#
#                 frame1, frame2 = pair
#                 feat1 = self.siamese_encoder.forward(frame1)
#                 feat2 = self.siamese_encoder.forward(frame2)
#                 list_of_delta_f = []
#                 delta_f = 0
#                 hidden_state = torch.empty((1, 64, H // 8, W // 8))
#                 for GRU_iteration in range(self.GRU_iterations):
#                     predicted_flow = self.curr_flow.detach() + delta_f
#                     context_feature = self.context_encoder.forward(frame1)
#                     motion_feature = self.motion_encoder.forward(feat1, feat2, predicted_flow)
#                     aggr_motion = self.mem_buffer.forward(context_feature, motion_feature)
#                     # Concatenate motion_feature, aggr_motion, context_feature
#                     concat_feat = torch.cat((motion_feature, aggr_motion, context_feature), dim=1)
#                     hidden_state = self.GRU.forward(concat_feat,hidden_state)
#                     delta_f = self.upsampler.forward(hidden_state)
#                     list_of_delta_f.append(delta_f)
#                 loss = self.loss_normal.forward(self.curr_flow, list_of_delta_f, flow_gt)
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#
#                 self.cutoff += 1
#

class PSNRModel(BaseModel):

    def __init__(self, inTrain: bool):

        super(PSNRModel, self).__init__(inTrain)

    def forward(self, frames_0, frames_1):

        feat1 = self.siamese_encoder.forward(frames_0)
        feat2 = self.siamese_encoder.forward(frames_1)

        context_feature = self.context_encoder.forward(frames_0)

        if self.inTrain:
            list_of_delta_f = []

        downsampled_delta = torch.zeros((batch_sz, 2, H // 8, W // 8)).to(device)
        downsampled_delta.requires_grad = True
        hidden_state = torch.zeros((batch_sz, hidden_dim, H // 8, W // 8)).to(device)

        for GRU_iteration in range(GRU_iterations):
            predicted_flow = self.curr_flow.detach() + downsampled_delta

            # pass in batch_sz x 2 x H / 8 x W / 8 as flow
            motion_lookup = self.motion_lookup.forward(feat1, feat2, predicted_flow)

            motion_feature = self.motion_encoder.forward(motion_lookup, predicted_flow)

            aggr_motion = self.mem_buffer.forward(context_feature, motion_feature)

            concat_feat = torch.cat((motion_feature, aggr_motion, context_feature), dim=1)

            hidden_state = self.GRU.forward(concat_feat, hidden_state)

            downsampled_delta = self.flow_predictor.forward(hidden_state)

            delta_f = self.upsampler.forward(downsampled_delta, hidden_state)

            if self.inTrain:
                list_of_delta_f.append(delta_f)

        if self.inTrain:
            return list_of_delta_f

        return delta_f