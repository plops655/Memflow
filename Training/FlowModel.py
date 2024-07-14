import torch
from Training.BaseModel import BaseModel

import sys

class FlowModel(BaseModel):

    def __init__(self, r = 2, N = 10, L = 2, dataset_str = None, cutoff=sys.maxsize, epochs=20, norm_fn='group', memory_buffer_length=2,
                 optimizer: str = "Adadelta", dropout=0.1, GRU_iterations=15,
                 loss_cascade_ratio: int = 0.85):

        super(FlowModel, self).__init__(r, N, L, dataset_str, cutoff, epochs, norm_fn, memory_buffer_length,
                                        optimizer, dropout, GRU_iterations, loss_cascade_ratio)

    def forward_loss_normal(self):
        H = self.H
        W = self.W
        for epoch in range(self.epochs):
            frame_pairs = self.slide_2_frames()
            pair = next(frame_pairs)
            flow_gt = pass # GET FROM DATASET!!!

            curr_frame_idx = 0
            while pair and curr_frame_idx < self.cutoff:

                frame1, frame2 = pair
                feat1 = self.siamese_encoder.forward(frame1)
                feat2 = self.siamese_encoder.forward(frame2)
                list_of_delta_f = []
                delta_f = 0
                hidden_state = torch.empty((1, 64, H // 8, W // 8))
                for GRU_iteration in range(self.GRU_iterations):
                    predicted_flow = self.curr_flow.detach() + delta_f
                    context_feature = self.context_encoder.forward(frame1)
                    motion_feature = self.motion_encoder.forward(feat1, feat2, predicted_flow)
                    aggr_motion = self.mem_buffer.forward(context_feature, motion_feature)
                    # Concatenate motion_feature, aggr_motion, context_feature
                    concat_feat = torch.cat((motion_feature, aggr_motion, context_feature), dim=1)
                    hidden_state = self.GRU.forward(concat_feat,hidden_state)
                    delta_f = self.upsampler.forward(hidden_state)
                    list_of_delta_f.append(delta_f)
                loss = self.loss_normal.forward(self.curr_flow, list_of_delta_f, flow_gt)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.cutoff += 1

