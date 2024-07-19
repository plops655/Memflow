import torch
import sys

from Models.BaseModel import BaseModel
import utils.consts as consts

class PSNRModel(BaseModel):

    def __init__(self, dataset, epochs, optimizer):

        super(PSNRModel, self).__init__(dataset, epochs, optimizer)


    def forward_loss_psnr(self):


        for epoch in range(self.epochs):

            frame_pairs = self.slide_2_frames()
            pair = next(frame_pairs)

            while pair:

                frame1, frame2 = pair
                feat1 = self.siamese_encoder.forward(frame1)
                feat2 = self.siamese_encoder.forward(frame2)
                list_of_delta_f = []
                delta_f = torch.zeros((2, consts.H, consts.W))
                hidden_state = torch.empty((1, 64, consts.H // 8, consts.W // 8))

                for GRU_iteration in range(consts.GRU_iterations):

                    predicted_flow = self.curr_flow.detach() + delta_f

                    context_feature = self.context_encoder.forward(frame1)

                    motion_lookup = self.motion_lookup.forward(feat1, feat2, predicted_flow)

                    motion_feature = self.motion_encoder.forward(motion_lookup, predicted_flow)

                    aggr_motion = self.mem_buffer.forward(context_feature, motion_feature)

                    concat_feat = torch.cat((motion_feature, aggr_motion, context_feature), dim=1)

                    hidden_state = self.GRU.forward(concat_feat, hidden_state)
                    delta_f = self.upsampler.forward(hidden_state)

                    list_of_delta_f.append(delta_f)

                loss = self.loss_psnr.forward(list_of_delta_f, frame1, frame2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
