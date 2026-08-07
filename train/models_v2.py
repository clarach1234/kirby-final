import torch
import torch.nn as nn
import torch.nn.functional as F

from models import (
    ChannelAttentionConv,
    ConvBlock,
    NewUNetModule,
    UNetModule,
)


class NewUNetModuleV2(NewUNetModule):
    """Low-memory CMRRWNet encoder-decoder with the missing HxW skip restored."""

    def __init__(self, input_ch, output_ch, base_ch):
        super().__init__(input_ch, output_ch, base_ch)

        # The baseline shares all FFA encoder weights. These tiny adapters let
        # arterial-phase and arteriovenous-phase intensities calibrate separately.
        self.a_adapter = nn.Conv2d(1, 1, 1, bias=True)
        self.av_adapter = nn.Conv2d(1, 1, 1, bias=True)
        nn.init.ones_(self.a_adapter.weight)
        nn.init.zeros_(self.a_adapter.bias)
        nn.init.ones_(self.av_adapter.weight)
        nn.init.zeros_(self.av_adapter.bias)

        # Additive projections avoid a large 3*base_ch full-resolution concat.
        self.full_rgb = nn.Conv2d(base_ch, base_ch, 1, bias=False)
        self.full_a = nn.Conv2d(base_ch, base_ch, 1, bias=False)
        self.full_av = nn.Conv2d(base_ch, base_ch, 1, bias=False)
        self.full_attention = ChannelAttentionConv(base_ch)

    def forward(self, x):
        rgb = x[:, 0:3]
        a = self.a_adapter(x[:, 3:4])
        av = self.av_adapter(x[:, 4:5])

        feats_rgb = self.forward_encoder_rgb(rgb)
        feats_a = self.forward_encoder_a(a, feats_rgb)
        feats_av = self.forward_encoder_a(av, feats_rgb)

        x1, x2, x3, x4 = self.forward_features(
            feats_rgb, feats_a, feats_av
        )
        full_skip = self.full_attention(
            self.full_rgb(feats_rgb[0])
            + self.full_a(feats_a[0])
            + self.full_av(feats_av[0])
        )

        decoded = self.upconv1(x4)
        decoded = self.conv6(torch.cat((x3, decoded), dim=1))
        decoded = self.upconv2(decoded)
        decoded = self.conv7(torch.cat((x2, decoded), dim=1))
        decoded = self.upconv3(decoded)
        decoded = self.conv8(torch.cat((x1, decoded), dim=1))
        decoded = self.upconv4(decoded)

        # This skip and conv9 exist in the baseline class but were commented out.
        decoded = self.conv9(torch.cat((full_skip, decoded), dim=1))
        return self.outconv(decoded)


class CMRRWNetV2(nn.Module):
    """CMRRWNet with [artery, vessel, vein] semantics throughout refinement."""

    def __init__(self, input_ch, output_ch, base_ch, num_iterations=1):
        super().__init__()
        if output_ch != 3:
            raise ValueError('CMRRWNetV2 requires three outputs: [A, vessel, V]')
        self.first_u = NewUNetModuleV2(input_ch, output_ch, base_ch)
        self.second_u = UNetModule(output_ch, 2, base_ch)
        self.num_iterations = num_iterations

    @staticmethod
    def _compose(av_logits, vessel_logits):
        return torch.cat(
            (av_logits[:, 0:1], vessel_logits, av_logits[:, 1:2]), dim=1
        )

    def forward(self, x):
        predictions = []

        first_logits = self.first_u(x)
        predictions.append(first_logits)

        vessel_logits = first_logits[:, 1:2]
        vessel_probability = torch.sigmoid(vessel_logits)
        state = torch.sigmoid(first_logits)

        av_logits = self.second_u(state)
        predictions.append(self._compose(av_logits, vessel_logits))

        for _ in range(self.num_iterations):
            state = torch.cat(
                (
                    torch.sigmoid(av_logits[:, 0:1]),
                    vessel_probability,
                    torch.sigmoid(av_logits[:, 1:2]),
                ),
                dim=1,
            )
            av_logits = self.second_u(state)
            predictions.append(self._compose(av_logits, vessel_logits))

        return predictions
