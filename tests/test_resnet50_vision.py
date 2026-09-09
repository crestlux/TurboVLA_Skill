import numpy as np
import pytest
import torch

from turbovla.models.configuration import InteractionConfig, TurboVLAConfig, VisionEncoderConfig
from turbovla.models.turbovla import VisionLanguageInteraction
from turbovla.models.vision_encoder import ResNet50VisionEncoder
from turbovla.vision_processing import IMAGENET_MEAN, IMAGENET_STD, build_resnet50_processor


def test_legacy_config_defaults_to_dinov3():
    config = TurboVLAConfig.from_mapping({"vision": {"image_size": 256}})
    assert config.vision.backbone_type == "dinov3"


@pytest.mark.parametrize("output_stride, expected_tokens", [(16, 16), (32, 4)])
def test_resnet50_token_shape(output_stride, expected_tokens):
    config = VisionEncoderConfig(
        backbone_type="resnet50",
        resnet_weights="none",
        resnet_output_stride=output_stride,
        load_pretrained=False,
        image_size=64,
        num_views=2,
        compute_precision="fp32",
    )
    encoder = ResNet50VisionEncoder(config).eval()
    with torch.inference_mode():
        output = encoder(torch.randn(1, 2, 3, 64, 64))
    assert output.shape == (1, 2, expected_tokens, 2048)
    assert not any(name.startswith("fc.") for name, _ in encoder.backbone.named_parameters())


def test_resnet50_processor_is_resize_free_imagenet_normalization():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    output = build_resnet50_processor(expected_size=64, output_stride=16)(image)["pixel_values"]
    expected = -torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) / torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    assert output.shape == (1, 3, 64, 64)
    torch.testing.assert_close(output[:, :, :1, :1], expected)


def test_resnet50_processor_rejects_unexpected_resolution():
    processor = build_resnet50_processor(expected_size=256, output_stride=16)
    with pytest.raises(ValueError, match="expects 256x256"):
        processor(np.zeros((224, 224, 3), dtype=np.uint8))


def test_zero_interaction_layers_pass_features_through_unchanged():
    interaction = VisionLanguageInteraction(InteractionConfig(num_layers=0))
    visual = torch.randn(2, 8, 256)
    text = torch.randn(2, 5, 256)
    padding_mask = torch.zeros(2, 5, dtype=torch.bool)

    visual_out, text_out = interaction(visual, text, padding_mask, None)

    assert len(interaction.fusion_layers) == 0
    assert len(interaction.text_layers) == 0
    torch.testing.assert_close(visual_out, visual)
    torch.testing.assert_close(text_out, text)
