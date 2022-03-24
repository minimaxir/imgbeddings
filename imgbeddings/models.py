# This file is not referenced by the main imgbeddings.py
# as it serves as a record of the models/techniques used
# to create the models for generating embeddings, and can be
# used to create a variant if needed.

import torch
from transformers.modeling_utils import PreTrainedModel
from pathlib import Path
from transformers.onnx import export
from transformers import AutoFeatureExtractor, CLIPVisionModel, CLIPVisionConfig
from typing import Mapping, OrderedDict
from transformers.onnx import OnnxConfig


def export_clip_vision_to_onnx(patch_size, opset):
    """Exports the specified CLIPVision model to ONNX"""

    model_ckpt = f"openai/clip-vit-base-patch{patch_size}"
    config = CLIPVisionConfig.from_pretrained(model_ckpt)
    config.output_attentions = True
    config.output_hidden_states = True

    processor = AutoFeatureExtractor.from_pretrained(model_ckpt, return_tensors="np")
    base_model = CLIPVisionModel.from_pretrained(model_ckpt, config=config)

    class ExportModel(PreTrainedModel):
        """A wrapper class around CLIPVisionModel to export to ONNX correctly."""

        def __init__(self):
            super().__init__(config)
            self.model = base_model

        # to work with export(), forward() must have the same signature.
        def forward(
            self,
            pixel_values=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):

            out = self.model(pixel_values)
            return {
                "last_hidden_state": out["last_hidden_state"],
                "pooler_output": out["pooler_output"],
                "hidden_states": torch.transpose(
                    torch.stack(list(out["hidden_states"])), 1, 0
                ),
                "attentions": torch.transpose(
                    torch.stack(list(out["attentions"])), 1, 0
                ),
            }

        def call(
            self,
            pixel_values=None,
        ):
            self.forward(pixel_values)

    class CLIPVisionOnnxConfig(OnnxConfig):
        @property
        def inputs(self) -> Mapping[str, Mapping[int, str]]:
            return OrderedDict(
                [
                    ("pixel_values", {0: "batch"}),
                ]
            )

        @property
        def outputs(self) -> Mapping[str, Mapping[int, str]]:
            return OrderedDict(
                [
                    ("last_hidden_state", {0: "batch"}),
                    ("pooler_output", {0: "batch"}),
                    ("hidden_states", {0: "batch"}),
                    ("attentions", {0: "batch"}),
                ]
            )

    new_model = ExportModel()
    onnx_config = CLIPVisionOnnxConfig(config)
    onnx_path = Path("/Users/maxwoolf/Desktop/imgbeddings-test/model.onnx")

    onnx_inputs, onnx_outputs = export(
        processor, new_model, onnx_config, opset, onnx_path
    )

    return
