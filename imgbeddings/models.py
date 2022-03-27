# This file is not referenced by the main imgbeddings.py
# as it serves as a record of the models/techniques used
# to create the models for generating embeddings, and can be
# used to create a variant if needed.

import os
from typing import Mapping, OrderedDict
from pathlib import Path

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.onnx import export
from transformers import AutoFeatureExtractor, CLIPVisionModel, CLIPVisionConfig
from transformers.onnx import OnnxConfig
from onnxruntime.quantization import quantize_dynamic, QuantType


def export_clip_vision_to_onnx(
    output_folder,
    output_name="model.quant.onnx",
    patch_size=32,
    opset=15,
    num_layers=3,
    remove_nonquantized=True,
):
    """Exports the specified CLIPVision model to ONNX"""

    if patch_size in [16, 32]:
        model_ckpt = f"openai/clip-vit-base-patch{patch_size}"
    elif patch_size == 14:
        model_ckpt = "openai/clip-vit-large-patch14"
    else:
        AssertionError
    config = CLIPVisionConfig.from_pretrained(model_ckpt)
    config.output_attentions = True
    config.output_hidden_states = True

    processor = AutoFeatureExtractor.from_pretrained(model_ckpt, return_tensors="np")
    base_model = CLIPVisionModel.from_pretrained(model_ckpt, config=config)
    # base_model = AutoModel.from_pretrained(model_ckpt, config=config)
    base_model = base_model.eval()

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
            embeddings = get_embeddings_from_output(out, num_layers)
            return {"embeddings": embeddings}
            # return {"embeddings": out["last_hidden_state"].mean(1)}

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
                    ("embeddings", {0: "batch"}),
                ]
            )

    new_model = ExportModel()
    onnx_config = CLIPVisionOnnxConfig(config)
    onnx_path = os.path.join(output_folder, "model.onnx")

    _, _ = export(processor, new_model, onnx_config, opset, Path(onnx_path))

    onnx_quant_path = os.path.join(output_folder, output_name)
    quantize_dynamic(onnx_path, onnx_quant_path, weight_type=QuantType.QUInt8)

    if remove_nonquantized:
        os.remove(onnx_path)
        os.remove(os.path.join(output_folder, "model-opt.onnx"))

    return


def get_embeddings_from_output(outputs, num_layers):

    hidden_states = torch.sum(
        torch.stack([outputs.hidden_states[i] for i in range(-num_layers, 0)]), 0
    )

    attentions = torch.stack([outputs.attentions[i] for i in range(-num_layers, 0)])
    # switch dimensions so batch dimension is first
    attentions = torch.transpose(attentions, 1, 0)

    attentions_reduced = torch.mean(attentions, (1, 2, 3))
    attentions_reweighted = attentions_reduced

    # the first value corresponds to the class token which is irrelevant
    attentions_reweighted[:, 0] = 0.0
    attentions_reweighted = attentions_reweighted / torch.unsqueeze(
        torch.sum(attentions_reweighted, 1), 1
    )

    embeddings = hidden_states * attentions_reweighted.unsqueeze(2)
    return embeddings.sum(1)
