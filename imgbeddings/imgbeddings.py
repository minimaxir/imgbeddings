import logging
from dataclasses import dataclass, field

from transformers import CLIPProcessor
from onnxruntime import InferenceSession
import numpy as np
from tqdm.auto import tqdm
from .utils import square_pad, create_session_for_provider

logger = logging.getLogger("imgbeddings")
logger.setLevel(logging.INFO)


@dataclass
class imgbeddings:
    model_path: str
    patch_size: int = 32
    processor: CLIPProcessor = field(init=False)
    session: InferenceSession = field(init=False)

    def __post_init__(self):
        patch_values = [14, 16, 32]
        assert self.patch_size in patch_values, f"patch_size must be in {patch_values}."

        self.session = create_session_for_provider(self.model_path)

        self.processor = CLIPProcessor.from_pretrained(
            f"openai/clip-vit-base-patch{self.patch_size}"
        )
        # for embeddings consistancy, do not center crop
        self.processor.feature_extractor.do_center_crop = False

    def to_embeddings(self, inputs, batch_size=64, return_format="np"):
        if not isinstance(inputs, list):
            inputs = [inputs]

        # if doing a small batch, run as normal, else need to run iteratively
        if len(inputs) < batch_size:
            image_inputs = self.process_inputs(inputs)
            embeddings = self.create_embeddings(image_inputs)
            return embeddings
        else:
            logging.info(f"Creating image embeddings in batches of {batch_size}.")

            # https://stackoverflow.com/a/8290508
            def batch(iterable, n=1):
                length = len(iterable)
                for ndx in range(0, length, n):
                    yield iterable[ndx : min(ndx + n, length)]

            embeddings = []
            pbar = tqdm(total=len(inputs), smoothing=0)
            for input_batch in batch(inputs, batch_size):
                image_inputs = self.process_inputs(input_batch)
                embeddings = self.create_embeddings(image_inputs)
                pbar.update(batch_size)

            pbar.close()
            embeddings = np.vstack(embeddings)
            return embeddings

    def process_inputs(self, inputs):
        inputs = [square_pad(x) for x in inputs]

        image_inputs = self.processor(images=inputs, return_tensors="np")
        return image_inputs

    def create_embeddings(self, inputs):
        return self.session.run(["embeddings"], dict(inputs))[0]
