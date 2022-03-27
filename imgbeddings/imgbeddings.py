import logging
from dataclasses import dataclass, field
import random
import itertools

from transformers import CLIPProcessor
from huggingface_hub import hf_hub_url, cached_download
from onnxruntime import InferenceSession
import numpy as np
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from PIL import Image

from .utils import square_pad, create_session_for_provider, symmetric_img_aug

logger = logging.getLogger("imgbeddings")
logger.setLevel(logging.INFO)


@dataclass
class imgbeddings:
    model_path: str = None
    patch_size: int = 32
    version: int = 1
    pca: PCA = None
    gpu: bool = False
    provider: str = None
    processor: CLIPProcessor = field(init=False)
    session: InferenceSession = field(init=False)

    def __post_init__(self):
        patch_values = [14, 16, 32]
        assert self.patch_size in patch_values, f"patch_size must be in {patch_values}."

        if self.model_path is None:
            model_filename = f"patch{self.patch_size}_v{self.version}.onnx"

            config_file_url = hf_hub_url(
                repo_id="minimaxir/imgbeddings", filename=model_filename
            )
            self.model_path = cached_download(
                config_file_url, force_filename=model_filename
            )

        self.session = create_session_for_provider(
            self.model_path, self.provider, self.gpu
        )

        self.processor = CLIPProcessor.from_pretrained(
            f"openai/clip-vit-base-patch{self.patch_size}"
        )
        # for embeddings consistancy, do not center crop
        self.processor.feature_extractor.do_center_crop = False

        # reload PCA if a saved PCA path is provided
        if self.pca:
            pca_path = self.pca
            with np.load(pca_path) as saved_pca:
                self.pca = PCA()
                self.pca.mean_ = saved_pca["mean"]
                self.pca.components_ = saved_pca["components"]
            logging.info(
                f"PCA loaded from {pca_path} and "
                + f"will transform imgbeddings to {self.pca.mean_.shape[0]}D."
            )

    def to_embeddings(self, inputs, batch_size=64, pca_transform=True):
        if not isinstance(inputs, list):
            inputs = [inputs]

        # if doing a small batch, run as normal, else need to run iteratively
        if len(inputs) < batch_size:
            image_inputs = self.process_inputs(inputs)
            embeddings = self.create_embeddings(image_inputs)
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
                embeddings.append(self.create_embeddings(image_inputs))
                pbar.update(len(input_batch))

            pbar.close()
            embeddings = np.vstack(embeddings)
        if self.pca and pca_transform:
            embeddings = self.pca_transform(embeddings)
        return embeddings

    def process_inputs(self, inputs):
        inputs = [square_pad(self.to_pil(x).convert("RGB")) for x in inputs]
        image_inputs = self.processor(images=inputs, return_tensors="np")
        return image_inputs

    def create_embeddings(self, inputs):
        return self.session.run(["embeddings"], dict(inputs))[0]

    def to_pil(self, input):
        # if a filepath is provided, load as a PIL Image
        if isinstance(input, str):
            input = Image.open(input)
        return input

    def augment_images(self, inputs, multiples=1, seed=-1, **kwargs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        if seed:
            random.seed(seed)
        aug_inputs = []
        for _ in tqdm(range(multiples)):
            new_inputs = [symmetric_img_aug(self.to_pil(x), **kwargs) for x in inputs]
            aug_inputs.append(new_inputs)

        # flatten the list
        aug_inputs = list(itertools.chain(*aug_inputs))
        random.shuffle(aug_inputs)
        return aug_inputs

    def pca_fit(self, embeddings, out_dim=128, save_path="pca.npz"):
        self.pca = PCA(n_components=out_dim)
        self.pca.fit(embeddings)

        if save_path:
            np.savez_compressed(
                save_path, mean=self.pca.mean_, components=self.pca.components_
            )
            logging.info(
                f'PCA saved at "{save_path}"; '
                + "provide this as `pca` to imgbeddings() to reload."
            )
        explained_var = np.sum(self.pca.explained_variance_ratio_)
        logging.info(f"The fitted PCA explains {explained_var:.1%} of the variance.")
        return

    def pca_transform(self, inputs):
        assert self.pca, "You need to fit/load a PCA first."
        return self.pca.transform(inputs)
