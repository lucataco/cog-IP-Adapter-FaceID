# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import cv2
import torch
from PIL import Image
from typing import List
from insightface.app import FaceAnalysis
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL

base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
base_cache = "model-cache"
vae_model_path = "stabilityai/sd-vae-ft-mse"
ip_cache = "./ip-cache"
device = "cuda"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Get ip-adapter-faceid model
        if not os.path.exists("ip-cache/ip-adapter-faceid_sd15.bin"):
            os.makedirs(ip_cache)
            os.system(f"wget -O ip-cache/ip-adapter-faceid_sd15.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin")
        # Face embedding
        self.app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        # SD
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        vae = AutoencoderKL.from_pretrained(
            vae_model_path
        ).to(dtype=torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None,
            cache_dir=base_cache,
        )
        self.pipe = pipe.to(device)
        # IP adapter
        self.ip_model = IPAdapterFaceID(
            pipe,
            "ip-cache/ip-adapter-faceid_sd15.bin",
            device
        )

    @torch.inference_mode()
    def predict(
        self,
        face_image: Path = Input(description="Input face image"),
        prompt: str = Input(
            description="Input prompt",
            default="photo of a woman in red dress in a garden",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="monochrome, lowres, bad anatomy, worst quality, low quality, blurry, multiple people",
        ),
        width: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        num_outputs: int = Input(
            description="Number of images to output",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=200, default=30
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        agree_to_research_only: bool = Input(
            description="You must agree to use this model only for research. It is not for commercial use.",
            default=False,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if not agree_to_research_only:
            raise Exception(
                "You must agree to use this model for research-only, you cannot use this model comercially."
            )
        
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")

        # Read image
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        image = cv2.imread(str(face_image))
        faces = self.app.get(image)

        faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

        images = self.ip_model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            faceid_embeds=faceid_embeds,
            num_samples=num_outputs,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            seed=seed
        )

        output_paths = []
        for i, image in enumerate(images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths