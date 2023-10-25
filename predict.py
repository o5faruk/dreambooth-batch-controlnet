# flake8: noqa: E501
import time
import os
from typing import List
import json

from tqdm.auto import tqdm
import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    ControlNetModel,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import CLIPFeatureExtractor
import shutil
import subprocess
from diffusers.utils import load_image
from stable_diffusion_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
from compel import Compel

SAFETY_MODEL_CACHE = "diffusers-cache"
SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"

DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_SCHEDULER = "DDIM"
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_STRENGTH = 0.8

# grab instance_prompt from weights,
# unless empty string or not existent

DEFAULT_PROMPT = "a photo of an astronaut riding a horse on mars"


class KerrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


class DPMPPSDEKarras:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(
            config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
        )


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KerrasDPM": KerrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "KLMS": LMSDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "UniPCMultistep": UniPCMultistepScheduler,
    "DPM++SDEKarras": DPMPPSDEKarras,
}


class Predictor(BasePredictor):
    def setup(self):
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_MODEL_ID,
            cache_dir=SAFETY_MODEL_CACHE,
            torch_dtype=torch.float16,
            local_files_only=False,
        ).to("cuda")
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir=SAFETY_MODEL_CACHE
        )
        self.url = None

    def download_tar_weights(self, url):
        """Download the model weights from the given URL"""
        print("Downloading weights...")

        if os.path.exists("weights"):
            shutil.rmtree("weights")
        os.makedirs("weights")
        subprocess.check_output(
            ["script/get_weights.sh", url], stderr=subprocess.STDOUT
        )

    def download_zip_weights_python(self, url):
        """Download the model weights from the given URL"""
        print("Downloading weights...")

        if os.path.exists("weights"):
            shutil.rmtree("weights")
        os.makedirs("weights")

        import zipfile
        from io import BytesIO
        import urllib.request

        url = urllib.request.urlopen(url)
        with zipfile.ZipFile(BytesIO(url.read())) as zf:
            zf.extractall("weights")

    def load_weights(self, url):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading Safety pipeline...")

        if url == self.url:
            return

        start_time = time.time()
        self.download_zip_weights_python(url)
        print("Downloaded weights in {:.2f} seconds".format(time.time() - start_time))

        start_time = time.time()
        print("Loading SD pipeline...")
        self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            "weights",
            safety_checker=self.safety_checker,
            feature_extractor=self.feature_extractor,
            torch_dtype=torch.float16,
        ).to("cuda")

        print("Loading SD img2img pipeline...")
        self.img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=self.txt2img_pipe.feature_extractor,
        ).to("cuda")

        print("Loading controlnet...")
        controlnet = ControlNetModel.from_pretrained(
            "thibaud/controlnet-sd21-openposev2-diffusers",
            torch_dtype=torch.float16,
            cache_dir="diffusers-cache",
            local_files_only=False,
        )

        print("Loading controlnet txt2img...")
        self.cnet_txt2img_pose_pipe = StableDiffusionControlNetPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=self.txt2img_pipe.feature_extractor,
            controlnet=controlnet,
        ).to("cuda")

        print("Loading controlnet img2img...")
        self.cnet_img2img_pose_pipe = StableDiffusionControlNetImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=self.txt2img_pipe.feature_extractor,
            controlnet=controlnet,
        ).to("cuda")

        print("Loaded pipelines in {:.2f} seconds".format(time.time() - start_time))

        self.txt2img_pipe.set_progress_bar_config(disable=True)
        self.img2img_pipe.set_progress_bar_config(disable=True)
        self.cnet_txt2img_pose_pipe.set_progress_bar_config(disable=True)
        self.url = url

    def generate_images(self, images, output_dir):
        with torch.autocast("cuda"), torch.inference_mode():
            for info in tqdm(images, desc="Generating samples"):
                inputs = info.get("input") or info.get("inputs")
                name = info["name"]
                print(name)

                num_outputs = int(inputs.get("num_outputs", 1))

                kwargs = {
                    "prompt": [inputs["prompt"]] * num_outputs,
                    "num_inference_steps": int(
                        inputs.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS)
                    ),
                    "guidance_scale": float(
                        inputs.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)
                    ),
                }
                print("GETTING IMAGES")
                image = inputs.get("image")
                pose_image = inputs.get("pose_image")
                if image is not None and pose_image is not None:
                    print("USING controlnet_img2img_pose_pipe")
                    kwargs["controlnet_conditioning_image"] = load_image(pose_image)
                    kwargs["image"] = load_image(image)
                    kwargs["strength"] = float(inputs.get("strength", DEFAULT_STRENGTH))
                    kwargs["width"] = int(inputs.get("width", DEFAULT_WIDTH))
                    kwargs["height"] = int(inputs.get("height", DEFAULT_HEIGHT))
                    pipeline = self.cnet_img2img_pose_pipe
                elif pose_image is not None:
                    print("USING controlnet_text2img_pose_pipe ")
                    kwargs["image"] = load_image(pose_image)
                    kwargs["width"] = int(inputs.get("width", DEFAULT_WIDTH))
                    kwargs["height"] = int(inputs.get("height", DEFAULT_HEIGHT))
                    pipeline = self.cnet_txt2img_pose_pipe
                    print("DONE GETTING POSE")
                elif image is not None:
                    kwargs["image"] = load_image(image)
                    kwargs["strength"] = float(inputs.get("strength", DEFAULT_STRENGTH))
                    pipeline = self.img2img_pipe
                else:
                    pipeline = self.txt2img_pipe
                    kwargs["width"] = int(inputs.get("width", DEFAULT_WIDTH))
                    kwargs["height"] = int(inputs.get("height", DEFAULT_HEIGHT))

                prompt = inputs.get("prompt")
                negative_prompt = inputs.get("negative_prompt")
                compel_proc = Compel(
                    tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder
                )
                kwargs["prompt_embeds"] = compel_proc(prompt)
                kwargs["negative_prompt_embeds"] = compel_proc(negative_prompt)
                # Remove prompt and negative prompt from kwargs
                kwargs.pop("prompt", None)
                kwargs.pop("negative_prompt", None)

                # if negative_prompt is not None:
                #     kwargs["negative_prompt"] = [negative_prompt] * num_outputs

                scheduler = inputs.get("scheduler", DEFAULT_SCHEDULER)
                pipeline.scheduler = SCHEDULERS[scheduler].from_config(
                    pipeline.scheduler.config
                )

                if bool(inputs.get("disable_safety_check", False)):
                    pipeline.safety_checker = None
                else:
                    pipeline.safety_checker = self.safety_checker

                seed = int(inputs.get("seed", int.from_bytes(os.urandom(2), "big")))
                generator = torch.Generator("cuda").manual_seed(seed)
                output = pipeline(
                    generator=generator,
                    **kwargs,
                )

                for i, image in enumerate(output.images):
                    if output.nsfw_content_detected and output.nsfw_content_detected[i]:
                        print("skipping nsfw detected for", inputs)
                        continue
                    image.save(os.path.join(output_dir, f"{name}-{i}.png"))

    @torch.inference_mode()
    def predict(
        self,
        images: str = Input(
            description="JSON input",
        ),
        weights: str = Input(
            description="URL to weights",
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""

        weights = weights.replace(
            "https://replicate.delivery/pbxt/",
            "https://storage.googleapis.com/replicate-files/",
        )

        images_json = json.loads(images)

        if weights is None:
            raise ValueError("No weights provided")
        self.load_weights(weights)

        cog_generated_images = "cog_generated_images"
        if os.path.exists(cog_generated_images):
            shutil.rmtree(cog_generated_images)
        os.makedirs(cog_generated_images)

        self.generate_images(images_json, cog_generated_images)

        directory = Path(cog_generated_images)

        results = []
        for file_path in directory.rglob("*"):
            print(file_path)
            results.append(file_path)
        return results
