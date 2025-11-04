# app.py — cleaned up, robust RGB handling, safer State passing, clearer errors
import os
import tempfile

import imageio
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange
from tqdm import tqdm
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import hf_hub_download

import gradio as gr

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics,
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_glb
from src.utils.infer_util import remove_background, resize_foreground

# ---------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------
if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
else:
    device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device1 = device0

device = device0 if device0.type == "cuda" else torch.device("cpu")

# ---------------------------------------------------------------------
# Model cache dir
# ---------------------------------------------------------------------
model_cache_dir = "./ckpts/"
os.makedirs(model_cache_dir, exist_ok=True)

# ---------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------
def get_render_cameras(batch_size=1, M=120, radius=2.5, elevation=10.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.

    Returns
    -------
    cameras : torch.Tensor
        If is_flexicubes: (B, M, 4, 4) camera-to-world inverses (i.e., world-to-camera).
        Else: (B, M, 24) flattened extrinsics (3x4) + intrinsics (3x3).
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)                   # (M, 4, 4)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)                      # (M, 16)
        intrinsics = (
            FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        )                                                  # (M, 9)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)  # (M, 25)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


# ---------------------------------------------------------------------
# Video writer (keep name used later; avoid shadowing imported names)
# ---------------------------------------------------------------------
def images_to_video(images, output_path, fps=30):
    """
    Write a sequence of CHW images in [0,1] to mp4 (H.264).

    Parameters
    ----------
    images : torch.Tensor
        Shape (N, C, H, W), values in [0,1].
    output_path : str
    fps : int
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"images must be torch.Tensor, got {type(images)}")
    if images.ndim != 4 or images.shape[1] not in (1, 3, 4):
        raise ValueError(f"images must be (N,C,H,W) with C in {{1,3,4}}, got {tuple(images.shape)}")

    frames = []
    for i in range(images.shape[0]):
        # (C,H,W) -> (H,W,C)
        frame = images[i].permute(1, 2, 0).detach().cpu().numpy()
        # drop alpha if present
        if frame.shape[2] == 4:
            frame = frame[..., :3]
        frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
        frames.append(frame)

    imageio.mimwrite(output_path, np.stack(frames, axis=0), fps=fps, codec="h264")


# ---------------------------------------------------------------------
# Seed and config
# ---------------------------------------------------------------------
seed_everything(0)

config_path = "configs/instant-mesh-large.yaml"
config = OmegaConf.load(config_path)
config_name = os.path.basename(config_path).replace(".yaml", "")
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = True if config_name.startswith("instant-mesh") else False

# ---------------------------------------------------------------------
# Load diffusion model (Zero123++)
# ---------------------------------------------------------------------
print("Loading diffusion model ...")
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2",
    custom_pipeline="zero123plus",
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    cache_dir=model_cache_dir,
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing="trailing"
)

# Load custom white-background UNet
unet_ckpt_path = hf_hub_download(
    repo_id="TencentARC/InstantMesh",
    filename="diffusion_pytorch_model.bin",
    repo_type="model",
    cache_dir=model_cache_dir,
)
state_dict = torch.load(unet_ckpt_path, map_location="cpu")
pipeline.unet.load_state_dict(state_dict, strict=True)

pipeline = pipeline.to(device0)

# ---------------------------------------------------------------------
# Load reconstruction model (InstantMesh)
# ---------------------------------------------------------------------
print("Loading reconstruction model ...")
model_ckpt_path = hf_hub_download(
    repo_id="TencentARC/InstantMesh",
    filename="instant_mesh_large.ckpt",
    repo_type="model",
    cache_dir=model_cache_dir,
)
model = instantiate_from_config(model_config)
ckpt = torch.load(model_ckpt_path, map_location="cpu")["state_dict"]
ckpt = {k[14:]: v for k, v in ckpt.items() if k.startswith("lrm_generator.") and "source_camera" not in k}
model.load_state_dict(ckpt, strict=True)

model = model.to(device1)
if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device1, fovy=30.0)
model = model.eval()

print("Loading Finished!")

# ---------------------------------------------------------------------
# Gradio pipeline functions
# ---------------------------------------------------------------------
def _ensure_rgb(pil_img: Image.Image) -> Image.Image:
    """Force an Image to RGB."""
    if not isinstance(pil_img, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(pil_img)}")
    if pil_img.mode != "RGB":
        return pil_img.convert("RGB")
    return pil_img


def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")
    # Normalize to RGB early so downstream never sees RGBA
    if isinstance(input_image, Image.Image) and input_image.mode != "RGB":
        input_image = input_image.convert("RGB")
    return input_image


def preprocess(input_image, do_remove_background):
    """
    Removes background (optional), resizes the foreground, and ensures RGB.
    """
    if input_image is None:
        raise gr.Error("No image to preprocess!")

    rembg_session = rembg.new_session() if do_remove_background else None
    out_img = input_image

    if do_remove_background:
        out_img = remove_background(out_img, rembg_session)  # may be RGBA
        out_img = resize_foreground(out_img, 0.85)

        # Composite over white (white-background UNet)
        if isinstance(out_img, Image.Image) and out_img.mode == "RGBA":
            bg = Image.new("RGB", out_img.size, (255, 255, 255))
            bg.paste(out_img, mask=out_img.split()[-1])
            out_img = bg

    # Ensure RGB regardless
    out_img = _ensure_rgb(out_img)
    return out_img


def generate_mvs(input_image, sample_steps, sample_seed):
    """
    Runs Zero123++ to generate the tiled multi-view image.
    Returns:
      - mv_images_state: the same PIL image (stored in Gradio State)
      - show_image: PIL for display
    """
    # Gradio Number yields float; cast to int
    sample_steps = int(sample_steps)
    sample_seed = int(sample_seed)

    seed_everything(sample_seed)
    input_image = _ensure_rgb(input_image)

    generator = torch.Generator(device=device0).manual_seed(sample_seed)
    result = pipeline(
        input_image,
        num_inference_steps=sample_steps,
        generator=generator,
    )
    z123_image = result.images[0]  # PIL RGB

    show_image = z123_image
    # mv_images_state: store as numpy to make State robust across threads/workers
    mv_images_state = np.array(z123_image, dtype=np.uint8)  # (H, W, 3)

    return mv_images_state, show_image


def _pil_or_numpy_to_rgb_numpy(img):
    """
    Accept PIL or NumPy; return HxWx3 uint8 NumPy array.
    """
    if isinstance(img, Image.Image):
        img = _ensure_rgb(img)
        return np.array(img, dtype=np.uint8)

    if isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected images of shape (H,W,3), got {img.shape}")
        if img.dtype != np.uint8:
            # Assume in [0,1] or general float; clip and convert
            img = np.clip(img, 0, 1) if img.dtype.kind == "f" else img
            img = (img * 255.0).astype(np.uint8) if img.dtype.kind == "f" else img.astype(np.uint8)
        return img

    raise TypeError(f"Unsupported type for images: {type(img)}")

def make3d(images):
    """
    Convert a tiled multi-view image into a 3D mesh and a turntable video.
    Returns:
      - video_fpath, mesh_fpath (obj), mesh_glb_fpath (glb)
    """
    if images is None:
        # Give a clear error that helps the user
        raise gr.Error(
            "No multi-view image found from the previous step. "
            "Please click Generate again (and ensure the multi-views appear) before running 3D."
        )

    # Accept PIL or NumPy from State; ensure HxWx3 uint8
    images_np = _pil_or_numpy_to_rgb_numpy(images)  # (H,W,3) uint8

    # Convert to float CHW in [0,1]
    images_t = torch.from_numpy(images_np).float() / 255.0  # (H,W,3)
    images_t = images_t.permute(2, 0, 1).contiguous()       # (C,H,W)

    # Your original code assumes a 3x2 grid -> 6 views
    # Fail-fast with a clear error if not divisible
    H, W = images_t.shape[1], images_t.shape[2]
    n, m = 3, 2
    if (H % n) != 0 or (W % m) != 0:
        raise ValueError(
            f"Multi-view tile size not divisible by ({n},{m}): got (H,W)=({H},{W}). "
            "This likely means the Zero123++ grid layout changed. Adjust (n,m) to match."
        )

    images_t = rearrange(images_t, "c (n h) (m w) -> (n m) c h w", n=n, m=m)  # (6, C, h, w)

    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device1)
    render_cameras = get_render_cameras(
        batch_size=1, radius=4.5, elevation=20.0, is_flexicubes=IS_FLEXICUBES
    ).to(device1)

    images_t = images_t.unsqueeze(0).to(device1)  # (B=1, 6, C, h, w)
    images_t = v2.functional.resize(images_t, (320, 320), interpolation=3, antialias=True).clamp(0, 1)

    mesh_fpath = tempfile.NamedTemporaryFile(suffix=".obj", delete=False).name
    mesh_basename = os.path.basename(mesh_fpath).split(".")[0]
    mesh_dirname = os.path.dirname(mesh_fpath)
    video_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.mp4")

    with torch.no_grad():
        # get triplane
        planes = model.forward_planes(images_t, input_cameras)

        # get turntable video
        chunk_size = 20 if IS_FLEXICUBES else 1
        render_size = 384

        frames = []
        for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
            if IS_FLEXICUBES:
                out = model.forward_geometry(
                    planes, render_cameras[:, i : i + chunk_size], render_size=render_size
                )["img"]
            else:
                out = model.synthesizer(
                    planes, cameras=render_cameras[:, i : i + chunk_size], render_size=render_size
                )["images_rgb"]
            frames.append(out)
        frames = torch.cat(frames, dim=1)  # (B=1, M, C, H, W)

        images_to_video(frames[0], video_fpath, fps=30)
        print(f"Video saved to {video_fpath}")

    mesh_fpath, mesh_glb_fpath = make_mesh(mesh_fpath, planes)

    return video_fpath, mesh_fpath, mesh_glb_fpath


def make_mesh(mesh_fpath, planes):
    mesh_basename = os.path.basename(mesh_fpath).split(".")[0]
    mesh_dirname = os.path.dirname(mesh_fpath)
    mesh_glb_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.glb")

    with torch.no_grad():
        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=False,
            **infer_config,
        )
        vertices, faces, vertex_colors = mesh_out
        vertices = vertices[:, [1, 2, 0]]  # align axis

        save_glb(vertices, faces, vertex_colors, mesh_glb_fpath)
        save_obj(vertices, faces, vertex_colors, mesh_fpath)
        print(f"Mesh saved to {mesh_fpath}")

    return mesh_fpath, mesh_glb_fpath



_HEADER_ = '''
<h2><b>Official 🤗 Gradio Demo</b></h2><h2><a href='https://github.com/TencentARC/InstantMesh' target='_blank'><b>InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models</b></a></h2>

**InstantMesh** is a feed-forward framework for efficient 3D mesh generation from a single image based on the LRM/Instant3D architecture.

Code: <a href='https://github.com/TencentARC/InstantMesh' target='_blank'>GitHub</a>. Techenical report: <a href='https://arxiv.org/abs/2404.07191' target='_blank'>ArXiv</a>.

❗️❗️❗️**Important Notes:**
- Our demo can export a .obj mesh with vertex colors or a .glb mesh now. If you prefer to export a .obj mesh with a **texture map**, please refer to our <a href='https://github.com/TencentARC/InstantMesh?tab=readme-ov-file#running-with-command-line' target='_blank'>Github Repo</a>.
- The 3D mesh generation results highly depend on the quality of generated multi-view images. Please try a different **seed value** if the result is unsatisfying (Default: 42).
'''

_CITE_ = r"""
If InstantMesh is helpful, please help to ⭐ the <a href='https://github.com/TencentARC/InstantMesh' target='_blank'>Github Repo</a>. Thanks! [![GitHub Stars](https://img.shields.io/github/stars/TencentARC/InstantMesh?style=social)](https://github.com/TencentARC/InstantMesh)
---
📝 **Citation**

If you find our work useful for your research or applications, please cite using this bibtex:
```bibtex
@article{xu2024instantmesh,
  title={InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models},
  author={Xu, Jiale and Cheng, Weihao and Gao, Yiming and Wang, Xintao and Gao, Shenghua and Shan, Ying},
  journal={arXiv preprint arXiv:2404.07191},
  year={2024}
}
```

📋 **License**

Apache-2.0 LICENSE. Please refer to the [LICENSE file](https://huggingface.co/spaces/TencentARC/InstantMesh/blob/main/LICENSE) for details.

📧 **Contact**

If you have any questions, feel free to open a discussion or contact us at <b>bluestyle928@gmail.com</b>.
"""

with gr.Blocks() as demo:
    gr.Markdown(_HEADER_)
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image",
                    image_mode="RGBA",
                    width=256,
                    height=256,
                    type="pil",
                )
                processed_image = gr.Image(
                    label="Processed Image", 
                    image_mode="RGBA", 
                    width=256,
                    height=256,
                    type="pil", 
                    interactive=False
                )
            with gr.Row():
                with gr.Group():
                    do_remove_background = gr.Checkbox(
                        label="Remove Background", value=True
                    )
                    sample_seed = gr.Number(value=42, label="Seed Value", precision=0)

                    sample_steps = gr.Slider(
                        label="Sample Steps",
                        minimum=30,
                        maximum=75,
                        value=75,
                        step=5
                    )

            with gr.Row():
                submit = gr.Button("Generate", elem_id="generate", variant="primary")

            with gr.Row(variant="panel"):
                gr.Examples(
                    examples=[
                        os.path.join("examples", img_name) for img_name in sorted(os.listdir("examples"))
                    ],
                    inputs=[input_image],
                    label="Examples",
                    examples_per_page=20
                )

        with gr.Column():

            with gr.Row():

                with gr.Column():
                    mv_show_images = gr.Image(
                        label="Generated Multi-views",
                        type="pil",
                        interactive=False
                    )

                with gr.Column():
                    output_video = gr.Video(
                        label="video", format="mp4",
                        autoplay=True,
                        interactive=False
                    )

            with gr.Row():
                with gr.Tab("OBJ"):
                    output_model_obj = gr.Model3D(
                        label="Output Model (OBJ Format)",
                        #width=768,
                        interactive=False,
                    )
                    gr.Markdown("Note: Downloaded .obj model will be flipped. Export .glb instead or manually flip it before usage.")
                with gr.Tab("GLB"):
                    output_model_glb = gr.Model3D(
                        label="Output Model (GLB Format)",
                        #width=768,
                        interactive=False,
                    )
                    gr.Markdown("Note: The model shown here has a darker appearance. Download to get correct results.")

            with gr.Row():
                gr.Markdown('''Try a different <b>seed value</b> if the result is unsatisfying (Default: 42).''')

    gr.Markdown(_CITE_)
    mv_images = gr.State()

    submit.click(
        fn=check_input_image,
        inputs=[input_image],
        outputs=[processed_image],
    ).then(
        fn=preprocess,
        inputs=[processed_image, do_remove_background],
        outputs=[processed_image],
    ).then(
        fn=generate_mvs,
        inputs=[processed_image, sample_steps, sample_seed],
        outputs=[mv_images, mv_show_images],
    ).then(
        fn=make3d,
        inputs=[mv_images],
        outputs=[output_video, output_model_obj, output_model_glb],
    )

demo.queue(max_size=10)
demo.launch(server_name="0.0.0.0", server_port=43839)
