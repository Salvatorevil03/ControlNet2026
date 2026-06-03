from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from transformers import CLIPTokenizer

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import ldm.modules.attention as attention_mod

# Inizializza il tokenizer di CLIP
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_scribble.pth', location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

def create_heatmap(attn_map, original_shape=(512, 512)):
    W, H = original_shape
    grid_w = W // 32
    grid_h = H // 32
    attn_map = attn_map.float()
    attn_map = attn_map - attn_map.min()
    attn_map = attn_map / (attn_map.max() + 1e-8)
    grid = attn_map.reshape(grid_h, grid_w).numpy()
    grid_resized = cv2.resize(grid, (W, H), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(np.uint8(255 * grid_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap

def process_attention_maps(prompt, target_word, num_steps, image_shape=(512, 512)):
    if len(attention_mod.STORE_ATTN) == 0: return [], []
    tokens = tokenizer.encode(prompt)
    words = tokenizer.convert_ids_to_tokens(tokens)
    all_maps = torch.stack(attention_mod.STORE_ATTN) 
    global_avg_map = all_maps.mean(dim=(0, 1))
    
    top_row_images = []
    for i, word in enumerate(words):
        if word not in ["<|startoftext|>", "<|endoftext|>"]:
            clean_word = word.replace("</w>", "")
            heatmap = create_heatmap(global_avg_map[:, i], image_shape)
            top_row_images.append((heatmap, clean_word))
            
    bottom_row_images = []
    target_idx = None
    for i, word in enumerate(words):
        if target_word.lower() in word.lower():
            target_idx = i; break
            
    if target_idx is not None:
        layers_per_step = len(attention_mod.STORE_ATTN) // num_steps
        step_indices = np.linspace(0, num_steps - 1, 7, dtype=int)
        for step in step_indices:
            start_idx = step * layers_per_step
            end_idx = start_idx + layers_per_step
            step_maps = all_maps[start_idx:end_idx]
            step_avg = step_maps.mean(dim=(0, 1))
            heatmap = create_heatmap(step_avg[:, target_idx], image_shape)
            bottom_row_images.append((heatmap, f"t={num_steps - step}"))
            
    return top_row_images, bottom_row_images

def process(input_dict, prompt, a_prompt, n_prompt, target_word, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    attention_mod.STORE_ATTN.clear()
    with torch.no_grad():
        # Gestione input da ImageEditor
        if isinstance(input_dict, dict) and 'composite' in input_dict:
            input_image = input_dict['composite']
        else:
            input_image = input_dict
            
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape
        attention_mod.EXPECTED_SHAPE = (H // 32) * (W // 32)

        detected_map = np.zeros_like(img, dtype=np.uint8)
        detected_map[np.min(img, axis=2) < 127] = 255

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1: seed = random.randint(0, 65535)
        seed_everything(seed)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)
        
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples, shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale, unconditional_conditioning=un_cond, log_every_t=1)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        results = [x_samples[i] for i in range(num_samples)]
        
        inter_images = [cv2.resize((einops.rearrange(model.decode_first_stage(s[0:1]), 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)[0], (W, H)) for s in intermediates['pred_x0'][1:]]
        inter_images_noise = [cv2.resize((einops.rearrange(model.decode_first_stage(s[0:1]), 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)[0], (W, H)) for s in intermediates['x_inter'][1:]]

    print(f"Ho catturato {len(attention_mod.STORE_ATTN)} mappe di attenzione!")
    top_row_maps, bottom_row_maps = process_attention_maps(prompt, target_word, ddim_steps, (W, H))
    return [255 - detected_map] + results, inter_images, inter_images_noise, top_row_maps, bottom_row_maps

def create_canvas(w, h): return np.zeros(shape=(h, w, 3), dtype=np.uint8) + 255

block = gr.Blocks().queue()
with block:
    with gr.Row(): gr.Markdown("## Control Stable Diffusion with Interactive Scribbles & Attention Viz")
    with gr.Row():
        with gr.Column():
            canvas_width = gr.Slider(label="Canvas Width", minimum=256, maximum=1024, value=512, step=1)
            canvas_height = gr.Slider(label="Canvas Height", minimum=256, maximum=1024, value=512, step=1)
            create_button = gr.Button(value='Open drawing canvas!')
            input_image = gr.ImageEditor(type='numpy', sources=['upload'], width=512, height=512)
            prompt = gr.Textbox(label="Prompt")
            target_word = gr.Textbox(label="Word to track (e.g. 'cat')", value="") 
            run_button = gr.Button(value="Run")
            create_button.click(fn=create_canvas, inputs=[canvas_width, canvas_height], outputs=[input_image])
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt", value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, columns=2)
            gr.Markdown("### Figure 4 (Top): Average Attention Maps")
            fig4_top = gr.Gallery(label='Average Maps', show_label=True, columns=5)
            gr.Markdown("### Figure 4 (Bottom): Attention Map Evolution")
            fig4_bot = gr.Gallery(label='Timeline Maps', show_label=True, columns=7)
            with gr.Accordion("Intermediate Preview", open=False):
                intermediate_gallery = gr.Gallery(label='Sample x0', show_label=True, columns=4)
                denoised_Step = gr.Gallery(label='Denoising Step', show_label=True, columns=4)

    ips = [input_image, prompt, a_prompt, n_prompt, target_word, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery, intermediate_gallery, denoised_Step, fig4_top, fig4_bot])

block.launch(server_name='0.0.0.0', share=True)