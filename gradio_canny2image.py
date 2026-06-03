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
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import ldm.modules.attention as attention_mod
# Inizializza il tokenizer di CLIP per mappare le parole ai token
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

apply_canny = CannyDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'), strict=False) 
model = model.cuda()
ddim_sampler = DDIMSampler(model)

def create_heatmap(attn_map, original_shape=(512, 512)):
    """Trasforma un tensore piatto in una heatmap colorata OpenCV"""
    W, H = original_shape
    
    # Calcoliamo dinamicamente la griglia (l'immagine è ridotta di 32 volte a questo layer)
    grid_w = W // 32
    grid_h = H // 32
    
    attn_map = attn_map.float()

    # Normalizza tra 0 e 1
    attn_map = attn_map - attn_map.min()
    attn_map = attn_map / (attn_map.max() + 1e-8)
    
    # Rimodella alla griglia corretta (es. 16x16 per quadrate, 16x24 per rettangolari)
    grid = attn_map.reshape(grid_h, grid_w).numpy()
    
    # Upscale all'immagine originale e applica mappa di calore
    grid_resized = cv2.resize(grid, (W, H), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(np.uint8(255 * grid_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap

def process_attention_maps(prompt, target_word, num_steps, image_shape=(512, 512)):
    """Elabora STORE_ATTN per replicare la Figura 4 del paper"""
    if len(attention_mod.STORE_ATTN) == 0:
        return [], []

    # 1. Tokenizza il prompt per avere parole e indici
    tokens = tokenizer.encode(prompt)
    words = tokenizer.convert_ids_to_tokens(tokens)
    
    # Raggruppa tutte le mappe salvate. Forma attesa di ogni elemento: (batch*heads, 256, 77)
    all_maps = torch.stack(attention_mod.STORE_ATTN) # (total_layers, heads, 256, 77)
    
    # --- TOP ROW: Mappa media per ogni parola nel prompt ---
    # Media globale su tutti i time steps e su tutte le teste
    global_avg_map = all_maps.mean(dim=(0, 1)) # Forma: (256, 77)
    
    top_row_images = []
    # Evitiamo i token speciali <|startoftext|> e <|endoftext|>
    for i, word in enumerate(words):
        if word not in ["<|startoftext|>", "<|endoftext|>"]:
            clean_word = word.replace("</w>", "") # Pulisce i token di fine parola
            heatmap = create_heatmap(global_avg_map[:, i], image_shape)
            top_row_images.append((heatmap, clean_word))
            
    # --- BOTTOM ROW: Evoluzione temporale per la target_word ---
    bottom_row_images = []
    target_idx = None
    
    # Trova l'indice della parola cercata
    for i, word in enumerate(words):
        if target_word.lower() in word.lower():
            target_idx = i
            break
            
    if target_idx is not None:
        # Calcoliamo quante mappe ci sono per ogni step
        layers_per_step = len(attention_mod.STORE_ATTN) // num_steps
        
        # Estraiamo 7 step equidistanti come nel paper
        step_indices = np.linspace(0, num_steps - 1, 7, dtype=int)
        
        for step in step_indices:
            start_idx = step * layers_per_step
            end_idx = start_idx + layers_per_step
            
            # Media delle mappe SOLO per questo step specifico
            step_maps = all_maps[start_idx:end_idx]
            step_avg = step_maps.mean(dim=(0, 1))
            
            heatmap = create_heatmap(step_avg[:, target_idx], image_shape)
            bottom_row_images.append((heatmap, f"t={num_steps - step}"))
            
    return top_row_images, bottom_row_images

def process(input_image, prompt, a_prompt, n_prompt, target_word, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    # Svuotiamo la lista usando il nuovo import
    attention_mod.STORE_ATTN.clear()

    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape 
        
        # --- FIX DEFINITIVO: Calcoliamo l'area attesa ---
        # L'immagine al layer desiderato è ridotta di un fattore 32
        attention_mod.EXPECTED_SHAPE = (H // 32) * (W // 32)
        # ------------------------------------------------

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond,
                                                     log_every_t=1)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        
        # Manteniamo la decodifica intermedia (ATTENZIONE AL RISCHIO OOM SE GLI STEP SONO > 20)
        inter_images = []
        for step_latent in intermediates['pred_x0'][1:]:
            single_latent = step_latent[0:1]
            dec = model.decode_first_stage(single_latent)
            dec = (einops.rearrange(dec, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            inter_images.append(dec[0])

        inter_images_noise = []
        for step_latent in intermediates['x_inter'][1:]: 
            single_latent = step_latent[0:1]
            dec = model.decode_first_stage(single_latent)
            dec = (einops.rearrange(dec, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            inter_images_noise.append(dec[0])

    print(f"Ho catturato {len(attention_mod.STORE_ATTN)} mappe di attenzione!")
    
    # Processa le mappe di attenzione
    top_row_maps, bottom_row_maps = process_attention_maps(prompt, target_word, ddim_steps, (W, H))

    return [255 - detected_map] + results, inter_images, inter_images_noise, top_row_maps, bottom_row_maps

# --- UI Setup ---
block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Canny Edge Maps & Attention Viz")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources=['upload'], type="numpy")
            prompt = gr.Textbox(label="Prompt")
            # NUOVO CAMPO: Scegli quale parola monitorare nel tempo (bottom fig 4)
            target_word = gr.Textbox(label="Word to track over time (e.g. 'bear')", value="") 
            run_button = gr.Button(value="Run")
            
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1)
                high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt", value='longbody, lowres, bad anatomy, bad hands, missing fingers')
                
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", columns=2)
            
            # NUOVE GALLERIES PER LE MAPPE DI ATTENZIONE (Figure 4)
            gr.Markdown("### Figure 4 (Top): Average Attention Maps for Each Word")
            fig4_top = gr.Gallery(label='Average Maps', show_label=True, columns=5)
            
            gr.Markdown("### Figure 4 (Bottom): Attention Map Evolution over Time")
            fig4_bot = gr.Gallery(label='Timeline Maps', show_label=True, columns=7)

            with gr.Accordion("Intermediate VAE Decoding (WARNING: High VRAM)", open=False):
                intermediate_gallery = gr.Gallery(label='Preview of Sample x0', show_label=True, elem_id="gallery_inter", columns=4)
                denoised_Step = gr.Gallery(label='Denoising Step', show_label=True, elem_id="gallery_inter", columns=4)

    # Aggiunto target_word agli input
    ips = [input_image, prompt, a_prompt, n_prompt, target_word, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold]
    
    # Aggiunti fig4_top e fig4_bot agli output
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery, intermediate_gallery, denoised_Step, fig4_top, fig4_bot])

if __name__ == "__main__":
    block.launch(server_name='0.0.0.0', share=True)
