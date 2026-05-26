import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import ldm.modules.attention as attention_mod
import config

# Inizializzazione dei detector (Ossatura per Canny predefinita)
apply_canny = CannyDetector()

# Caricamento del modello ControlNet base
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'), strict=False) 
model = model.cuda()
ddim_sampler = DDIMSampler(model)

def process_word_swap(input_image, source_prompt, target_prompt, control_type, tau, num_samples, 
                      image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, 
                      low_threshold, high_threshold, a_prompt, n_prompt):
    
    # Pulizia preliminare delle liste globali di Prompt-to-Prompt
    attention_mod.P2P_SOURCE_ATTN.clear()
    attention_mod.P2P_INJECT_ATTN.clear()
    attention_mod.STORE_ATTN.clear()

    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape 
        
        # Calcolo dinamico dell'area attesa per le mappe di attenzione (riduzione di 32x nella U-Net)
        attention_mod.EXPECTED_SHAPE = (H // 32) * (W // 32)

        # --- OSSATURA RICHIESTA: SELEZIONE DELLA TIPOLOGIA DI CONTROLNET ---
        # Qui viene implementato lo scheletro per gestire i vari modelli di ControlNet.
        # Al momento la logica reale esegue Canny, per gli altri c'è il placeholder.
        if control_type == "Canny":
            detected_map = apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)
        elif control_type == "Depth":
            # TODO: Inserire qui l'invocazione del detector Depth (es. MidasDetector)
            # e il caricamento dinamico del rispettivo checkpoint se necessario.
            print("Modalità Depth selezionata (Placeholder)")
            detected_map = np.zeros_like(img) # Placeholder temporaneo
        elif control_type == "OpenPose":
            # TODO: Inserire qui la logica per OpenPose
            print("Modalità OpenPose selezionata (Placeholder)")
            detected_map = np.zeros_like(img) # Placeholder temporaneo
        else: # "Nessuno / Solo SD Base"
            print("Nessun condizionamento ControlNet applicato (Mappa vuota)")
            detected_map = np.zeros_like(img)

        # Preparazione del condizionamento di controllo (se strength > 0 influenza la generazione)
        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)

        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)
        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

        # --- FASE 1: GENERAZIONE SOURCE (ESTRAZIONE MAPPE) ---
        seed_everything(seed)
        attention_mod.P2P_READ_MODE = True
        attention_mod.P2P_INJECT_MODE = False
        
        cond_source = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([source_prompt + ', ' + a_prompt] * num_samples)]}
        
        samples_source, _ = ddim_sampler.sample(ddim_steps, num_samples, shape, cond_source, verbose=False, eta=eta,
                                                unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)
        
        x_source = model.decode_first_stage(samples_source)
        img_source = (einops.rearrange(x_source, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)[0]

        # --- FASE 2: CALCOLO DELLA SOGLIA TEMPORALE TAU ---
        total_maps = len(attention_mod.P2P_SOURCE_ATTN)
        maps_per_step = total_maps // ddim_steps
        inject_limit = int(tau * ddim_steps) * maps_per_step
        
        # Riempiamo la lista di iniezione con le mappe di sorgente fino al limite imposto da tau, poi None
        attention_mod.P2P_INJECT_ATTN.extend(attention_mod.P2P_SOURCE_ATTN[:inject_limit])
        attention_mod.P2P_INJECT_ATTN.extend([None] * (total_maps - inject_limit))

        # --- FASE 3: GENERAZIONE TARGET (INIEZIONE DELLE MAPPE COPIATE) ---
        seed_everything(seed)
        attention_mod.P2P_READ_MODE = False
        attention_mod.P2P_INJECT_MODE = True
        attention_mod.P2P_INJECT_INDEX = 0
        
        cond_target = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([target_prompt + ', ' + a_prompt] * num_samples)]}
        
        samples_target, _ = ddim_sampler.sample(ddim_steps, num_samples, shape, cond_target, verbose=False, eta=eta,
                                                unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)
        
        x_target = model.decode_first_stage(samples_target)
        img_target = (einops.rearrange(x_target, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)[0]

        # Ripristino dello stato di default per sicurezza
        attention_mod.P2P_INJECT_MODE = False

    # Restituisce i risultati alle gallerie della UI di Gradio
    # Mostriamo: Mappa di controllo, Immagine Originale (Source), Immagine Modificata (Target)
    return [255 - detected_map, img_source, img_target]

# --- INTERFACCIA GRAFICA GRADIO ---
block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## ControlNet + Prompt-to-Prompt (Word Swap) Orchestrator")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources=['upload'], type="numpy", label="Immagine di Input")
            source_prompt = gr.Textbox(label="Source Prompt (Originale)", value="Photo of a lemon cake")
            target_prompt = gr.Textbox(label="Target Prompt (Modificato con Word Swap)", value="Photo of a carrot cake")
            
            # NUOVO ELEMENTO: Dropdown rettangolare per la scelta del tipo di ControlNet
            control_type = gr.Dropdown(
                label="Tipologia ControlNet (Ossatura di Selezione)", 
                choices=["Canny", "Depth", "OpenPose", "Nessuno (SD Base)"], 
                value="Canny"
            )
            
            # Slider fondamentale per regolare la coerenza geometrica del Word Swap
            tau = gr.Slider(label="Tau (Soglia temporale iniezione Attention)", minimum=0.0, maximum=1.0, value=0.8, step=0.05)
            run_button = gr.Button(value="Run Prompt-to-Prompt")
            
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=4, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                
                # Default a 0.0 come preferito dall'utente per i test SD base puri, alzabile a piacimento
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=0.0, step=0.01)
                
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
            result_gallery = gr.Gallery(label='Output Visivi (Control Map, Source, Target)', show_label=True, elem_id="gallery", columns=3)

    # Elenco degli input ordinati per la funzione process_word_swap
    inputs_list = [
        input_image, source_prompt, target_prompt, control_type, tau, num_samples, 
        image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, 
        low_threshold, high_threshold, a_prompt, n_prompt
    ]
    
    run_button.click(fn=process_word_swap, inputs=inputs_list, outputs=[result_gallery])

if __name__ == "__main__":
    block.launch(server_name='0.0.0.0', share=True)