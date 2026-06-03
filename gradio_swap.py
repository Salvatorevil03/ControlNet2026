import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3

# --- IMPORT DI TUTTI I DETECTOR ---
from annotator.canny import CannyDetector
from annotator.midas import MidasDetector
from annotator.openpose import OpenposeDetector
from annotator.uniformer import UniformerDetector
from annotator.hed import HEDdetector, nms
from annotator.mlsd import MLSDdetector  # <-- NUOVO

from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import ldm.modules.attention as attention_mod
import config

# --- 1. INIZIALIZZAZIONE DETECTOR (Globali per evitare lag) ---
print("Inizializzazione dei detector (Annotators)...")
apply_canny = CannyDetector()
apply_midas = MidasDetector()
apply_openpose = OpenposeDetector()
apply_uniformer = UniformerDetector()
apply_hed = HEDdetector()
apply_mlsd = MLSDdetector()  # <-- NUOVO

# --- 2. CONFIGURAZIONE DEI PESI DEI MODELLI CONTROLNET ---
CONTROLNET_WEIGHTS = {
    "Canny": './models/control_sd15_canny.pth',
    "Depth": './models/control_sd15_depth.pth',
    "Normal": './models/control_sd15_normal.pth',
    "OpenPose": './models/control_sd15_openpose.pth',
    "Segmentation": './models/control_sd15_seg.pth',
    "HED": './models/control_sd15_hed.pth',
    "Fake Scribble": './models/control_sd15_scribble.pth',
    "Scribble (Da Immagine/Bozza)": './models/control_sd15_scribble.pth',
    "MLSD (Hough Lines)": './models/control_sd15_mlsd.pth'  # <-- NUOVO
}

# Inizializzazione del modello base SD 1.5
print("Inizializzazione dell'architettura del modello...")
model = create_model('./models/cldm_v15.yaml').cpu()
model = model.cuda()
ddim_sampler = DDIMSampler(model)

CURRENT_LOADED_MODEL = None

def switch_controlnet_model(control_type):
    """Gestisce il cambio dinamico dei pesi ControlNet per evitare colli di bottiglia sulla VRAM."""
    global CURRENT_LOADED_MODEL
    
    if control_type == "Nessuno (SD Base)":
        CURRENT_LOADED_MODEL = "Nessuno (SD Base)"
        return
        
    if control_type != CURRENT_LOADED_MODEL:
        weight_path = CONTROLNET_WEIGHTS.get(control_type)
        if weight_path:
            print(f"Cambio modello rilevato: Caricamento pesi per {control_type} da {weight_path}...")
            model.load_state_dict(load_state_dict(weight_path, location='cuda'), strict=False)
            CURRENT_LOADED_MODEL = control_type
        else:
            print(f"ATTENZIONE: Percorso dei pesi per {control_type} non trovato!")

# --- 3. FUNZIONE PRINCIPALE (WORD SWAP + MULTI-PREPROCESSING) ---
def process_word_swap(input_image, source_prompt, target_prompt, control_type, tau, num_samples, 
                      image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, 
                      low_threshold, high_threshold, bg_threshold, mlsd_value_threshold, mlsd_distance_threshold, a_prompt, n_prompt): # <-- Aggiunti Parametri MLSD
    
    # Pulizia preliminare delle liste globali di Prompt-to-Prompt
    attention_mod.P2P_SOURCE_ATTN.clear()
    attention_mod.P2P_INJECT_ATTN.clear()
    attention_mod.STORE_ATTN.clear()

    # Cambio pesi dinamico
    switch_controlnet_model(control_type)

    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape 
        
        # Calcolo dinamico dell'area attesa per le mappe di attenzione (riduzione 32x)
        attention_mod.EXPECTED_SHAPE = (H // 32) * (W // 32)
        
        input_image_full = HWC3(input_image)
        detected_map_for_control = None

        # --- SELEZIONE DEL PREPROCESSING ESTRAZIONE MAPPE ---
        if control_type == "Canny":
            detected_map = apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)
            
        elif control_type == "Depth":
            detected_map, _ = apply_midas(resize_image(input_image_full, detect_resolution))
            detected_map = HWC3(detected_map)
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
            
        elif control_type == "Normal":
            _, detected_map = apply_midas(resize_image(input_image_full, detect_resolution), bg_th=bg_threshold)
            detected_map = HWC3(detected_map)
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
            # La mappa normale necessita di BGR -> RGB per il tensore di controllo
            detected_map_for_control = detected_map[:, :, ::-1].copy()

        elif control_type == "OpenPose":
            detected_map, _ = apply_openpose(resize_image(input_image_full, detect_resolution))
            detected_map = HWC3(detected_map)
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        elif control_type == "Segmentation":
            detected_map = apply_uniformer(resize_image(input_image_full, detect_resolution))
            detected_map = HWC3(detected_map)
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        elif control_type == "HED":
            detected_map = apply_hed(resize_image(input_image_full, detect_resolution))
            detected_map = HWC3(detected_map)
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        elif control_type == "Fake Scribble":
            detected_map = apply_hed(resize_image(input_image_full, detect_resolution))
            detected_map = HWC3(detected_map)
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
            detected_map = nms(detected_map, 127, 3.0)
            detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
            detected_map[detected_map > 4] = 255
            detected_map[detected_map < 255] = 0

        elif control_type == "MLSD (Hough Lines)": # <-- NUOVA LOGICA MLSD
            detected_map = apply_mlsd(resize_image(input_image_full, detect_resolution), mlsd_value_threshold, mlsd_distance_threshold)
            detected_map = HWC3(detected_map)
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        elif control_type == "Scribble (Da Immagine/Bozza)":
            detected_map = np.zeros_like(img, dtype=np.uint8)
            detected_map[np.min(img, axis=2) < 127] = 255
            
        else: # "Nessuno (SD Base)"
            detected_map = np.zeros_like(img)

        # Se non è stata forzata un'altra mappa per il controllo (es. Normal), usiamo quella base
        if detected_map_for_control is None:
            detected_map_for_control = detected_map.copy()

        # Preparazione del tensore di controllo
        control = torch.from_numpy(detected_map_for_control).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)

        actual_strength = 0.0 if control_type == "Nessuno (SD Base)" else strength

        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)
        model.control_scales = [actual_strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([actual_strength] * 13)

        # --- FASE 1: GENERAZIONE SOURCE ---
        seed_everything(seed)
        attention_mod.P2P_READ_MODE = True
        attention_mod.P2P_INJECT_MODE = False
        
        cond_source = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([source_prompt + ', ' + a_prompt] * num_samples)]}
        
        samples_source, _ = ddim_sampler.sample(ddim_steps, num_samples, shape, cond_source, verbose=False, eta=eta,
                                                unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)
        
        x_source = model.decode_first_stage(samples_source)
        img_source = (einops.rearrange(x_source, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)[0]

        # --- FASE 2: SOGLIA TAU ---
        total_maps = len(attention_mod.P2P_SOURCE_ATTN)
        if total_maps > 0:
            maps_per_step = total_maps // ddim_steps
            inject_limit = int(tau * ddim_steps) * maps_per_step
            attention_mod.P2P_INJECT_ATTN.extend(attention_mod.P2P_SOURCE_ATTN[:inject_limit])
            attention_mod.P2P_INJECT_ATTN.extend([None] * (total_maps - inject_limit))

        # --- FASE 3: GENERAZIONE TARGET ---
        seed_everything(seed)
        attention_mod.P2P_READ_MODE = False
        attention_mod.P2P_INJECT_MODE = True
        attention_mod.P2P_INJECT_INDEX = 0
        
        cond_target = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([target_prompt + ', ' + a_prompt] * num_samples)]}
        
        samples_target, _ = ddim_sampler.sample(ddim_steps, num_samples, shape, cond_target, verbose=False, eta=eta,
                                                unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)
        
        x_target = model.decode_first_stage(samples_target)
        img_target = (einops.rearrange(x_target, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)[0]

        attention_mod.P2P_INJECT_MODE = False

    if control_type in ["Segmentation", "Normal", "Depth"]:
        display_map = detected_map
    else:
        display_map = 255 - detected_map

    return [display_map, img_source, img_target]

# --- INTERFACCIA GRAFICA GRADIO ---
block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Universal ControlNet + P2P Word Swap Orchestrator")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources=['upload'], type="numpy", label="Immagine di Input")
            source_prompt = gr.Textbox(label="Source Prompt (Originale)", value="Photo of a room")
            target_prompt = gr.Textbox(label="Target Prompt (Modificato)", value="Photo of a modern room")
            
            control_type = gr.Dropdown(
                label="Tipologia ControlNet", 
                choices=[
                    "Canny", "Depth", "Normal", "OpenPose", 
                    "Segmentation", "HED", "Fake Scribble", 
                    "Scribble (Da Immagine/Bozza)", "MLSD (Hough Lines)", "Nessuno (SD Base)" # <-- AGGIUNTO MLSD
                ], 
                value="Canny"
            )
            
            tau = gr.Slider(label="Tau (Soglia iniezione Attention P2P)", minimum=0.0, maximum=1.0, value=0.8, step=0.05)
            run_button = gr.Button(value="Run Prompt-to-Prompt")
            
            with gr.Accordion("Advanced ControlNet & Image Options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=4, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                
                detect_resolution = gr.Slider(label="Detector Resolution", minimum=128, maximum=1024, value=512, step=1)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                
                # Canny
                low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1)
                high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1)
                
                # Normal
                bg_threshold = gr.Slider(label="Normal background threshold", minimum=0.0, maximum=1.0, value=0.4, step=0.01)
                
                # MLSD (Hough Lines)
                mlsd_value_threshold = gr.Slider(label="MLSD Value Threshold", minimum=0.01, maximum=2.0, value=0.1, step=0.01)
                mlsd_distance_threshold = gr.Slider(label="MLSD Distance Threshold", minimum=0.01, maximum=20.0, value=0.1, step=0.01)
                
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt", value='longbody, lowres, bad anatomy, bad hands, missing fingers')
                
        with gr.Column():
            result_gallery = gr.Gallery(label='Output Visivi (Control Map, Source, Target)', show_label=True, elem_id="gallery", columns=3)

    inputs_list = [
        input_image, source_prompt, target_prompt, control_type, tau, num_samples, 
        image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, 
        low_threshold, high_threshold, bg_threshold, mlsd_value_threshold, mlsd_distance_threshold, a_prompt, n_prompt
    ]
    
    run_button.click(fn=process_word_swap, inputs=inputs_list, outputs=[result_gallery])

if __name__ == "__main__":
    block.launch(server_name='0.0.0.0', share=True)
