import cv2
import einops
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

# Inizializzazione globale del modello (come nel tuo script Gradio)
apply_canny = CannyDetector()
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

def run_word_swap(input_image, source_prompt, target_prompt, tau=0.8, num_samples=1, 
                  image_resolution=512, ddim_steps=30, guess_mode=False, strength=0.0, 
                  scale=9.0, seed=-1, eta=0.0, low_threshold=100, high_threshold=200, 
                  a_prompt='best quality, extremely detailed', 
                  n_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers'):
    
    """
    Esegue il Prompt-to-Prompt Word Swap in due passaggi.
    tau: float tra 0.0 e 1.0. Determina la percentuale di step in cui iniettare le mappe.
         es. 0.8 significa inietta per l'80% degli step, lascia libero il modello per il restante 20%.
    """
    
    with torch.no_grad():
        # 1. Preparazione dell'immagine di input e della mappa Canny
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape
        
        # Imposta dinamicamente la dimensione attesa della mappa (FIX DEFINITIVO)
        attention_mod.EXPECTED_SHAPE = (H // 32) * (W // 32)
        
        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        # Fissiamo il seed. Fondamentale per condividere z_T tra i due passaggi!
        if seed == -1:
            seed = random.randint(0, 65535)
        
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)
        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

        print(f"--- FASE 1: Generazione Source (Seed: {seed}) ---")
        seed_everything(seed)
        
        # Configura attention_mod per la Lettura
        attention_mod.P2P_READ_MODE = True
        attention_mod.P2P_INJECT_MODE = False
        attention_mod.P2P_SOURCE_ATTN.clear()
        
        cond_source = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([source_prompt + ', ' + a_prompt] * num_samples)]}
        
        samples_source, _ = ddim_sampler.sample(ddim_steps, num_samples, shape, cond_source, verbose=False, eta=eta,
                                                unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)
        
        x_source = model.decode_first_stage(samples_source)
        img_source = (einops.rearrange(x_source, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)[0]
        
        print(f"Mappe estratte: {len(attention_mod.P2P_SOURCE_ATTN)}")


        print(f"--- FASE 2: Preparazione Tau (tau = {tau}) ---")
        # Calcoliamo quante mappe iniettare in base a tau
        total_maps = len(attention_mod.P2P_SOURCE_ATTN)
        maps_per_step = total_maps // ddim_steps
        inject_limit = int(tau * ddim_steps) * maps_per_step
        
        attention_mod.P2P_INJECT_ATTN.clear()
        # Copia le mappe per i primi X step
        attention_mod.P2P_INJECT_ATTN.extend(attention_mod.P2P_SOURCE_ATTN[:inject_limit])
        # Riempi di None per i restanti step (il modello userà la sua attenzione naturale)
        attention_mod.P2P_INJECT_ATTN.extend([None] * (total_maps - inject_limit))


        print(f"--- FASE 3: Generazione Target / Iniezione ---")
        # Ripristiniamo ESATTAMENTE lo stesso rumore iniziale (z_T)
        seed_everything(seed)
        
        # Configura attention_mod per la Scrittura
        attention_mod.P2P_READ_MODE = False
        attention_mod.P2P_INJECT_MODE = True
        attention_mod.P2P_INJECT_INDEX = 0
        
        cond_target = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([target_prompt + ', ' + a_prompt] * num_samples)]}
        
        samples_target, _ = ddim_sampler.sample(ddim_steps, num_samples, shape, cond_target, verbose=False, eta=eta,
                                                unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)
        
        x_target = model.decode_first_stage(samples_target)
        img_target = (einops.rearrange(x_target, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)[0]
        
        # Spegniamo il meccanismo P2P per sicurezza
        attention_mod.P2P_INJECT_MODE = False
        
        return img_source, img_target, 255 - detected_map

# --- Esempio di Utilizzo Standalone ---
if __name__ == "__main__":
    # Carica un'immagine di test
    input_img = cv2.imread('/content/Huskiesatrest.jpg')
    if input_img is not None:
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        
        src_prompt = "A lemon cake"
        tgt_prompt = "A carrot cake" # Word Swap: bicycle -> car
        
        img_src, img_tgt, canny_map = run_word_swap(
            input_image=input_img,
            source_prompt=src_prompt,
            target_prompt=tgt_prompt,
            tau=0.95, # Manteniamo la composizione per l'80% degli step
            seed=14300  # Seed fisso per test ripetibili
        )
        
        # Salva i risultati
        cv2.imwrite("01_canny.jpg", cv2.cvtColor(canny_map, cv2.COLOR_RGB2BGR))
        cv2.imwrite("02_source.jpg", cv2.cvtColor(img_src, cv2.COLOR_RGB2BGR))
        cv2.imwrite("03_target.jpg", cv2.cvtColor(img_tgt, cv2.COLOR_RGB2BGR))
        print("Test completato con successo. Immagini salvate.")
        # --- PLOT DEI RISULTATI ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot Mappa di Controllo (Canny)
        axes[0].imshow(canny_map)
        axes[0].set_title("Controllo (Canny)", fontsize=14)
        axes[0].axis('off')
        
        # Plot Immagine Source
        axes[1].imshow(img_src)
        axes[1].set_title(f"Source\n'{src_prompt}'", fontsize=14)
        axes[1].axis('off')
        
        # Plot Immagine Target
        axes[2].imshow(img_tgt)
        axes[2].set_title(f"Target (Word Swap)\n'{tgt_prompt}'", fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        print("Errore: Immagine di test 'test_image.jpg' non trovata.")