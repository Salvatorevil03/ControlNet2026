import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from annotator.canny import CannyDetector
from annotator.util import HWC3, resize_image
from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from torch.optim import Adam
from tqdm import tqdm

def null_text_inversion_guess_mode(
    model, 
    sampler, 
    z0, 
    control_map, 
    num_steps=50, 
    cfg_scale=7.5, 
    lr=1e-2, 
    inner_iters=10
):
    """
    Esegue la Null Text Inversion in Guess Mode.
    
    Args:
        model: Il modello ControlLDM caricato (cldm.py)
        sampler: La tua istanza DDIM (ddim_hacked.py adattata per l'inversione)
        z0: Il latente dell'immagine originale codificato tramite VAE (B, C, H, W)
        control_map: L'immagine di controllo (es. Canny) processata (B, C, H, W)
    """
    device = model.device
    
    # --------------------------------------------------------------------------
    # FASE 1: Estrazione della Traiettoria (Inversione DDIM)
    # --------------------------------------------------------------------------
    print("Estrazione della traiettoria DDIM...")
    
    # In Guess Mode, il testo condizionante è vuoto
    cond_text = model.get_learned_conditioning([""]).to(device)
    
    # Prepariamo il dizionario condizionale come fa ControlNet
    cond_dict = {
        "c_crossattn": [cond_text],
        "c_concat": [control_map]
    }
    
    sampler.make_schedule(ddim_num_steps=num_steps, ddim_eta=0.0, verbose=False)
    z_trajectory = _ddim_inversion(
        sampler,
        z0,
        cond_dict,
        num_steps=num_steps,
        unconditional_guidance_scale=1.0,
        guess_mode=True
    )
    
    # --------------------------------------------------------------------------
    # FASE 2: Setup dell'Ottimizzazione
    # --------------------------------------------------------------------------
    print("Inizio Null Text Inversion...")
    
    # Inizializziamo l'embedding incondizionato con un prompt vuoto
    uncond_emb_initial = model.get_learned_conditioning([""]).detach().to(device)
    
    # L'embedding che andremo ad ottimizzare
    uncond_emb = uncond_emb_initial.clone().requires_grad_(True)
    
    # Ottimizzatore che agisce SOLO sull'embedding incondizionato
    optimizer = Adam([uncond_emb], lr=lr)
    
    optimized_null_texts = []
    
    # --------------------------------------------------------------------------
    # FASE 3: Loop di Ottimizzazione (a ritroso, da T scendendo verso 1)
    # --------------------------------------------------------------------------
    
    # Gli step temporali del DDIM scendono da T a 0
    time_steps = reversed(sampler.ddim_timesteps)
    
    for i, t in enumerate(tqdm(time_steps, desc="NTI Loop")):
        # Prendi il latente attuale e il target (quello allo step precedente)
        z_t = z_trajectory[-(i + 1)].detach()
        z_t_minus_1_target = z_trajectory[-(i + 2)].detach()
        
        # Array con il timestep formattato per l'UNet
        t_batch = torch.full((z_t.shape[0],), t, device=device, dtype=torch.long)
        
        for _ in range(inner_iters):
            optimizer.zero_grad()
            
            # --- Passaggio Condizionato (Guess Mode Attiva) ---
            with torch.no_grad():
                # In ControlNet, eps_cond include i residui scalati del ControlNet.
                # Richiama la funzione apply_model internamente passando cond_dict.
                eps_cond = model.apply_model(z_t, t_batch, cond_dict) 
                
            # --- Passaggio Incondizionato (Null Text Ottimizzato) ---
            # Il condizionamento incondizionato non deve avere il control_map (o deve averlo neutro)
            uncond_dict = {
                "c_crossattn": [uncond_emb],
                "c_concat": [torch.zeros_like(control_map)] # Spesso si usa una mappa neutra
            }
            
            # Qui scorrono i gradienti
            eps_uncond = model.apply_model(z_t, t_batch, uncond_dict)
            
            # --- Classifier-Free Guidance ---
            eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            
            # --- Step DDIM (Mock) ---
            # Usa la formula matematica standard del DDIM per calcolare z_{t-1} predetto
            # a partire da z_t e eps_pred.
            alpha_t = sampler.ddim_alphas[t]
            alpha_prev = sampler.ddim_alphas_prev[t]
            sqrt_one_minus_alpha_prev = sampler.ddim_sqrt_one_minus_alphas[t] # Semplificato
            
            pred_x0 = (z_t - torch.sqrt(1. - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
            dir_xt = torch.sqrt(1. - alpha_prev) * eps_pred
            z_t_minus_1_pred = torch.sqrt(alpha_prev) * pred_x0 + dir_xt
            
            # --- Loss e Backpropagation ---
            loss = F.mse_loss(z_t_minus_1_pred, z_t_minus_1_target)
            loss.backward()
            optimizer.step()
            
        # Salva l'embedding ottimizzato per questo timestep
        optimized_null_texts.append(uncond_emb.detach().clone())
        
        # Imposta l'embedding iniziale del prossimo step con quello ottimizzato ora
        # (aiuta la convergenza e garantisce fluidità nella traiettoria)
        with torch.no_grad():
            uncond_emb.copy_(uncond_emb.detach())

    # La lista contiene gli embedding da T a 0. Li invertiamo per averli da 0 a T.
    return optimized_null_texts[::-1], z_trajectory


def _ddim_inversion(sampler, z0, cond_dict, num_steps, unconditional_guidance_scale=1.0, guess_mode=True):
    """Fallback per ottenere la traiettoria DDIM usando il sampler esistente."""
    if hasattr(sampler, 'ddim_inversion'):
        return sampler.ddim_inversion(
            z0,
            cond_dict,
            unconditional_guidance_scale=unconditional_guidance_scale,
            guess_mode=guess_mode
        )

    _, out = sampler.encode(
        z0,
        cond_dict,
        t_enc=num_steps,
        return_intermediates=num_steps,
        unconditional_guidance_scale=unconditional_guidance_scale,
        unconditional_conditioning=None
    )

    trajectory = [z0.detach().clone()]
    for x in out.get('intermediates', []):
        trajectory.append(x.detach().clone())
    return trajectory


def run_null_text_inversion_example():
    """Esempio di chiamata della funzione nello stesso file usando il progetto ControlNet."""
    model_config = './models/cldm_v15.yaml'
    model_checkpoint = './models/control_sd15_canny.pth'
    image_path = './test_imgs/human.png'
    ddim_steps = 50
    image_resolution = 512
    low_threshold = 100
    high_threshold = 200

    print("Loading model and sampler...")
    model = create_model(model_config).cpu()
    model.load_state_dict(load_state_dict(model_checkpoint, location='cuda'), strict=False)
    model = model.cuda()
    sampler = DDIMSampler(model)

    print(f"Loading test image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Immagine non trovata: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = HWC3(image)
    image = resize_image(image, image_resolution)

    print("Calcolo la mappa di controllo Canny...")
    canny = CannyDetector()
    detected_map = canny(image, low_threshold, high_threshold)
    detected_map = HWC3(detected_map)
    control_map = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    control_map = control_map.unsqueeze(0).permute(0, 3, 1, 2)

    print("Calcolo il latente z0 dell'immagine originale...")
    image_tensor = torch.from_numpy(image.copy()).float().cuda() / 127.5 - 1.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    z0 = model.encode_first_stage(image_tensor)

    optimized_null_texts, z_trajectory = null_text_inversion_guess_mode(
        model=model,
        sampler=sampler,
        z0=z0,
        control_map=control_map,
        num_steps=ddim_steps,
        cfg_scale=7.5,
        lr=1e-2,
        inner_iters=10
    )

    print(f"Ottimizzati {len(optimized_null_texts)} embedding null text")
    print(f"Traiettoria latente ottenuta: {len(z_trajectory)} step")
    return optimized_null_texts, z_trajectory


if __name__ == "__main__":
    run_null_text_inversion_example()
