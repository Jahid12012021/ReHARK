import torch
import torch.nn.functional as F
import gc
import optuna
import os

# Suppress Optuna verbosity for clean logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

def power_transform(features, power=0.5):
    """Non-linear scaling to rectify feature distribution."""
    if not isinstance(features, torch.Tensor): return features
    return torch.sign(features) * (torch.abs(features) ** power)

def batched_inference(support, test_features, alpha, params, refined_prior, batch_size=4096):
    """Memory-efficient inference for Re-HARK using Multi-Scale RBF."""
    device = support.device
    num_test = test_features.shape[0]
    logits_list = []
    
    beta1, beta2 = params['beta1'], params['beta2']
    mix_k, scale_zs = params['mix_k'], params['scale_zs']
    
    with torch.no_grad():
        for i in range(0, num_test, batch_size):
            t_batch = test_features[i : i + batch_size]
            
            # 1. Zero-Shot Path (Global Prior)
            t_norm = F.normalize(t_batch, dim=1)
            zs_logits = scale_zs * (t_norm @ refined_prior.T)
            
            # 2. Kernel Path (Local Adaptation)
            dist_sq = torch.cdist(t_batch, support, p=2) ** 2
            k1 = torch.exp(-beta1 * dist_sq)
            k2 = torch.exp(-beta2 * dist_sq)
            k_matrix = mix_k * k1 + (1.0 - mix_k) * k2
            
            kernel_logits = k_matrix @ alpha
            logits_list.append(zs_logits + kernel_logits)
            
    return torch.cat(logits_list, dim=0)

def ReHARK(vecs, labels, val_features, val_labels, test_features, clip_weights, gpt3_weights, dataset, shots, seed, hp_selection, backbone='RN50'):
    """
    ReHARK: Refined Hybrid Adaptive RBF Kernels
    Full synergistic adaptation with 1000-trial optimization budget.
    """
    gc.collect()
    torch.cuda.empty_cache()
    
    device = vecs.device
    n_classes = len(labels.unique())
    feature_dim = vecs.shape[1]
    
    # Move data to device
    vecs_raw = vecs.float().to(device)
    val_raw = val_features.float().to(device)
    test_raw = test_features.float().to(device)
    
    if clip_weights.shape[0] == feature_dim: clip_weights = clip_weights.T
    if gpt3_weights.shape[0] == feature_dim: gpt3_weights = gpt3_weights.T
        
    clip_w_raw = clip_weights.float().to(device)
    gpt_w_raw = gpt3_weights.float().to(device)
    
    aug_labels = torch.cat([labels, labels], dim=0).to(device)
    cache = F.one_hot(aug_labels, num_classes=n_classes).float()
    val_targets = val_labels.to(device)
    labels = labels.to(device)

    print(f"🚀 ReHARK Optimization ({dataset}) - 1000 Trials")
    
    def objective(trial):
        # 1. Sample Hyperparameters
        beta1 = trial.suggest_float("beta1", 0.1, 30.0)
        beta2 = trial.suggest_float("beta2", 0.01, 5.0) 
        mix_k = trial.suggest_float("mix_k", 0.0, 1.0)
        lmbda = trial.suggest_float("lmbda", 0.001, 5.0, log=True)
        soft = trial.suggest_float("soft", 0.0, 1.0)
        var_adj = trial.suggest_float("var_adj", 0.0, 1.0)
        alpha_txt = trial.suggest_float("alpha_txt", 0.0, 1.0)
        refine_w = trial.suggest_float("refine_w", 0.0, 1.0)
        blend_img = trial.suggest_float("blend_img", 0.0, 2.0)
        scale_zs = trial.suggest_float("scale_zs", 1.0, 20.0)
        power = trial.suggest_float("power", 0.5, 1.0) 
        
        # 2. Dynamic Subsampling
        n_val = val_raw.shape[0]
        curr_val, curr_targets = (val_raw, val_targets) if n_val <= 4096 else (val_raw[:4096], val_targets[:4096])

        # 3. Hybrid Prior Refinement
        t_vecs = F.normalize(power_transform(vecs_raw, power), dim=1)
        t_val = F.normalize(power_transform(curr_val, power), dim=1)
        cw = F.normalize(power_transform(clip_w_raw, power), dim=1)
        gw = F.normalize(power_transform(gpt_w_raw, power), dim=1)
        
        base_text = F.normalize((1 - alpha_txt) * cw + alpha_txt * gw, dim=1)
        support_proto = torch.stack([t_vecs[labels == i].mean(0) for i in range(n_classes)])
        refined_prior = F.normalize((1 - refine_w) * base_text + refine_w * F.normalize(support_proto, dim=1), dim=1)
        
        # 4. Support Augmentation & Rectification
        bridge = F.normalize(t_vecs + (blend_img * refined_prior[labels]), dim=1)
        aug_support = torch.cat([t_vecs, bridge], dim=0)
        
        val_rect = F.normalize((t_val - t_val.mean(0)) * ((aug_support.std(0)+1e-6)/(t_val.std(0)+1e-6))**var_adj + t_val.mean(0) - soft*(t_val.mean(0)-aug_support.mean(0)), dim=1)
        
        # 5. Solver
        R = cache - scale_zs * (aug_support @ refined_prior.T)
        d_ss = torch.cdist(aug_support, aug_support, p=2) ** 2
        K = mix_k * torch.exp(-beta1 * d_ss) + (1.0 - mix_k) * torch.exp(-beta2 * d_ss)
        
        try:
            alpha = torch.linalg.solve(K + lmbda * torch.eye(K.size(0), device=device), R)
        except RuntimeError: return 0.0
        
        # 6. Evaluation
        d_val = torch.cdist(val_rect, aug_support, p=2) ** 2
        k_val = mix_k * torch.exp(-beta1 * d_val) + (1.0 - mix_k) * torch.exp(-beta2 * d_val)
        final = (scale_zs * (val_rect @ refined_prior.T)) + (k_val @ alpha)
        return (final.argmax(1) == curr_targets).float().mean().item()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1000, show_progress_bar=False)
    best = study.best_params
    
    # Final Inference using best parameters
    t_vecs = F.normalize(power_transform(vecs_raw, best['power']), dim=1)
    t_test = F.normalize(power_transform(test_raw, best['power']), dim=1)
    base_text = F.normalize((1 - best['alpha_txt']) * F.normalize(power_transform(clip_w_raw, best['power']), dim=1) + best['alpha_txt'] * F.normalize(power_transform(gpt_w_raw, best['power']), dim=1), dim=1)
    support_proto = torch.stack([t_vecs[labels == i].mean(0) for i in range(n_classes)])
    refined_prior = F.normalize((1 - best['refine_w']) * base_text + best['refine_w'] * F.normalize(support_proto, dim=1), dim=1)
    
    bridge = F.normalize(t_vecs + (best['blend_img'] * refined_prior[labels]), dim=1)
    aug_support = torch.cat([t_vecs, bridge], dim=0)
    test_rect = F.normalize((t_test - t_test.mean(0)) * ((aug_support.std(0)+1e-6)/(t_test.std(0)+1e-6))**best['var_adj'] + t_test.mean(0) - best['soft']*(t_test.mean(0)-aug_support.mean(0)), dim=1)
    
    K = (best['mix_k'] * torch.exp(-best['beta1'] * torch.cdist(aug_support, aug_support, p=2)**2) + 
         (1.0 - best['mix_k']) * torch.exp(-best['beta2'] * torch.cdist(aug_support, aug_support, p=2)**2))
    alpha = torch.linalg.solve(K + best['lmbda'] * torch.eye(K.size(0), device=device), cache - best['scale_zs']*(aug_support @ refined_prior.T))
    
    return batched_inference(aug_support, test_rect, alpha, best, refined_prior).cpu()
