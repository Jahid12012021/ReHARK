import os
import shutil
import sys
import torch
from tqdm import tqdm
import clip

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
# Establishes the project root where main.py resides
ROOT_DIR = os.getcwd()
LIB_DIR = os.path.join(ROOT_DIR, "gpt_lib")
PROMPTS_SRC = os.path.join(ROOT_DIR, "gpt3_prompts")

# ==========================================
# 2. DYNAMIC LIBRARY CONSTRUCTION (gpt_lib)
# ==========================================
# This section converts the .py prompt files into an importable package
if os.path.exists(PROMPTS_SRC):
    print(f"🔍 Building gpt_lib from: {PROMPTS_SRC}")
    
    if os.path.exists(LIB_DIR):
        shutil.rmtree(LIB_DIR)
    os.makedirs(LIB_DIR)

    init_content = ["# Master Loader\nALL_TEMPLATES = {}\n"]
    
    # Identify files to include in the library
    search_names = [
        'imagenet.py', 'caltech101.py', 'dtd.py', 'eurosat.py', 
        'fgvc_aircraft.py', 'food101.py', 'oxford_flowers.py', 
        'oxford_pets.py', 'stanford_cars.py', 'sun397.py', 'ucf101.py'
    ]

    for fname in os.listdir(PROMPTS_SRC):
        if fname in search_names:
            src_path = os.path.join(PROMPTS_SRC, fname)
            # Process and rename for internal loading
            new_name = fname.replace(".py", "_p.py")
            dst_path = os.path.join(LIB_DIR, new_name)
            shutil.copy(src_path, dst_path)
            
            # Key normalization for dataset lookup
            dataset_key = fname.replace(".py", "").replace("_", "").replace("-","").lower()
            var_name = None
            
            # Variable Mapping logic
            if 'imagenet' in fname: var_name = "IMAGENET_TEMPLATES"
            elif 'caltech' in fname: var_name = "CALTECH101_TEMPLATES"
            elif 'dtd' in fname: var_name = "DTD_TEMPLATES"; dataset_key='describabletextures'
            elif 'eurosat' in fname: var_name = "EUROSAT_TEMPLATES"
            elif 'fgvc' in fname: var_name = "FGVC_AIRCRAFT_TEMPLATES"; dataset_key='fgvcaircraft'
            elif 'food' in fname: var_name = "FOOD101_TEMPLATES"
            elif 'flowers' in fname: var_name = "OXFORD_FLOWERS_TEMPLATES"; dataset_key='oxfordflowers'
            elif 'pets' in fname: var_name = "OXFORD_PETS_TEMPLATES"; dataset_key='oxfordpets'
            elif 'cars' in fname: var_name = "STANFORD_CARS_TEMPLATES"; dataset_key='stanfordcars'
            elif 'sun397' in fname: var_name = "SUN397_TEMPLATES"
            elif 'ucf' in fname: var_name = "UCF101_TEMPLATES"
            
            if var_name:
                mod_name = new_name[:-3]
                init_content.append(f"try:\n    from .{mod_name} import {var_name}")
                init_content.append(f"    ALL_TEMPLATES['{dataset_key}'] = {var_name}")
                init_content.append(f"except ImportError: pass\n")

    with open(os.path.join(LIB_DIR, "__init__.py"), "w") as f:
        f.write("\n".join(init_content))
    print(f"✅ gpt_lib successfully built at {LIB_DIR}")
else:
    print(f"⚠️ Warning: gpt3_prompts/ not found at {PROMPTS_SRC}")

# ==========================================
# 3. GLOBAL TEMPLATE LOADER
# ==========================================
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

try:
    import gpt_lib
    ALL_TEMPLATES = gpt_lib.ALL_TEMPLATES
except ImportError:
    ALL_TEMPLATES = {}

# ==========================================
# 4. WEIGHT GENERATION UTILITY
# ==========================================
def get_gpt3_weights(dataset_name, classnames, clip_model, device):
    """Generates normalized GPT-3 text embeddings for ReHARK adaptation."""
    print(f"Generating GPT-3 Text Weights for {dataset_name}...")
    
    # Normalize keys to match the ALL_TEMPLATES library
    key = dataset_name.lower().replace("_", "").replace("-", "")
    mapping = {
        'aircraft': 'fgvcaircraft', 'flower': 'oxfordflowers', 
        'pet': 'oxfordpets', 'car': 'stanfordcars', 
        'texture': 'describabletextures', 'food': 'food101'
    }
    for k, v in mapping.items():
        if k in key: key = v
    
    templates_dict = ALL_TEMPLATES.get(key, None)
    gpt_features = []
    
    with torch.no_grad():
        for classname in tqdm(classnames):
            prompts = []
            if templates_dict and classname in templates_dict:
                prompts = templates_dict[classname]
            
            # Fallback to standard prompt if no GPT-3 entry exists
            if not prompts:
                prompts = [f"a photo of a {classname}."]
            
            texts = clip.tokenize(prompts, truncate=True).to(device)
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            gpt_features.append(class_embedding)
            
    gpt_features = torch.stack(gpt_features, dim=0)
    
    # Ensure correct shape [Classes, Dim] for the solver
    if gpt_features.shape[0] == 1024 and gpt_features.shape[1] != 1024:
        gpt_features = gpt_features.T
        
    return gpt_features