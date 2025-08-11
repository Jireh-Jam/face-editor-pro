"""
AI Face Editor with LoRA weights
Streamlit application for face editing using Stable Diffusion + ControlNet + Custom LoRA
"""

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import io
import os
import gdown
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="AI Face Editor Pro",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        margin-top: 10px;
        border-radius: 10px;
    }
    div[data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    .edit-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ================== Model Management ==================
class ModelManager:
    """Manages model downloading and caching"""
    
    @staticmethod
    def ensure_models_downloaded():
        """Download models if they don't exist"""
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Model URLs (replace with your actual URLs)
        models = {
            "79999_iter.pth": {
                "url": "https://www.dropbox.com/scl/fi/beh64brrnirwep2phodcw/79999_iter.pth?dl=1",
                "path": models_dir / "79999_iter.pth"
            },
            "to8contrast.safetensors": {
                "url": "https://www.dropbox.com/scl/fi/r64z2v97sirsjcd4nnmtx/to8contrast.safetensors?dl=1",
                "path": models_dir / "to8contrast.safetensors"
            }
        }
        
        for model_name, model_info in models.items():
            if not model_info["path"].exists():
                with st.spinner(f"Downloading {model_name}..."):
                    try:
                        gdown.download(model_info["url"], str(model_info["path"]), quiet=False)
                        st.success(f"✅ Downloaded {model_name}")
                    except Exception as e:
                        st.error(f"Failed to download {model_name}: {e}")
                        st.info("Please download manually and place in 'models' folder")
                        return False
        return True

# ================== BiSeNet Face Parser ==================
class FaceParser:
    """Face parsing using BiSeNet"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = model_path
        
    def load_model(self):
        """Load BiSeNet model"""
        try:
            # Import the model architecture
            # Note: You'll need to include the model.py file in your repo
            sys.path.append(str(Path(__file__).parent))
            from model import BiSeNet
            
            self.model = BiSeNet(n_classes=19)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            return True
        except Exception as e:
            logger.error(f"Failed to load BiSeNet: {e}")
            return False
    
    def parse_face(self, image: Image.Image) -> np.ndarray:
        """Parse face regions"""
        import torchvision.transforms as transforms
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Parse
        with torch.no_grad():
            output = self.model(img_tensor)[0]
            parsing = output.squeeze(0).cpu().numpy().argmax(0)
        
        return parsing

# ================== LoRA Weight Loader ==================
class LoRALoader:
    """Handles LoRA weight loading and application"""
    
    @staticmethod
    def apply_lora_weights(pipe, lora_path: str, alpha: float = 0.65):
        """Apply LoRA weights to the pipeline"""
        from safetensors.torch import load_file
        
        try:
            state_dict = load_file(lora_path)
            
            LORA_PREFIX_UNET = 'lora_unet'
            LORA_PREFIX_TEXT_ENCODER = 'lora_te'
            
            visited = []
            
            # Apply LoRA weights
            for key in state_dict:
                if '.alpha' in key or key in visited:
                    continue
                
                # Determine target model
                if 'text' in key:
                    layer_infos = key.split('.')[0].split(LORA_PREFIX_TEXT_ENCODER+'_')[-1].split('_')
                    curr_layer = pipe.text_encoder
                else:
                    layer_infos = key.split('.')[0].split(LORA_PREFIX_UNET+'_')[-1].split('_')
                    curr_layer = pipe.unet
                
                # Navigate to target layer
                temp_name = layer_infos.pop(0)
                while len(layer_infos) > -1:
                    try:
                        curr_layer = curr_layer.__getattr__(temp_name)
                        if len(layer_infos) > 0:
                            temp_name = layer_infos.pop(0)
                        elif len(layer_infos) == 0:
                            break
                    except Exception:
                        if len(temp_name) > 0:
                            temp_name += '_'+layer_infos.pop(0)
                        else:
                            temp_name = layer_infos.pop(0)
                
                # Get weight pair
                pair_keys = []
                if 'lora_down' in key:
                    pair_keys.append(key.replace('lora_down', 'lora_up'))
                    pair_keys.append(key)
                else:
                    pair_keys.append(key)
                    pair_keys.append(key.replace('lora_up', 'lora_down'))
                
                # Apply weights
                if len(state_dict[pair_keys[0]].shape) == 4:
                    weight_up = state_dict[pair_keys[0]].to(torch.float32)
                    weight_down = state_dict[pair_keys[1]].to(torch.float32)
                else:
                    weight_up = state_dict[pair_keys[0]].to(torch.float32)
                    weight_down = state_dict[pair_keys[1]].to(torch.float32)
                    
                    if weight_up.shape[0] != weight_down.shape[1]:
                        weight_down = weight_down.T
                    
                    # Apply LoRA weights with alpha scaling
                    curr_layer.weight.data += alpha * torch.mm(weight_down, weight_up).to(curr_layer.weight.data.device)
                
                visited.extend(pair_keys)
            
            logger.info(f"Successfully applied LoRA weights with alpha={alpha}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply LoRA weights: {e}")
            return False

# ================== Main Face Editor ==================
class FaceEditorPro:
    """Enhanced face editor with LoRA support"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None
        self.controlnet = None
        self.face_parser = None
        
    @st.cache_resource
    def load_models(_self):
        """Load all models with caching"""
        try:
            from diffusers import (
                StableDiffusionControlNetInpaintPipeline,
                ControlNetModel,
                DEISMultistepScheduler
            )
            
            # Ensure models are downloaded
            if not ModelManager.ensure_models_downloaded():
                return False
            
            # Load BiSeNet
            with st.spinner("Loading face parser..."):
                _self.face_parser = FaceParser("models/79999_iter.pth", str(_self.device))
                if not _self.face_parser.load_model():
                    st.warning("Face parser not available - using basic masking")
            
            # Load ControlNet
            with st.spinner("Loading ControlNet..."):
                _self.controlnet = ControlNetModel.from_pretrained(
                    "thibaud/controlnet-sd21-lineart-diffusers",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            
            # Load Stable Diffusion Pipeline
            with st.spinner("Loading Stable Diffusion..."):
                _self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-2-inpainting",
                    controlnet=_self.controlnet,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                
                _self.pipe.scheduler = DEISMultistepScheduler.from_config(
                    _self.pipe.scheduler.config
                )
                _self.pipe = _self.pipe.to(_self.device)
            
            # Apply LoRA weights
            with st.spinner("Applying LoRA weights for enhanced quality..."):
                lora_applied = LoRALoader.apply_lora_weights(
                    _self.pipe, 
                    "models/to8contrast.safetensors",
                    alpha=0.65
                )
                if not lora_applied:
                    st.warning("LoRA weights not applied - using base model")
            
            st.success("✅ All models loaded successfully!")
            return True
            
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False
    
    def generate_mask_smart(self, image: np.ndarray, selection_type: str, coords: List[Tuple[int, int]] = None) -> np.ndarray:
        """Generate mask using face parsing or manual selection"""
        h, w = image.shape[:2]
        
        if self.face_parser and self.face_parser.model:
            # Use BiSeNet parsing
            parsing = self.face_parser.parse_face(Image.fromarray(image))
            parsing = cv2.resize(parsing.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Map selection type to parsing labels
            label_map = {
                "eyes": [4, 5],  # left eye, right eye
                "hair": [17],
                "lips": [12, 13],  # upper lip, lower lip
                "skin": [1, 14],  # face, neck
                "nose": [10],
                "eyebrows": [2, 3]  # left brow, right brow
            }
            
            if selection_type in label_map:
                for label in label_map[selection_type]:
                    mask[parsing == label] = 255
            
            return mask
        else:
            # Fallback to simple mask
            return self.generate_simple_mask(image, coords)
    
    def generate_simple_mask(self, image: np.ndarray, coords: List[Tuple[int, int]]) -> np.ndarray:
        """Generate simple circular mask from coordinates"""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if coords:
            for x, y in coords:
                cv2.circle(mask, (x, y), 50, 255, -1)
            
            # Smooth the mask
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return mask
    
    def edit_face(self, image: np.ndarray, mask: np.ndarray, prompt: str,
                 negative_prompt: str = "", num_steps: int = 20,
                 guidance_scale: float = 7.5) -> np.ndarray:
        """Apply face edits with LoRA-enhanced model"""
        
        # Prepare images
        h, w = image.shape[:2]
        proc_size = (768, 768)
        
        # Resize for processing
        image_resized = cv2.resize(image, proc_size)
        mask_resized = cv2.resize(mask, proc_size)
        
        # Generate Canny edge control image
        edges = cv2.Canny(image_resized, 100, 200)
        edges_rgb = np.stack([edges] * 3, axis=-1)
        
        # Convert to PIL
        image_pil = Image.fromarray(image_resized)
        mask_pil = Image.fromarray(mask_resized)
        control_image = Image.fromarray(edges_rgb)
        
        # Generate with progress tracking
        progress_bar = st.progress(0, text="Generating enhanced result...")
        
        # Add LoRA-specific prompt enhancements
        enhanced_prompt = f"{prompt}, to8contrast style, ultraskin, high quality, detailed"
        enhanced_negative = f"{negative_prompt}, bad anatomy, worst quality, low quality"
        
        with torch.no_grad():
            result = self.pipe(
                prompt=enhanced_prompt,
                negative_prompt=enhanced_negative,
                image=image_pil,
                mask_image=mask_pil,
                control_image=control_image,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=0.15,
                generator=torch.Generator(self.device).manual_seed(42)
            ).images[0]
        
        progress_bar.progress(100, text="✨ Complete!")
        
        # Resize back to original
        result = np.array(result)
        result = cv2.resize(result, (w, h))
        
        return result

# ================== Streamlit UI ==================
def init_session_state():
    """Initialize session state"""
    defaults = {
        'original_image': None,
        'current_mask': None,
        'edited_image': None,
        'editor': None,
        'models_loaded': False,
        'selection_type': 'manual',
        'edit_history': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def render_sidebar():
    """Render sidebar controls"""
    st.sidebar.header("🎨 Edit Controls")
    
    # Predefined edits
    st.sidebar.subheader("Quick Edits")
    
    edit_presets = {
        "👁️ Ocean Blue Eyes": "ocean blue eyes, bright blue iris, detailed eyes",
        "👁️ Emerald Green Eyes": "deep green eyes, emerald iris, beautiful eyes",
        "👁️ Warm Brown Eyes": "warm brown eyes, chocolate iris, gentle eyes",
        "💇 Golden Blonde Hair": "golden blonde hair, silky, healthy shine",
        "💇 Jet Black Hair": "jet black hair, glossy, luxurious",
        "💇 Auburn Red Hair": "auburn red hair, copper tones, vibrant",
        "🧔 Light Stubble": "light beard stubble, masculine, well-groomed",
        "😊 Perfect Smile": "beautiful smile, perfect teeth, happy expression",
        "✨ Clear Skin": "clear smooth skin, healthy glow, even tone"
    }
    
    selected_preset = st.sidebar.selectbox("Choose a preset:", ["Custom"] + list(edit_presets.keys()))
    
    if selected_preset != "Custom":
        prompt = edit_presets[selected_preset]
    else:
        prompt = st.sidebar.text_area(
            "Custom Prompt",
            placeholder="Describe the changes you want...",
            height=100
        )
    
    negative_prompt = st.sidebar.text_area(
        "Negative Prompt",
        value="monochrome, bad anatomy, worst quality, low quality, bad teeth, bad eyes",
        height=80
    )
    
    # Advanced settings
    with st.sidebar.expander("⚙️ Advanced Settings"):
        num_steps = st.slider("Inference Steps", 10, 50, 20)
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5)
        controlnet_scale = st.slider("ControlNet Scale", 0.0, 1.0, 0.15)
    
    # Mask selection type
    st.sidebar.subheader("Mask Selection")
    selection_type = st.sidebar.radio(
        "Selection Method",
        ["Smart (Face Parts)", "Manual Draw"],
        help="Smart selection uses AI to detect face parts"
    )
    
    if selection_type == "Smart (Face Parts)":
        face_part = st.sidebar.selectbox(
            "Select Face Part",
            ["eyes", "hair", "lips", "skin", "nose", "eyebrows"]
        )
    else:
        face_part = None
    
    return prompt, negative_prompt, num_steps, guidance_scale, face_part

def main():
    """Main application"""
    init_session_state()
    
    st.title("🎨 AI Face Editor Pro")
    st.markdown("**Enhanced with Custom LoRA for Superior Quality**")
    
    # Load models once
    if not st.session_state.models_loaded:
        with st.container():
            st.info("🚀 Initializing AI models with LoRA enhancement...")
            editor = FaceEditorPro()
            if editor.load_models():
                st.session_state.editor = editor
                st.session_state.models_loaded = True
            else:
                st.error("Failed to load models. Please check your setup.")
                st.stop()
    
    # Get sidebar controls
    prompt, negative_prompt, num_steps, guidance_scale, face_part = render_sidebar()
    
    # Main interface
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("📸 Original")
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=['png', 'jpg', 'jpeg'],
            help="Best results with high-quality portraits"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            image = np.array(image.convert('RGB'))
            st.session_state.original_image = image
            st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("🎭 Mask")
        
        if st.session_state.original_image is not None:
            if face_part:  # Smart selection
                if st.button(f"Select {face_part.title()}", use_container_width=True):
                    mask = st.session_state.editor.generate_mask_smart(
                        st.session_state.original_image,
                        face_part
                    )
                    st.session_state.current_mask = mask
            else:  # Manual selection
                st.info("Draw on the image to create mask")
                # Note: In production, you'd use streamlit-drawable-canvas here
                # For simplicity, using a placeholder
                if st.button("Generate Sample Mask", use_container_width=True):
                    # Generate a sample mask for demo
                    h, w = st.session_state.original_image.shape[:2]
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.circle(mask, (w//2, h//3), 50, 255, -1)  # Sample eye region
                    st.session_state.current_mask = mask
            
            if st.session_state.current_mask is not None:
                # Visualize mask
                mask_viz = st.session_state.original_image.copy()
                mask_viz[st.session_state.current_mask > 0] = [255, 0, 0]
                st.image(mask_viz, use_column_width=True)
    
    with col3:
        st.subheader("✨ Result")
        
        if st.button("🎨 Apply Edit", type="primary", use_container_width=True):
            if st.session_state.original_image is None:
                st.error("Please upload an image!")
            elif st.session_state.current_mask is None:
                st.error("Please create a mask!")
            elif not prompt:
                st.error("Please enter a prompt!")
            else:
                with st.spinner("Creating magic... ✨"):
                    result = st.session_state.editor.edit_face(
                        st.session_state.original_image,
                        st.session_state.current_mask,
                        prompt,
                        negative_prompt,
                        num_steps,
                        guidance_scale
                    )
                    st.session_state.edited_image = result
                    st.session_state.edit_history.append(result)
                    st.balloons()
        
        if st.session_state.edited_image is not None:
            st.image(st.session_state.edited_image, use_column_width=True)
            
            # Download button
            result_pil = Image.fromarray(st.session_state.edited_image)
            buf = io.BytesIO()
            result_pil.save(buf, format='PNG')
            
            st.download_button(
                "💾 Download",
                data=buf.getvalue(),
                file_name="edited_face.png",
                mime="image/png",
                use_container_width=True
            )
    
    # History
    if st.session_state.edit_history:
        st.divider()
        st.subheader("📜 Edit History")
        cols = st.columns(len(st.session_state.edit_history[-5:]))  # Show last 5
        for i, (col, img) in enumerate(zip(cols, st.session_state.edit_history[-5:])):
            with col:
                st.image(img, caption=f"Edit {i+1}", use_column_width=True)

if __name__ == "__main__":
    main()