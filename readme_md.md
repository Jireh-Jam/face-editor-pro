# 🎨 AI Face Editor Pro

An advanced face editing application powered by Stable Diffusion, ControlNet, and custom LoRA weights for high-quality facial feature manipulation.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.29.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- **Smart Face Parsing**: AI-powered face segmentation using BiSeNet
- **Custom LoRA Enhancement**: Fine-tuned weights for superior portrait quality
- **ControlNet Integration**: Precise editing with edge-guided generation
- **Interactive UI**: User-friendly Streamlit interface
- **Multiple Edit Options**:
  - Eye color modification
  - Hair style and color changes
  - Facial hair editing
  - Skin texture improvements
  - Expression changes

## 🚀 Live Demo

[Try it on Streamlit Cloud](https://your-app-name.streamlit.app) *(Update after deployment)*

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- At least 8GB RAM (16GB recommended)
- 5GB free disk space for models

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/face-editor-pro.git
cd face-editor-pro
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Model Weights

The app will automatically download required models on first run. Alternatively, you can manually download:

1. **BiSeNet Model** (`79999_iter.pth`): [Download Link](your-link)
2. **LoRA Weights** (`to8contrast.safetensors`): [Download Link](your-link)

Place them in the `models/` directory.

### 5. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
face-editor-pro/
├── app.py                 # Main Streamlit application
├── model.py              # BiSeNet model architecture
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore file
├── README.md            # This file
├── models/              # Model weights directory
│   ├── 79999_iter.pth
│   └── to8contrast.safetensors
├── examples/            # Sample images
│   └── portrait1.jpg
└── .streamlit/          # Streamlit configuration
    └── config.toml
```

## 🎯 Usage

1. **Upload Image**: Select a portrait photo
2. **Select Region**: Choose face parts to edit (eyes, hair, etc.)
3. **Choose Edit**: Pick from presets or write custom prompt
4. **Apply**: Click "Apply Edit" to generate
5. **Download**: Save your edited image

### Example Prompts

- **Eyes**: "ocean blue eyes, bright iris, detailed"
- **Hair**: "golden blonde hair, wavy, healthy shine"
- **Skin**: "clear smooth skin, even tone, healthy glow"
- **Expression**: "warm smile, happy expression"

## ⚙️ Configuration

### Streamlit Config (`.streamlit/config.toml`)

```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 10
enableCORS = false
enableXsrfProtection = true
```

### Environment Variables (`.env`)

```env
# GPU Settings
CUDA_VISIBLE_DEVICES=0

# Model Paths (optional)
BISENET_MODEL_PATH=models/79999_iter.pth
LORA_MODEL_PATH=models/to8contrast.safetensors

# API Keys (if using external services)
# HUGGINGFACE_TOKEN=your_token_here
```

## 🚀 Deployment to Streamlit Cloud

### Step 1: Prepare Repository

1. Create a new GitHub repository
2. Push your code:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/face-editor-pro.git
git push -u origin main
```

### Step 2: Configure Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub account
4. Select your repository
5. Set main file path: `app.py`
6. Click "Deploy"

### Step 3: Add Secrets (if needed)

In Streamlit Cloud settings, add secrets:

```toml
[models]
bisenet_url = "your-model-url"
lora_url = "your-lora-url"
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce image size
   - Lower inference steps
   - Use CPU mode

2. **Model Download Fails**
   - Check internet connection
   - Download manually and place in `models/`

3. **Slow Performance**
   - Use GPU if available
   - Reduce image resolution
   - Lower inference steps

## 📊 Performance

| Hardware | Inference Time | Quality |
|----------|---------------|---------|
| RTX 3090 | ~5 seconds    | High    |
| RTX 3060 | ~10 seconds   | High    |
| CPU only | ~60 seconds   | Medium  |

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [BiSeNet](https://github.com/zllrunning/face-parsing.PyTorch)
- [Streamlit](https://streamlit.io)

## 📧 Contact

For questions or support, please open an issue or contact [your-email@example.com]

---

**Made with ❤️ by Jireh Jam**
