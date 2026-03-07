import os
import io
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import gradio as gr
import numpy as np

# ─── Config ───────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food',
    'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit'
]

CLASS_EMOJIS = {
    'Bread': '🍞', 'Dairy product': '🧀', 'Dessert': '🍰',
    'Egg': '🥚', 'Fried food': '🍟', 'Meat': '🥩',
    'Noodles/Pasta': '🍝', 'Rice': '🍚', 'Seafood': '🦐',
    'Soup': '🍲', 'Vegetable/Fruit': '🥗'
}

# ─── Model ────────────────────────────────────────────────────────────────────
def load_model():
    model = models.mobilenet_v3_small(pretrained=False)
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, 11)
    )
    # Load weights if saved model exists
    if os.path.exists("model.pth"):
        state_dict = torch.load("model.pth", map_location="cpu")
        model.load_state_dict(state_dict)
        print("✅ Loaded saved weights")
    else:
        print("⚠️  No model.pth found — using random weights (for demo only)")
    model.eval()
    return model

model = load_model()

# ─── Transform ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(112),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ─── Inference ────────────────────────────────────────────────────────────────
def predict(image):
    if image is None:
        return {name: 0.0 for name in CLASS_NAMES}

    img = Image.fromarray(image).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]

    results = {}
    for i, name in enumerate(CLASS_NAMES):
        label = f"{CLASS_EMOJIS[name]} {name}"
        results[label] = float(probs[i])

    return results

# ─── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Food-11 Classifier",
    theme=gr.themes.Soft(),
    css="""
        .gradio-container { max-width: 800px !important; margin: auto; }
        footer { display: none !important; }
    """
) as demo:
    gr.Markdown("""
    # 🍽️ Food-11 Classifier
    **Built by [Harsh](https://harshgiri.site) · MobileNetV3 · Transfer Learning**

    Upload a food photo and the model will classify it into one of 11 categories.
    """)

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="Upload Food Image", type="numpy", height=300)
            btn = gr.Button("🔍 Classify", variant="primary", size="lg")

        with gr.Column():
            label_output = gr.Label(label="Predictions", num_top_classes=5)

    btn.click(fn=predict, inputs=img_input, outputs=label_output)
    img_input.change(fn=predict, inputs=img_input, outputs=label_output)

    gr.Examples(
        examples=[],
        inputs=img_input
    )

demo.launch()
