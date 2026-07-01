import os
from pathlib import Path

import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image

from xray.ml.model.arch import Net

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "xray_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

LABEL_MAP = {
    0: "Normal",
    1: "Pneumonia",
}


def load_model():
    global MODEL
    if MODEL is None:
        MODEL = Net().to(DEVICE)
        MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        MODEL.eval()
    return MODEL


def describe_prediction(prediction_index: int) -> str:
    if prediction_index == 1:
        return (
            "Pneumonia is likely. This result is a screening aid, not a diagnosis.\n"
            "What to do next: seek medical care promptly, rest, stay hydrated, and follow your clinician's guidance.\n"
            "Suggested care: a clinician may recommend antibiotics or further evaluation depending on symptoms and medical history."
        )
    return (
        "Normal / no pneumonia detected. This X-ray does not appear to show pneumonia.\n"
        "What to do next: if symptoms continue, consult a doctor for a full evaluation.\n"
        "Suggested care: maintain hydration, rest, and avoid respiratory irritants."
    )


def predict_image(image):
    if image is None:
        return "Please upload a chest X-ray image first."

    load_model()
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    else:
        image = Image.open(image).convert("RGB")

    input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = MODEL(input_tensor)
        prediction_index = int(torch.argmax(output, dim=1).item())

    label = LABEL_MAP.get(prediction_index, "Unknown")
    details = describe_prediction(prediction_index)
    return f"Prediction: {label}\n\n{details}"


iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload chest X-ray"),
    outputs=gr.Textbox(label="Result"),
    title="Lung Disease Classifier",
    description="Upload a chest X-ray image to get a quick pneumonia screening result and basic guidance.",
    examples=[[str(ROOT_DIR / "sample_xray.png")]] if (ROOT_DIR / "sample_xray.png").exists() else None,
)

app = iface.launch


if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
