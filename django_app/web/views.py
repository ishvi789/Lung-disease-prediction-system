import os
import torch
import torchvision.transforms as transforms
from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from PIL import Image
from xray.ml.model.arch import Net

MODEL_PATH = os.path.join(settings.BASE_DIR, '..', 'xray_model.pth')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = None

LABEL_MAP = {
    0: 'Normal',
    1: 'Pneumonia'
}


def load_model():
    global MODEL
    if MODEL is None:
        MODEL = Net().to(DEVICE)
        MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        MODEL.eval()


def index(request):
    return render(request, 'index.html')


def build_result_details(prediction_index):
    prediction_label = LABEL_MAP.get(prediction_index, 'Unknown')

    if prediction_label == 'Pneumonia':
        return {
            'prediction_index': prediction_index,
            'prediction_label': prediction_label,
            'status': 'Pneumonia is likely',
            'message': 'This X-ray appears to show signs consistent with pneumonia.',
            'suggestion': 'Please seek medical care promptly, rest, stay hydrated, and follow your clinician\'s treatment plan.',
            'prevention': 'Avoid smoke exposure, keep the room ventilated, wash hands regularly, and rest well.',
            'care': 'A clinician may recommend antibiotics or other treatment depending on symptoms and medical history.'
        }

    return {
        'prediction_index': prediction_index,
        'prediction_label': prediction_label,
        'status': 'No pneumonia detected',
        'message': 'This X-ray does not appear to show pneumonia.',
        'suggestion': 'If you still have symptoms, consult a doctor for a full evaluation.',
        'prevention': 'Maintain good hydration, rest, and avoid respiratory irritants.',
        'care': 'No urgent medication is indicated by this prediction alone, but follow-up care is still wise if symptoms continue.'
    }


def predict(request):
    load_model()
    if request.method == 'POST' and request.FILES.get('file'):
        image = Image.open(request.FILES['file']).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = MODEL(input_tensor)
            prediction_index = torch.argmax(output, dim=1).item()

        return JsonResponse(build_result_details(prediction_index))
    return JsonResponse({'error': 'No file uploaded'}, status=400)
