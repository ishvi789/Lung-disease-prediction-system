import numpy as np
from PIL import Image

from app import describe_prediction, predict_image


def test_describe_prediction_for_pneumonia():
    text = describe_prediction(1)
    assert 'Pneumonia' in text
    assert 'medical care' in text.lower()


def test_describe_prediction_for_normal():
    text = describe_prediction(0)
    assert 'Normal' in text
    assert 'no pneumonia' in text.lower()


def test_predict_image_returns_readable_result_for_pil_image():
    image = Image.new('RGB', (224, 224), color='white')
    result = predict_image(image)
    assert 'Prediction:' in result
    assert 'What to do next' in result


def test_predict_image_accepts_numpy_array_uploads():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    result = predict_image(image)
    assert 'Prediction:' in result
    assert 'What to do next' in result
