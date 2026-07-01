from app import describe_prediction


def test_describe_prediction_for_pneumonia():
    text = describe_prediction(1)
    assert 'Pneumonia' in text
    assert 'medical care' in text.lower()


def test_describe_prediction_for_normal():
    text = describe_prediction(0)
    assert 'Normal' in text
    assert 'no pneumonia' in text.lower()
