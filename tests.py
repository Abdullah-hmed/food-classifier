from app import get_usda_nutrition, classification
from PIL import Image

def test_get_usda_nutrition_success():
    result = get_usda_nutrition("apple")
    expected_result = {
        'description': 'APPLE', 
        'calories': 52.0, 
        'protein': 0.0, 
        'fat': 0.65, 
        'carbs': 14.3
    }
    assert result == expected_result


def test_classification():
    image = Image.open('apple-pie.jpg')
    food_name, confidence = classification(image)
    assert food_name == 'Apple pie'

