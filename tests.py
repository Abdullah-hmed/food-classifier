from app import get_usda_nutrition, classification
from PIL import Image
import pytest, os
import requests

def test_get_usda_api_success():
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

@pytest.fixture(scope="module")
def usda_api_key():
    return os.getenv("USDA_API_KEY")

def test_usda_api(usda_api_key):
    
    # api key check
    assert usda_api_key is not None, "no API key found"

    food_name = "apple pie"
    api_url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={food_name}&api_key={usda_api_key}&pageSize=1"

    response = requests.get(api_url)
    assert response.status_code == 200, "API didn't successfully respond"

    data = response.json()

    # Verify the response contains the expected structure
    assert len(data["foods"]) > 0, "No food returned by API"
