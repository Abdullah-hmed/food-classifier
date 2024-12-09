from flask import Flask, render_template, request, jsonify
import requests, os, json
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
import io
import gdown
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
app = Flask(__name__)
class_names = io.open('labels.txt', 'r').read().split('\n')


def model_available():
    models_dir = 'model'
    model_file = 'improved_best_model.pth'
    drive_link = 'https://drive.google.com/uc?export=download&id=1mFLtnUbKCMJpULPvurGoE1jix3U__eaM'

    # Ensure the models directory exists
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Check if the model file exists
    model_path = os.path.join(models_dir, model_file)
    if os.path.exists(model_path):
        print("Model present, continue :)")
        return True
    else:
        print("Model not present, downloading...")

        try:
            gdown.download(drive_link, model_path, quiet=False)
            print(f"{model_file} downloaded and saved to {models_dir} folder.")
            return True
        except Exception as e:
            print(f"Failed to download {model_file} from Google Drive. Error: {str(e)}")
            return False


if  model_available():
    print("Model Loaded, Continuing.")
else:
    print("Exiting the application: Model Not Available.")
    exit(1)


def setup_model():
    torch.cuda.empty_cache()    
    model = models.inception_v3(pretrained=True)
    num_classes = 101
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    num_ftrs_aux = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(num_ftrs_aux, num_classes)
    return model.to(device)

model = setup_model()
# Load the checkpoint
checkpoint = torch.load('model/improved_best_model.pth', weights_only=True)
# Restore model state
model.load_state_dict(checkpoint['model_state_dict'])

@app.route('/')
def index():
    return render_template('index.html', output="Confidence")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image_file = request.files['image']
        
        if image_file:
            image = Image.open(image_file)
            food_name, confidence = classification(image)
            nutrition_facts = get_usda_nutrition(food_name)
            nutrition_facts['query'] = food_name
            nutrition_facts['confidence'] = confidence
            print(food_name,':', confidence)
            print(nutrition_facts)
            return nutrition_facts
    

def get_usda_nutrition(food_name):
    api_key = os.getenv("USDA_API_KEY")
    api_url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={food_name}&api_key={api_key}&pageSize=1"
    
    try:
        response = requests.get(api_url)
        response.raise_for_status() 
        data = response.json()
        
        if 'foods' in data and len(data['foods']) > 0:
            food = data['foods'][0]  # Get the first food result
            food_nutrients = food.get("foodNutrients", [])
            
            calories = protein = fat = carbs = "N/A"
            
            for nutrient in food_nutrients:
                if nutrient.get("nutrientName") == "Energy":
                    calories = nutrient.get("value", "N/A")
                elif nutrient.get("nutrientName") == "Protein":
                    protein = nutrient.get("value", "N/A")
                elif nutrient.get("nutrientName") == "Total lipid (fat)":
                    fat = nutrient.get("value", "N/A")
                elif nutrient.get("nutrientName") == "Carbohydrate, by difference":
                    carbs = nutrient.get("value", "N/A")
            
            return {
                "description": food.get("description", "N/A"),
                "calories": calories,
                "protein": protein,
                "fat": fat,
                "carbs": carbs
            }
        else:
            return {"error": "Food not found"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def classification(image):
    val_transforms = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(299),  # Keep original size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    image_tensor = val_transforms(image).to(device)
    
     
    
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)
        return class_names[predicted.item()], confidence.item()
    

if __name__ == '__main__':
    app.run(debug=True)