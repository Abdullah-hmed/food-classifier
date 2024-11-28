from flask import Flask, render_template, request, jsonify
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
import io
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
app = Flask(__name__)
class_names = io.open('model/labels.txt', 'r').read().split('\n')

def setup_model():
    torch.cuda.empty_cache()    
    model = models.inception_v3(pretrained=True)
    # Keep auxiliary classifier
    num_classes = 101
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    num_ftrs_aux = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(num_ftrs_aux, num_classes)
    return model.to(device)

model = setup_model()
# Load the checkpoint
checkpoint = torch.load('model/best_model.pth')
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
            image_class, confidence = classification(image)
            data = f'class: {image_class}, confidence: {confidence:.2f}'
            return data
    


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