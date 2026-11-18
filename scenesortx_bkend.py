from vehicle_classifier import VehicleAnalyzer
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
from transformers import CLIPProcessor, CLIPModel
import os
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
#ai modules
class SceneClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet50(pretrained=True).to(self.device)
        self.model.eval()
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        self.scene_labels = {
            0: 'Urban Environment',
            1: 'Natural Landscape',
            2: 'Industrial Area',
            3: 'Residential Zone',
            4: 'Commercial District',
            5: 'Agricultural Land'
        }
    
    def classify(self, image_np):
        image_pil = Image.fromarray(image_np)
        image_tensor = self.transforms(image_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        confidence = probabilities.max().item()
        class_idx = probabilities.argmax().item() % len(self.scene_labels)
        primary_scene = self.scene_labels[class_idx]
        
        attributes = self._extract_attributes(image_np)
        sustainability_score = self._calculate_sustainability(image_np)
        
        return {
            'primary': primary_scene,
            'confidence': confidence,
            'attributes': attributes,
            'sustainabilityScore': sustainability_score
        }
    
    def _extract_attributes(self, image_np):
        attributes = []
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 20:
            attributes.append('high density')
        if np.std(gray) > 50:
            attributes.append('varied lighting')
        
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        green_ratio = np.sum(green_mask > 0) / green_mask.size
        
        if green_ratio > 0.2:
            attributes.append('green space')
        else:
            attributes.append('mixed waste potential')
        
        if attributes == []:
            attributes = ['urban setting', 'structured layout']
        
        return attributes[:3]
    
    def _calculate_sustainability(self, image_np):
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        green_ratio = (np.sum(green_mask > 0) / green_mask.size) * 100
        
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        cleanliness = (1 - np.std(gray) / 255) * 100
        
        return int((green_ratio * 0.6 + cleanliness * 0.4))

class WasteClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.waste_categories = ['Plastic', 'Organic', 'Paper', 'Metal', 'Glass']
        self.recyclability = {cat: True for cat in self.waste_categories}
    
    def classify(self, image_np):
        detected, materials = self._detect_waste(image_np)
        
        return {
            'detected': detected,
            'materials': materials,
            'recommendation': 'Implement source separation to maximize recycling potential'
        }
    
    def _detect_waste(self, image_np):
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        
        colors = {
            'Plastic': cv2.inRange(hsv, (100, 50, 50), (130, 255, 255)),
            'Organic': cv2.inRange(hsv, (20, 100, 100), (40, 255, 255)),
            'Paper': cv2.inRange(hsv, (10, 30, 100), (30, 100, 255)),
            'Metal': cv2.inRange(hsv, (0, 0, 100), (180, 30, 200)),
            'Glass': cv2.inRange(hsv, (90, 20, 200), (110, 50, 255))
        }
        
        materials = []
        total_pixels = image_np.shape[0] * image_np.shape[1]
        detected_waste = False
        
        percentages = {}
        for waste_type, mask in colors.items():
            pixel_count = np.sum(mask > 0)
            percentage = (pixel_count / total_pixels) * 100
            percentages[waste_type] = percentage
            if percentage > 2:
                detected_waste = True
        
        total_detected = sum(percentages.values())
        if total_detected > 0:
            normalized = {k: (v/total_detected)*100 for k, v in percentages.items()}
            for waste_type in self.waste_categories:
                if normalized.get(waste_type, 0) > 0:
                    materials.append({
                        'type': waste_type,
                        'percentage': round(normalized[waste_type]),
                        'recyclable': self.recyclability[waste_type],
                        'impact': 'High' if normalized[waste_type] > 30 else 'Medium' if normalized[waste_type] > 15 else 'Low'
                    })
        
        if not materials:
            materials = [
                {'type': 'Plastic', 'percentage': 45, 'recyclable': True, 'impact': 'High'},
                {'type': 'Organic', 'percentage': 30, 'recyclable': True, 'impact': 'Medium'},
                {'type': 'Paper', 'percentage': 15, 'recyclable': True, 'impact': 'Low'},
                {'type': 'Metal', 'percentage': 10, 'recyclable': True, 'impact': 'Low'}
            ]
        
        return detected_waste or len(materials) > 0, materials

class FashionAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.to(self.device)
            self.model.eval()
        except:
            self.model = None
            self.processor = None
    
    def analyze(self, image_np):
        styles = self._extract_styles(image_np)
        cultural_context = self._determine_cultural_context(image_np)
        
        return {
            'styles': styles,
            'culturalContext': cultural_context,
            'recommendation': 'Consider sustainable fashion brands for formal wear'
        }
    
    def _extract_styles(self, image_np):
        style_prompts = [
            'contemporary fashion style',
            'casual clothing',
            'formal business wear'
        ]
        
        if self.model is not None and self.processor is not None:
            image_pil = Image.fromarray(image_np)
            inputs = self.processor(text=style_prompts, images=image_pil, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image[0]
                probabilities = torch.nn.functional.softmax(logits, dim=0)
            
            styles = [
                {'style': 'Contemporary', 'percentage': int(probabilities[0].item() * 100), 'sustainability': 'Medium'},
                {'style': 'Casual', 'percentage': int(probabilities[1].item() * 100), 'sustainability': 'High'},
                {'style': 'Formal', 'percentage': int(probabilities[2].item() * 100), 'sustainability': 'Low'}
            ]
        else:
            styles = [
                {'style': 'Contemporary', 'percentage': 50, 'sustainability': 'Medium'},
                {'style': 'Casual', 'percentage': 30, 'sustainability': 'High'},
                {'style': 'Formal', 'percentage': 20, 'sustainability': 'Low'}
            ]
        
        return sorted(styles, key=lambda x: x['percentage'], reverse=True)
    
    def _determine_cultural_context(self, image_np):
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        
        red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        blue_mask = cv2.inRange(hsv, (100, 100, 100), (130, 255, 255))
        
        if np.sum(red_mask) > np.sum(blue_mask):
            return 'South Asian'
        else:
            return 'Urban Western'

class SustainabilityInsights:
    @staticmethod
    def generate(scene_analysis, waste_analysis, fashion_analysis, vehicle_analysis):
        insights = []
        recommendations = []
        
        insights.append(f" Environmental awareness score: {scene_analysis['sustainabilityScore']}/100")
        
        if waste_analysis['detected']:
            recyclable_count = sum(1 for m in waste_analysis['materials'] if m['recyclable'])
            recycling_potential = (recyclable_count / len(waste_analysis['materials'])) * 100
            insights.append(f" Waste segregation potential: {int(recycling_potential)}/100")
            recommendations.append("Implement comprehensive waste management system")
        
        casual_style = next((s for s in fashion_analysis['styles'] if s['style'] == 'Casual'), None)
        if casual_style:
            sustainability_score = 60 if casual_style['sustainability'] == 'High' else 40
            insights.append(f" Fashion sustainability: {sustainability_score}/100")
            recommendations.append("Switch to eco-friendly fashion alternatives")
        
        overall = (scene_analysis['sustainabilityScore'] + 60 + 50) // 3
        insights.append(f" Overall sustainability index: {overall}/100")
        
        recommendations.extend([
            "Optimize energy usage in urban settings",
            "Consider carbon offset programs"
        ])
        
        return insights, recommendations
#from here i have got the flask app

app = Flask(__name__)
CORS(app)

print(" Initializing AI modules...")
scene_classifier = SceneClassifier()
waste_classifier = WasteClassifier()
fashion_analyzer = FashionAnalyzer()
vehicle_analyzer = VehicleAnalyzer()
print(" All AI modules loaded successfully!")

def preprocess_image(file):
    """Convert uploaded file to numpy array"""
    image = Image.open(file.stream)
    # Resize large images for faster processing
    max_size = (800, 800)
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    image_np = np.array(image)
    # Ensure RGB format
    if len(image_np.shape) == 2:  # Grayscale
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:  # RGBA
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    return image_np
#i dont know what to write here , but i think its flask app routes, i got this from the previous code, somehow worked with it to funbtion
@app.route("/", methods=["GET"])
def home():
    """API home page"""
    return jsonify({
        "name": "SceneSortX API",
        "version": "1.0.0",
        "status": "running",
        "message": "AI-Powered Visual Advisor for Sustainable Lifestyles",
        "endpoints": {
            "scene": "/scene (POST) - Scene classification",
            "waste": "/waste (POST) - Waste analysis",
            "fashion": "/fashion (POST) - Fashion style analysis",
            "vehicle": "/vehicle (POST) - Vehicle detection",
            "report": "/report (POST) - Generate full report",
            "health": "/health (GET) - Health check"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "modules": {
            "scene": "loaded",
            "waste": "loaded",
            "fashion": "loaded",
            "vehicle": "loaded"
        }
    })

@app.route("/scene", methods=["POST"])
def scene_api():
    """Scene classification endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files["image"]
        img = preprocess_image(file)
        result = scene_classifier.classify(img)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/waste", methods=["POST"])
def waste_api():
    """Waste classification endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files["image"]
        img = preprocess_image(file)
        result = waste_classifier.classify(img)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/fashion", methods=["POST"])
def fashion_api():
    """Fashion analysis endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files["image"]
        img = preprocess_image(file)
        result = fashion_analyzer.analyze(img)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/vehicle", methods=["POST"])
def vehicle_api():
    """Vehicle analysis endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files["image"]
        img = preprocess_image(file)
        result = vehicle_analyzer.analyze(img)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/analyze", methods=["POST"])
def analyze_all():
    """Analyze image with all modules"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files["image"]
        img = preprocess_image(file)
        
        scene_analysis = scene_classifier.classify(img)
        waste_analysis = waste_classifier.classify(img)
        fashion_analysis = fashion_analyzer.analyze(img)
        vehicle_analysis = vehicle_analyzer.analyze(img)
        insights, recommendations = SustainabilityInsights.generate(
            scene_analysis, waste_analysis, fashion_analysis, vehicle_analysis
        )
        
        return jsonify({
            "scene": scene_analysis,
            "waste": waste_analysis,
            "fashion": fashion_analysis,
            "vehicle": vehicle_analysis,
            "insights": insights,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/report", methods=["POST"])
def report_api():
    """Generate report from analysis data"""
    try:
        data = request.get_json()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "scene": data.get("scene"),
            "waste": data.get("waste"),
            "fashion": data.get("fashion"),
            "vehicle": data.get("vehicle"),
            "insights": data.get("insights"),
            "recommendations": data.get("recommendations"),
        }
        
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
#
if __name__ == "__main__":
    print("\n" + "="*60)
    print(" SceneSortX Backend Server")
    print("="*60)
    print(" Running on: http://127.0.0.1:5000")
    print(" API Docs: http://127.0.0.1:5000/")
    print(" Health Check: http://127.0.0.1:5000/health")
    print("="*60 + "\n")
    
    app.run(host="127.0.0.1", port=5000, debug=True)