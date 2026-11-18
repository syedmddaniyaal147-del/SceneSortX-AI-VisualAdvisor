"""
Vehicle Classifier using YOLOv8 + CLIP
Author: Mohammed Hassan (4MH23CA030)
"""

from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

class VehicleAnalyzer:
    """
    Main Vehicle Analyzer using YOLOv8 for detection + CLIP for classification
    """
    
    def __init__(self):
        print("🚀 Initializing Vehicle Analyzer...")
        
        # Load YOLOv8 model
        print("📥 Loading YOLOv8 model...")
        self.yolo_model = YOLO('yolov8n.pt')  # Downloads automatically on first run
        print("✅ YOLOv8 loaded!")
        
        # Load CLIP model for brand classification
        print("📥 Loading CLIP model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.to(self.device)
        self.clip_model.eval()
        print("✅ CLIP loaded!")
        
        # Indian Car Brands
        self.car_brands = [
            "Maruti Suzuki Swift hatchback",
            "Maruti Suzuki Dzire sedan",
            "Maruti Suzuki Brezza SUV",
            "Maruti Suzuki WagonR hatchback",
            "Maruti Suzuki Baleno hatchback",
            "Hyundai Creta SUV",
            "Hyundai Venue SUV",
            "Hyundai i20 hatchback",
            "Hyundai Verna sedan",
            "Tata Nexon SUV",
            "Tata Punch SUV",
            "Tata Harrier SUV",
            "Tata Altroz hatchback",
            "Mahindra XUV700 SUV",
            "Mahindra Thar SUV",
            "Mahindra Scorpio SUV",
            "Honda City sedan",
            "Honda Amaze sedan",
            "Toyota Fortuner SUV",
            "Toyota Innova MPV",
            "Kia Seltos SUV",
            "Kia Sonet SUV",
            "MG Hector SUV",
            "MG Astor SUV"
        ]
        
        # Indian Bike Brands
        self.bike_brands = [
            "Royal Enfield Classic 350 motorcycle",
            "Royal Enfield Bullet motorcycle",
            "Royal Enfield Himalayan motorcycle",
            "Honda CB Shine commuter bike",
            "Honda Unicorn motorcycle",
            "Honda Hornet sport bike",
            "Yamaha FZ street bike",
            "Yamaha MT-15 naked bike",
            "Yamaha R15 sport bike",
            "Bajaj Pulsar motorcycle",
            "Bajaj Dominar touring bike",
            "Bajaj Avenger cruiser",
            "KTM Duke 200 naked bike",
            "KTM Duke 390 sport bike",
            "KTM RC 390 sport bike",
            "Hero Splendor commuter",
            "Hero Passion commuter",
            "Hero Xtreme sport bike",
            "TVS Apache sport bike",
            "TVS Jupiter scooter",
            "TVS Ntorq scooter"
        ]
        
        # Price ranges (INR)
        self.price_data = {
            'Maruti Suzuki': (300000, 1200000),
            'Hyundai': (500000, 2000000),
            'Tata': (400000, 2500000),
            'Mahindra': (700000, 3000000),
            'Honda': (600000, 1500000),
            'Toyota': (800000, 5000000),
            'Kia': (700000, 2000000),
            'MG': (1000000, 3000000),
            'Royal Enfield': (150000, 350000),
            'Yamaha': (70000, 300000),
            'Bajaj': (50000, 250000),
            'KTM': (150000, 350000),
            'Hero': (60000, 150000),
            'TVS': (60000, 180000)
        }
        
        print("✅ Vehicle Analyzer ready!\n")
    
    def analyze(self, image_np):
        """
        Main analysis method
        
        Args:
            image_np: numpy array of image (RGB format)
            
        Returns:
            dict: Complete vehicle analysis results
        """
        try:
            # Convert RGB to BGR for YOLOv8
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # STEP 1: YOLOv8 Detection
            results = self.yolo_model(image_bgr, verbose=False)
            
            # Extract detections
            detections = results[0].boxes
            
            if len(detections) == 0:
                return {
                    'detected': False,
                    'message': 'No vehicles detected in image',
                    'vehicle_type': 'None',
                    'total_vehicles': 0
                }
            
            # STEP 2: Find and process vehicles
            vehicles = self._extract_vehicles(detections, image_np)
            
            if not vehicles:
                return {
                    'detected': False,
                    'message': 'No cars or motorcycles found',
                    'vehicle_type': 'None',
                    'total_vehicles': 0
                }
            
            # Get main vehicle (largest)
            main_vehicle = max(vehicles, key=lambda x: x['area'])
            
            # STEP 3: Crop vehicle region
            vehicle_crop = self._crop_vehicle(image_np, main_vehicle['box'])
            
            # STEP 4: Classify brand using CLIP
            brand_info = self._classify_brand(vehicle_crop, main_vehicle['type'])
            
            # STEP 5: Compile final results
            result = {
                'detected': True,
                'vehicle_type': main_vehicle['type'],
                'brand': brand_info['brand'],
                'model': brand_info['model'],
                'category': brand_info['category'],
                'confidence': main_vehicle['confidence'],
                'brand_confidence': brand_info['brand_confidence'],
                'estimated_price': brand_info['price'],
                'price_formatted': brand_info['price_formatted'],
                'price_range': brand_info['price_range'],
                'total_vehicles': len(vehicles),
                'all_vehicles': [f"{v['type']} ({v['confidence']}%)" for v in vehicles],
                'detection_method': 'YOLOv8 + CLIP',
                'message': f"Detected {brand_info['brand']} {brand_info['model']}",
                'bounding_box': main_vehicle['box']
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error in analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'detected': False,
                'error': str(e),
                'message': 'Error analyzing vehicle',
                'vehicle_type': 'Error'
            }
    
    def _extract_vehicles(self, detections, image_np):
        """Extract vehicle information from YOLOv8 detections"""
        vehicles = []
        
        for box in detections:
            class_id = int(box.cls[0])
            class_name = self.yolo_model.names[class_id]
            confidence = float(box.conf[0])
            
            # COCO dataset classes: car=2, motorcycle=3, bus=5, truck=7
            if class_id in [2, 3, 5, 7]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                
                # Determine type
                if class_id == 3:  # motorcycle
                    vehicle_type = 'Motorcycle'
                else:  # car, bus, truck
                    vehicle_type = 'Car'
                
                vehicles.append({
                    'type': vehicle_type,
                    'confidence': round(confidence * 100, 2),
                    'box': [x1, y1, x2, y2],
                    'area': area,
                    'yolo_class': class_name
                })
        
        return vehicles
    
    def _crop_vehicle(self, image_np, box):
        """Crop vehicle region with padding"""
        x1, y1, x2, y2 = box
        
        # Add padding (10 pixels)
        pad = 10
        h, w = image_np.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        return image_np[y1:y2, x1:x2]
    
    def _classify_brand(self, vehicle_crop, vehicle_type):
        """Classify vehicle brand using CLIP"""
        
        # Convert to PIL Image
        image_pil = Image.fromarray(vehicle_crop)
        
        # Choose brand list
        if vehicle_type == 'Car':
            brand_options = self.car_brands
        else:
            brand_options = self.bike_brands
        
        # CLIP classification
        inputs = self.clip_processor(
            text=brand_options,
            images=image_pil,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = torch.nn.functional.softmax(logits, dim=0)
        
        # Get best match
        best_idx = probs.argmax().item()
        confidence = probs[best_idx].item()
        
        # Parse result
        result_text = brand_options[best_idx]
        brand, model, category = self._parse_vehicle_string(result_text)
        
        # Get price info
        price_range = self.price_data.get(brand, (0, 0))
        avg_price = (price_range[0] + price_range[1]) // 2
        
        return {
            'brand': brand,
            'model': model,
            'category': category,
            'brand_confidence': round(confidence * 100, 2),
            'price': avg_price,
            'price_formatted': f"₹{avg_price:,}",
            'price_range': f"₹{price_range[0]:,} - ₹{price_range[1]:,}"
        }
    
    def _parse_vehicle_string(self, text):
        """Parse brand, model, and category from text"""
        # Remove category keywords
        clean_text = text.replace(' hatchback', '').replace(' sedan', '').replace(' SUV', '') \
                         .replace(' MPV', '').replace(' motorcycle', '').replace(' bike', '') \
                         .replace(' scooter', '').replace(' cruiser', '').replace(' sport', '') \
                         .replace(' commuter', '').replace(' touring', '').replace(' naked', '') \
                         .replace(' street', '')
        
        parts = clean_text.split()
        
        # Extract brand
        if len(parts) >= 2 and parts[0] in ['Royal', 'Maruti']:
            brand = parts[0] + ' ' + parts[1]
            model = ' '.join(parts[2:]) if len(parts) > 2 else parts[1]
        else:
            brand = parts[0] if parts else 'Unknown'
            model = ' '.join(parts[1:]) if len(parts) > 1 else 'Unknown'
        
        # Extract category
        if 'SUV' in text:
            category = 'SUV'
        elif 'sedan' in text:
            category = 'Sedan'
        elif 'hatchback' in text:
            category = 'Hatchback'
        elif 'scooter' in text:
            category = 'Scooter'
        elif 'cruiser' in text:
            category = 'Cruiser'
        elif 'sport' in text:
            category = 'Sport Bike'
        else:
            category = 'Standard'
        
        return brand, model, category


# For testing
if __name__ == "__main__":
    print("="*60)
    print("Testing Vehicle Analyzer")
    print("="*60)
    
    analyzer = VehicleAnalyzer()
    
    print("\n✅ Analyzer ready for use!")
