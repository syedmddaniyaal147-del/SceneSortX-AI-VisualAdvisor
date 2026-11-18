"""
Complete Batch Image Analyzer
Analyzes: Scene, Waste, Fashion, Vehicle
here this version is made in order to analyze images in batch mode across all four domains. orr you may even add images of on domain upto 50 . however the vehical analyzer may not 
work properly if there are multiple vehicles in the image , as it is trained on single vehicle images mostly. the acuracy is not so good , 
it works fine for basic images . the logo need to be more visible .
"""

import os
import cv2
import pandas as pd
from datetime import datetime
import numpy as np
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import vehicle analyzer
from vehicle_classifier import VehicleAnalyzer

# Import other analyzers from backend
try:
    # Try to import from app module
    import importlib.util
    spec = importlib.util.spec_from_file_location("app", "app.py")
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)
    
    SceneClassifier = app_module.SceneClassifier
    WasteClassifier = app_module.WasteClassifier
    FashionAnalyzer = app_module.FashionAnalyzer
    
    MODULES_AVAILABLE = True
    print(" Successfully imported Scene, Waste, Fashion analyzers from app.py")
    
except Exception as e:
    print(f"  Could not import from app.py: {e}")
    print("Creating standalone analyzers...")
    MODULES_AVAILABLE = False
    
    # Create dummy classes
    class SceneClassifier:
        def classify(self, image_np):
            return {
                'primary': 'Urban Environment',
                'confidence': 0.5,
                'sustainabilityScore': 50,
                'attributes': ['urban', 'mixed']
            }
    
    class WasteClassifier:
        def classify(self, image_np):
            return {
                'detected': False,
                'materials': []
            }
    
    class FashionAnalyzer:
        def analyze(self, image_np):
            return {
                'styles': [{'style': 'Contemporary', 'percentage': 50, 'sustainability': 'Medium'}],
                'culturalContext': 'Urban'
            }


class CompleteBatchAnalyzer:
    """
    Analyze images across all 4 domains:
    1. Scene Classification
    2. Waste Detection
    3. Fashion Analysis
    4. Vehicle Detection
    """
    
    def __init__(self):
        print(" Initializing Complete Batch Analyzer...")
        print("Loading all AI modules...\n")
        
        # Initialize all analyzers
        try:
            self.scene_classifier = SceneClassifier()
            print(" Scene Classifier loaded")
        except Exception as e:
            print(f" Scene Classifier failed: {e}")
            self.scene_classifier = None
        
        try:
            self.waste_classifier = WasteClassifier()
            print(" Waste Classifier loaded")
        except Exception as e:
            print(f" Waste Classifier failed: {e}")
            self.waste_classifier = None
        
        try:
            self.fashion_analyzer = FashionAnalyzer()
            print(" Fashion Analyzer loaded")
        except Exception as e:
            print(f" Fashion Analyzer failed: {e}")
            self.fashion_analyzer = None
        
        try:
            self.vehicle_analyzer = VehicleAnalyzer()
            print(" Vehicle Analyzer loaded")
        except Exception as e:
            print(f" Vehicle Analyzer failed: {e}")
            self.vehicle_analyzer = None
        
        print("\n Batch Analyzer ready!\n")
    
    def process_folder(self, folder_path, output_csv='complete_analysis_results.csv'):
        """
        Process all images in folder with all 4 analyzers
        
        Args:
            folder_path: Path to folder containing images
            output_csv: Output CSV filename
            
        Returns:
            DataFrame with complete analysis
        """
        # Get all image files
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        
        if not os.path.exists(folder_path):
            print(f" Folder '{folder_path}' does not exist!")
            return None
        
        all_files = os.listdir(folder_path)
        image_files = [f for f in all_files if os.path.splitext(f.lower())[1] in extensions]
        
        if not image_files:
            print(f" No images found in '{folder_path}'")
            return None
        
        print(f" Found {len(image_files)} images to analyze")
        print("="*80)
        
        results = []
        successful = 0
        failed = 0
        
        for idx, filename in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] Processing: {filename}")
            print("-"*80)
            
            try:
                # Load image
                img_path = os.path.join(folder_path, filename)
                image = cv2.imread(img_path)
                
                if image is None:
                    print(f"   Could not load image")
                    failed += 1
                    results.append({
                        'filename': filename,
                        'error': 'Could not load image'
                    })
                    continue
                
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Initialize result row
                row = {
                    'filename': filename,
                    'timestamp': datetime.now().isoformat(),
                    'analysis_complete': False
                }
                
                # Analyze with all 4 modules
                if self.scene_classifier:
                    print("   Analyzing scene...")
                    scene_result = self.scene_classifier.classify(image_rgb)
                    row.update({
                        'scene_type': scene_result.get('primary', 'Unknown'),
                        'scene_confidence': scene_result.get('confidence', 0),
                        'sustainability_score': scene_result.get('sustainabilityScore', 0),
                        'scene_attributes': ', '.join(scene_result.get('attributes', []))
                    })
                
                if self.waste_classifier:
                    print("    Analyzing waste...")
                    waste_result = self.waste_classifier.classify(image_rgb)
                    row.update({
                        'waste_detected': waste_result.get('detected', False),
                        'waste_materials': self._format_waste_materials(waste_result.get('materials', [])),
                        'dominant_waste': waste_result['materials'][0]['type'] if waste_result.get('materials') else 'None',
                        'waste_recyclable_count': sum(1 for m in waste_result.get('materials', []) if m.get('recyclable'))
                    })
                
                if self.fashion_analyzer:
                    print("   Analyzing fashion...")
                    fashion_result = self.fashion_analyzer.analyze(image_rgb)
                    row.update({
                        'fashion_primary_style': fashion_result['styles'][0]['style'] if fashion_result.get('styles') else 'Unknown',
                        'fashion_style_percentage': fashion_result['styles'][0]['percentage'] if fashion_result.get('styles') else 0,
                        'fashion_sustainability': fashion_result['styles'][0]['sustainability'] if fashion_result.get('styles') else 'Unknown',
                        'cultural_context': fashion_result.get('culturalContext', 'Unknown'),
                        'fashion_all_styles': self._format_fashion_styles(fashion_result.get('styles', []))
                    })
                
                if self.vehicle_analyzer:
                    print("   Analyzing vehicle...")
                    vehicle_result = self.vehicle_analyzer.analyze(image_rgb)
                    row.update({
                        'vehicle_detected': vehicle_result.get('detected', False),
                        'vehicle_type': vehicle_result.get('vehicle_type', 'None'),
                        'vehicle_brand': vehicle_result.get('brand', 'N/A'),
                        'vehicle_model': vehicle_result.get('model', 'N/A'),
                        'vehicle_category': vehicle_result.get('category', 'N/A'),
                        'vehicle_confidence': vehicle_result.get('confidence', 0),
                        'vehicle_brand_confidence': vehicle_result.get('brand_confidence', 0),
                        'vehicle_price': vehicle_result.get('price_formatted', 'N/A'),
                        'total_vehicles_in_image': vehicle_result.get('total_vehicles', 0)
                    })
                
                row['analysis_complete'] = True
                results.append(row)
                
                # Print summary
                print(f"\n   Analysis Complete:")
                if 'scene_type' in row:
                    print(f"     Scene: {row['scene_type']}")
                if 'waste_detected' in row:
                    print(f"     Waste: {'Yes' if row['waste_detected'] else 'No'}")
                if 'fashion_primary_style' in row:
                    print(f"     Fashion: {row['fashion_primary_style']}")
                if 'vehicle_detected' in row:
                    if row['vehicle_detected']:
                        print(f"     Vehicle: {row['vehicle_brand']} {row['vehicle_model']}")
                    else:
                        print(f"     Vehicle: Not detected")
                
                successful += 1
                
            except Exception as e:
                print(f"   Error: {str(e)}")
                import traceback
                traceback.print_exc()
                failed += 1
                results.append({
                    'filename': filename,
                    'error': str(e),
                    'analysis_complete': False
                })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        
        # Print summary
        print("\n" + "="*80)
        print(" BATCH PROCESSING COMPLETE")
        print("="*80)
        print(f"Total images: {len(image_files)}")
        print(f"Successfully analyzed: {successful}")
        print(f"Failed: {failed}")
        print(f"\n Results saved to: {output_csv}")
        
        # Detailed summary
        self._print_detailed_summary(df)
        
        return df
    
    def _format_waste_materials(self, materials):
        """Format waste materials for CSV"""
        if not materials:
            return 'None'
        return ', '.join([f"{m['type']} ({m['percentage']}%)" for m in materials])
    
    def _format_fashion_styles(self, styles):
        """Format fashion styles for CSV"""
        if not styles:
            return 'None'
        return ', '.join([f"{s['style']} ({s['percentage']}%)" for s in styles])
    
    def _print_detailed_summary(self, df):
        """Print comprehensive analysis summary"""
        print("\n" + "="*80)
        print(" DETAILED ANALYSIS SUMMARY")
        print("="*80)
        
        successful_df = df[df.get('analysis_complete', False)]
        
        if len(successful_df) == 0:
            print("No successful analyses to summarize")
            return
        
        # Scene Analysis
        if 'scene_type' in successful_df.columns:
            print("\n SCENE ANALYSIS:")
            scene_counts = successful_df['scene_type'].value_counts()
            for scene, count in scene_counts.items():
                print(f"  {scene}: {count} images")
            if 'sustainability_score' in successful_df.columns:
                avg_sustainability = successful_df['sustainability_score'].mean()
                print(f"  Average Sustainability Score: {avg_sustainability:.1f}/100")
        
        # Waste Analysis
        if 'waste_detected' in successful_df.columns:
            print("\n  WASTE ANALYSIS:")
            waste_count = successful_df['waste_detected'].sum()
            waste_rate = (waste_count / len(successful_df)) * 100
            print(f"  Waste detected in: {waste_count}/{len(successful_df)} images ({waste_rate:.1f}%)")
            if waste_count > 0 and 'dominant_waste' in successful_df.columns:
                print(f"  Most common waste: {successful_df['dominant_waste'].mode()[0]}")
        
        # Fashion Analysis
        if 'fashion_primary_style' in successful_df.columns:
            print("\n FASHION ANALYSIS:")
            style_counts = successful_df['fashion_primary_style'].value_counts()
            print("  Top styles:")
            for style, count in style_counts.head(3).items():
                print(f"    {style}: {count} images")
        
        # Vehicle Analysis
        if 'vehicle_detected' in successful_df.columns:
            print("\n VEHICLE ANALYSIS:")
            vehicle_count = successful_df['vehicle_detected'].sum()
            vehicle_rate = (vehicle_count / len(successful_df)) * 100
            print(f"  Vehicles detected in: {vehicle_count}/{len(successful_df)} images ({vehicle_rate:.1f}%)")
            
            if vehicle_count > 0:
                vehicle_df = successful_df[successful_df['vehicle_detected']]
                
                if 'vehicle_type' in vehicle_df.columns:
                    print(f"\n  Vehicle Types:")
                    type_counts = vehicle_df['vehicle_type'].value_counts()
                    for vtype, count in type_counts.items():
                        print(f"    {vtype}: {count}")
                
                if 'vehicle_brand' in vehicle_df.columns:
                    print(f"\n  Top Brands:")
                    brand_counts = vehicle_df['vehicle_brand'].value_counts().head(5)
                    for brand, count in brand_counts.items():
                        print(f"    {brand}: {count}")
                
                if 'vehicle_confidence' in vehicle_df.columns:
                    avg_confidence = vehicle_df['vehicle_confidence'].mean()
                    print(f"\n  Average Detection Confidence: {avg_confidence:.2f}%")
        
        print("="*80)


# Main execution
if __name__ == '__main__':
    print("\n" + "="*80)
    print(" Complete Multi-Domain Batch Analyzer")
    print("   Scene | Waste | Fashion | Vehicle")
    print("="*80 + "\n")
    
    # Initialize
    analyzer = CompleteBatchAnalyzer()
    
    # Get folder path
    folder_path = input("Enter folder path with images (or press Enter for 'test_images'): ").strip()
    if not folder_path:
        folder_path = 'test_images'
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"\n Folder '{folder_path}' not found!")
        print(f"Creating folder '{folder_path}'...")
        os.makedirs(folder_path, exist_ok=True)
        print(f" Folder created! Please add images and run again.")
    else:
        # Process images
        print(f"\n Processing folder: {folder_path}\n")
        results_df = analyzer.process_folder(
            folder_path, 
            output_csv='complete_analysis_results.csv'
        )
        
        if results_df is not None:
            print("\n Complete analysis done!")
            print(" Check 'complete_analysis_results.csv' for detailed results.")
