"""
Data Download and Export to Google Drive for DeepOSWSRM

This script exports Sentinel-1, Sentinel-2, and reference data to Google Drive.
Large datasets can be handled without local storage constraints.

Usage:
    python data_download.py
    
Then monitor exports at: https://code.earthengine.google.com/tasks
"""

import ee
import os
from pathlib import Path
import json
import time
from tqdm import tqdm


class SentinelDataDownloader:
    """Download Sentinel data to Google Drive"""
    
    def __init__(self, project_id='remote-sensing-469118', drive_folder='DeepOSWSRM_Data'):
        """
        Initialize downloader
        
        Args:
            project_id: Google Cloud project ID
            drive_folder: Google Drive folder name for exports
        """
        # NEW CODE - With API Key
        try:
            # Get API key from environment or set directly
            import os
            api_key = os.environ.get('GOOGLE_API_KEY', 'YOUR_API_KEY_HERE')  # Replace with your key
            
            # Initialize with API key
            if project_id:
                ee.Initialize(
                    project=project_id,
                    opt_url='https://earthengine.googleapis.com',
                    http_transport=None,
                    credentials=None
                )
            else:
                ee.Initialize()
            
            print("Google Earth Engine initialized successfully!")
        except Exception as e:
            print(f"Error initializing GEE: {e}")
            print("Make sure GOOGLE_API_KEY environment variable is set")
            raise
        
        self.drive_folder = drive_folder
    
    def get_sentinel1_image(self, roi, start_date, end_date, orbit='DESCENDING'):
        """
        Get Sentinel-1 SAR image
        
        Args:
            roi: Region of interest as ee.Geometry
            start_date: Start date string 'YYYY-MM-DD'
            end_date: End date string 'YYYY-MM-DD'
            orbit: Orbit direction 'ASCENDING' or 'DESCENDING'
        
        Returns:
            ee.Image with VV and VH bands
        """
        s1_collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                        .filterBounds(roi)
                        .filterDate(start_date, end_date)
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                        .filter(ee.Filter.eq('orbitProperties_pass', orbit))
                        .filter(ee.Filter.eq('instrumentMode', 'IW')))
        
        # Get median composite to reduce speckle
        s1_image = s1_collection.median().select(['VV', 'VH'])
        
        return s1_image
    
    def get_sentinel2_image(self, roi, start_date, end_date, cloud_cover=20):
        """
        Get Sentinel-2 optical image
        
        Args:
            roi: Region of interest as ee.Geometry
            start_date: Start date string 'YYYY-MM-DD'
            end_date: End date string 'YYYY-MM-DD'
            cloud_cover: Maximum cloud cover percentage
        
        Returns:
            ee.Image with B, G, R, NIR bands at 10m resolution
        """
        s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .filterBounds(roi)
                        .filterDate(start_date, end_date)
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover)))
        
        # Get median composite and select 10m bands
        s2_image = s2_collection.median().select(['B2', 'B3', 'B4', 'B8'])
        
        return s2_image
    
    def export_to_drive(self, image, description, roi, scale=10):
        """
        Export image to Google Drive
        
        Args:
            image: ee.Image to export
            description: Description for the export task
            roi: Region of interest
            scale: Export resolution in meters
        
        Returns:
            Export task
        """
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder='DeepOSWSRM_Data_exports',
            fileNamePrefix=description,
            region=roi,
            scale=scale,
            maxPixels=1e13,  # Allow very large exports
            fileFormat='GeoTIFF',
            formatOptions={'cloudOptimized': True}
        )
        
        task.start()
        print(f"   Started export: {description}")
        return task
    
    def wait_for_tasks(self, tasks, check_interval=60):
        """
        Wait for all export tasks to complete
        
        Args:
            tasks: List of export tasks
            check_interval: Seconds between status checks
        """
        if not tasks:
            return [], []
        
        print(f"\n{'='*60}")
        print(f"Monitoring {len(tasks)} export tasks...")
        print(f"{'='*60}")
        print(f"⏱  This will take 10-30 minutes depending on data size")
        print(f"🌐 Monitor at: https://code.earthengine.google.com/tasks")
        print(f"💡 You can safely close this window - exports continue in background")
        print(f"{'='*60}\n")
        
        completed = set()
        failed = set()
        
        with tqdm(total=len(tasks), desc="Overall progress", unit="task") as pbar:
            while len(completed) + len(failed) < len(tasks):
                for i, task in enumerate(tasks):
                    if i in completed or i in failed:
                        continue
                    
                    try:
                        status = task.status()
                        state = status['state']
                        desc = status.get('description', f'Task {i}')
                        
                        if state == 'COMPLETED':
                            completed.add(i)
                            pbar.update(1)
                            tqdm.write(f" Completed: {desc}")
                        elif state == 'FAILED':
                            failed.add(i)
                            pbar.update(1)
                            error = status.get('error_message', 'Unknown error')
                            tqdm.write(f" Failed: {desc}")
                            tqdm.write(f" Error: {error}")
                        elif state == 'RUNNING':
                            progress = status.get('progress', 0)
                            tqdm.write(f"⏳ Running: {desc} ({progress:.1%})", end='\r')
                    except Exception as e:
                        tqdm.write(f"⚠ Error checking task {i}: {e}")
                
                # Only sleep if there are still pending tasks
                if len(completed) + len(failed) < len(tasks):
                    time.sleep(check_interval)
        
        print(f"\n{'='*60}")
        print(f"Export Summary")
        print(f"{'='*60}")
        print(f" Completed: {len(completed)}/{len(tasks)}")
        print(f" Failed: {len(failed)}/{len(tasks)}")
        print(f"{'='*60}\n")
        
        return list(completed), list(failed)
    
    def prepare_site_export(self, roi, start_date, end_date, site_name, scale_factor=4):
        """
        Prepare training data export for one site
        
        Args:
            roi: Region of interest
            start_date: Start date
            end_date: End date
            site_name: Name for this site
            scale_factor: Super-resolution scale factor
        
        Returns:
            Dictionary with metadata and list of export tasks
        """
        print(f"\n📍 Preparing: {site_name}")
        print(f"   Date range: {start_date} to {end_date}")
        
        tasks = []
        
        try:
            # Get Sentinel-1
            print(f"   Getting Sentinel-1...")
            s1_image = self.get_sentinel1_image(roi, start_date, end_date)
            task_s1 = self.export_to_drive(
                s1_image,
                f"{site_name}_S1",
                roi,
                scale=10
            )
            tasks.append(task_s1)
            
            # Get Sentinel-2
            print(f"   Getting Sentinel-2...")
            s2_image = self.get_sentinel2_image(roi, start_date, end_date, cloud_cover=20)
            task_s2 = self.export_to_drive(
                s2_image,
                f"{site_name}_S2",
                roi,
                scale=10
            )
            tasks.append(task_s2)
            
            # Get reference (cleaner Sentinel-2)
            print(f"   Getting reference...")
            ref_image = self.get_sentinel2_image(roi, start_date, end_date, cloud_cover=5)
            task_ref = self.export_to_drive(
                ref_image,
                f"{site_name}_reference",
                roi,
                scale=10
            )
            tasks.append(task_ref)
            
            metadata = {
                'site_name': site_name,
                'roi': roi.getInfo(),
                'dates': {'start': start_date, 'end': end_date},
                'scale_factor': scale_factor,
                'drive_folder': self.drive_folder,
                'files': {
                    'sentinel1': f"{site_name}_S1.tif",
                    'sentinel2': f"{site_name}_S2.tif",
                    'reference': f"{site_name}_reference.tif"
                },
                'status': 'exporting'
            }
            
            return metadata, tasks
            
        except Exception as e:
            print(f"    Error: {e}")
            return None, []


def prepare_sample_training_sites():
    """ training sites """
    
    sites = [
          {
    'name': 'Rabindra_Sarobar_Kolkata',
    'roi': ee.Geometry.Rectangle([88.2829, 22.4445, 88.4171, 22.5825]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'East_Kolkata_Wetlands',
    'roi': ee.Geometry.Rectangle([88.4500, 22.4700, 88.5200, 22.5400]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Sundarbans_Creek_WB',
    'roi': ee.Geometry.Rectangle([88.9229, 22.0229, 88.9970, 22.0970]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Hirakud_Reservoir_Odisha',
    'roi': ee.Geometry.Rectangle([83.8550, 21.5500, 83.8950, 21.5900]), # REDUCED from 229MB
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Chilika_Lake_Odisha',
    'roi': ee.Geometry.Rectangle([85.3193, 19.6793, 85.3407, 19.7007]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Vembanad_Lake_Kerala',
    'roi': ee.Geometry.Rectangle([76.3503, 9.5803, 76.3696, 9.5996]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Kochi_Coast_Kerala',
    'roi': ee.Geometry.Rectangle([76.1765, 9.9165, 76.2855, 10.0255]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Loktak_Lake_Manipur',
    'roi': ee.Geometry.Rectangle([93.7403, 24.5003, 93.7996, 24.5596]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Dal_Lake_Kashmir',
    'roi': ee.Geometry.Rectangle([74.8266, 34.0466, 74.9233, 34.1433]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Gobind_Sagar_Himachal',
    'roi': ee.Geometry.Rectangle([76.3395, 31.3695, 76.5005, 31.5305]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Brahmaputra_Assam',
    'roi': ee.Geometry.Rectangle([91.6608, 26.1358, 91.7892, 26.2642]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Tehri_Dam_Uttarakhand',
    'roi': ee.Geometry.Rectangle([78.3900, 30.3500, 78.4600, 30.4200]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Nagarjuna_Sagar_AP',
    'roi': ee.Geometry.Rectangle([79.2506, 16.5506, 79.3493, 16.6493]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Pulicat_Lake_TN',
    'roi': ee.Geometry.Rectangle([80.2946, 13.4146, 80.3253, 13.4453]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Visakhapatnam_Bay_AP',
    'roi': ee.Geometry.Rectangle([83.2704, 17.6434, 83.3695, 17.7425]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Mumbai_Harbor_MH',
    'roi': ee.Geometry.Rectangle([72.7781, 18.8481, 72.8618, 18.9318]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Chennai_Marina_TN',
    'roi': ee.Geometry.Rectangle([80.2334, 13.0034, 80.2865, 13.0565]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Pune_Khadakwasla_Reservoir',
    'roi': ee.Geometry.Rectangle([73.7400, 18.4200, 73.7800, 18.4600]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },

  {
    'name': 'Gandhinagar_Sabarmati_River_GJ',
    'roi': ee.Geometry.Rectangle([72.6500, 23.1700, 72.7000, 23.2200]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  { 'name': 'Mettur_Dam_TN', 'roi': ee.Geometry.Rectangle([77.770, 11.750, 77.830, 11.810]), 'start_date': '2023-05-01', 'end_date': '2023-05-31' },
  { 'name': 'Hemavathi_Reservoir_KA', 'roi': ee.Geometry.Rectangle([76.000, 12.750, 76.060, 12.810]), 'start_date': '2023-05-01', 'end_date': '2023-05-31' },
  { 'name': 'Alappuzha_Backwaters_Kerala', 'roi': ee.Geometry.Rectangle([76.310, 9.460, 76.360, 9.510]), 'start_date': '2023-05-01', 'end_date': '2023-05-31' },
  { 'name': 'Sardar_Sarovar_Dam_Gujarat', 'roi': ee.Geometry.Rectangle([73.730, 21.800, 73.800, 21.870]), 'start_date': '2023-05-01', 'end_date': '2023-05-31' },
  { 'name': 'Ukai_Dam_Gujarat', 'roi': ee.Geometry.Rectangle([73.530, 21.000, 73.600, 21.070]), 'start_date': '2023-05-01', 'end_date': '2023-05-31' },
  { 'name': 'Panna_Tiger_Reserve_Ken_River', 'roi': ee.Geometry.Rectangle([80.020, 24.250, 80.090, 24.320]), 'start_date': '2023-05-01', 'end_date': '2023-05-31' },
  { 'name': 'Ganga_River_Varanasi_UP', 'roi': ee.Geometry.Rectangle([83.000, 25.280, 83.060, 25.340]), 'start_date': '2023-05-01', 'end_date': '2023-05-31' },
  { 'name': 'Manas_River_Assam', 'roi': ee.Geometry.Rectangle([90.950, 26.680, 91.010, 26.740]), 'start_date': '2023-05-01', 'end_date': '2023-05-31' },
  { 'name': 'Teesta_River_Siliguri_WB', 'roi': ee.Geometry.Rectangle([88.380, 26.730, 88.440, 26.790]), 'start_date': '2023-05-01', 'end_date': '2023-05-31' },
  { 'name': 'Mandovi_River_Goa', 'roi': ee.Geometry.Rectangle([73.810, 15.480, 73.870, 15.540]), 'start_date': '2023-05-01', 'end_date': '2023-05-31' },
  
  { 'name': 'Bhima_River_Solapur_MH',
  'roi': ee.Geometry.Rectangle([75.05, 17.85, 75.12, 17.95]), # REDUCED from 1185MB
  'start_date': '2023-05-01', 'end_date': '2023-05-31' },

{ 'name': 'Yamuna_River_Delhi_UP',
  'roi': ee.Geometry.Rectangle([77.15, 28.55, 77.30, 28.70]),
  'start_date': '2023-05-01', 'end_date': '2023-05-31' },

{ 'name': 'Wular_Lake_Kashmir',
  'roi': ee.Geometry.Rectangle([74.40, 34.18, 74.55, 34.33]), # REDUCED from 1871MB
  'start_date': '2023-05-01', 'end_date': '2023-05-31' },
{ 'name': 'Dibang_River_Arunachal',
  'roi': ee.Geometry.Rectangle([95.74, 28.14, 95.84, 28.28]), # REDUCED from 249MB
  'start_date': '2023-05-01', 'end_date': '2023-05-31' },
  
  
  {
    'name': 'Tehri_Dam_Uttarakhand',
    'roi': ee.Geometry.Rectangle([78.38, 30.35, 78.55, 30.45]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Bhakra_Nangal_HP_Punjab',
    'roi': ee.Geometry.Rectangle([76.38, 31.40, 76.48, 31.50]), # REDUCED from 520MB
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Sardar_Sarovar_Gujarat',
    'roi': ee.Geometry.Rectangle([73.65, 21.78, 73.75, 21.88]), # REDUCED from 567MB
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Ukai_Dam_Gujarat',
    'roi': ee.Geometry.Rectangle([73.52, 21.23, 73.62, 21.33]), # REDUCED from 316MB
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Pong_Dam_HP',
    'roi': ee.Geometry.Rectangle([75.90, 32.04, 76.04, 32.16]), # REDUCED from 479MB
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Hussain_Sagar_Hyderabad',
    'roi': ee.Geometry.Rectangle([78.45, 17.40, 78.50, 17.47]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Sambhar_Lake_Rajasthan',
    'roi': ee.Geometry.Rectangle([75.04, 26.92, 75.12, 27.02]), # REDUCED from 1331MB
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
 
  
  {
    'name': 'Chilka_North_Channel',
    'roi': ee.Geometry.Rectangle([85.38, 19.78, 85.48, 19.88]), # REDUCED from 390MB
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Bhojtal_Bhopal',
    'roi': ee.Geometry.Rectangle([77.33, 23.22, 77.45, 23.30]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
 
  
  {
    'name': 'Kainji_Lake_Jharkhand',
    'roi': ee.Geometry.Rectangle([86.04, 23.84, 86.16, 23.96]), # REDUCED from 431MB
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  
  {
    'name': 'Gandhisagar_MP',
    'roi': ee.Geometry.Rectangle([75.64, 24.40, 75.80, 24.54]), # REDUCED from 773MB
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Mettur_Dam_TN',
    'roi': ee.Geometry.Rectangle([77.77, 11.76, 77.92, 11.88]), # REDUCED from 369MB
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },

{
  'name': 'Mayurakshi_Reservoir_Jharkhand_WB',
  'roi': ee.Geometry.Rectangle([87.12, 24.12, 87.26, 24.26]), # REDUCED from 344MB
  'start_date': '2023-05-01',
  'end_date': '2023-05-31'
},

{
    'name': 'Ukai_Dam_Gujarat',
    'roi': ee.Geometry.Rectangle([73.52, 21.24, 73.62, 21.36]), # REDUCED from 422MB
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Sardar_Sarovar_Reservoir_Gujarat',
    'roi': ee.Geometry.Rectangle([73.66, 21.78, 73.78, 21.88]), # REDUCED from 462MB
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Bhakra_Dam_HP_Punjab',
    'roi': ee.Geometry.Rectangle([76.38, 31.38, 76.52, 31.52]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  
  
  {
    'name': 'Matatila_Dam_UP',
    'roi': ee.Geometry.Rectangle([78.58, 25.02, 78.66, 25.12]), # REDUCED from 192MB (kept small but optimized)
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Bansagar_Reservoir_MP',
    'roi': ee.Geometry.Rectangle([81.22, 24.08, 81.36, 24.20]), # REDUCED from 186MB (kept small but optimized)
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  
  {
    'name': 'Koyna_Dam_Maharashtra',
    'roi': ee.Geometry.Rectangle([73.68, 17.38, 73.80, 17.50]), # REDUCED from 180MB (kept small but optimized)
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  
  {
    'name': 'Krishna_River_Bridge_Vijayawada_AP',
    'roi': ee.Geometry.Rectangle([80.56, 16.42, 80.66, 16.52]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Dowleswaram_Barrage_AP',
    'roi': ee.Geometry.Rectangle([81.72, 16.90, 81.85, 17.00]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Hemavathi_Reservoir_Karnataka',
    'roi': ee.Geometry.Rectangle([76.18, 12.80, 76.30, 12.94]), # REDUCED from 221MB
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  
  {
    'name': 'Harangi_Reservoir_Karnataka',
    'roi': ee.Geometry.Rectangle([75.88, 12.37, 75.96, 12.46]), # REDUCED from 207MB
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Idukki_Reservoir_Kerala',
    'roi': ee.Geometry.Rectangle([76.87, 9.79, 76.96, 9.88]), # REDUCED from 335MB
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Aliyar_Reservoir_TN',
    'roi': ee.Geometry.Rectangle([76.94, 10.45, 77.00, 10.52]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Pykara_Lake_TN',
    'roi': ee.Geometry.Rectangle([76.58, 11.38, 76.70, 11.48]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  
  {
    'name': 'RanaPratap_Sagar_Rajasthan',
    'roi': ee.Geometry.Rectangle([75.50, 24.90, 75.64, 25.04]), # REDUCED from 205MB
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Chamera_Lake_Himachal',
    'roi': ee.Geometry.Rectangle([75.93, 32.53, 76.03, 32.63]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Pandoh_Dam_Himachal',
    'roi': ee.Geometry.Rectangle([77.02, 31.66, 77.12, 31.76]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Kol_Dam_HP',
    'roi': ee.Geometry.Rectangle([76.75, 31.40, 76.85, 31.50]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Hasdeo_Bango_Reservoir_Chhattisgarh',
    'roi': ee.Geometry.Rectangle([82.50, 22.66, 82.66, 22.80]), # REDUCED from 313MB
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  

  {
    'name': 'Siang_River_Arunachal',
    'roi': ee.Geometry.Rectangle([94.73, 28.03, 94.93, 28.23]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Farakka_Barrage_WB',
    'roi': ee.Geometry.Rectangle([87.85, 24.78, 88.00, 24.88]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Kangsabati_Reservoir_WB',
    'roi': ee.Geometry.Rectangle([86.80, 22.94, 86.94, 23.08]), # REDUCED from 347MB
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  
  {
    'name': 'Mahanadi_Delta_Odisha',
    'roi': ee.Geometry.Rectangle([86.62, 20.32, 86.76, 20.46]), # REDUCED from 177MB (kept small but optimized)
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Godavari_Delta_AP',
    'roi': ee.Geometry.Rectangle([81.78, 16.58, 81.88, 16.70]), # REDUCED from 434MB
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Cauvery_Delta_TN',
    'roi': ee.Geometry.Rectangle([79.70, 10.40, 79.90, 10.60]),
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Sharavathi_Backwaters_Karnataka',
    'roi': ee.Geometry.Rectangle([74.75, 14.22, 74.86, 14.35]), # REDUCED from 768MB
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
    'name': 'Bhima_River_Solapur_MH',
    'roi': ee.Geometry.Rectangle([75.88, 17.58, 76.00, 17.70]), # REDUCED from 396MB
    'start_date': '2023-05-01',
    'end_date': '2023-05-31'
  },
  {
  'name': 'Indravati_Reservoir_Chhattisgarh',
  'roi': ee.Geometry.Rectangle([81.25, 19.10, 81.40, 19.25]),
  'start_date': '2023-05-01',
  'end_date': '2023-05-31'
},
{
  'name': 'Subarnarekha_Barrage_Jharkhand',
  'roi': ee.Geometry.Rectangle([86.02, 22.37, 86.16, 22.48]), # REDUCED from 314MB
  'start_date': '2023-05-01',
  'end_date': '2023-05-31'
},

{
  'name': 'Sela_Lake_Sela_Pass_Arunachal',
  'roi': ee.Geometry.Rectangle([92.080, 27.490, 92.130, 27.525]),
  'start_date': '2023-05-01',
  'end_date': '2023-05-31'
}


    ]

    return sites


def main():
    """Main function to export data to Google Drive"""
    
    print("\n" + "="*60)
    print("DeepOSWSRM - Export to Google Drive")
    print("="*60)
    print("This script will export satellite data to Google Drive.")
    print("="*60 + "\n")
    
    # Configuration
    DRIVE_FOLDER = 'DeepOSWSRM_Data'
    LOCAL_DIR = './deeposwsrm_data'
    scale_factor = 4
    WAIT_FOR_COMPLETION = True  # Set False to start exports and exit
    
    # Initialize downloader
    print("Initializing Google Earth Engine...")
    try:
        downloader = SentinelDataDownloader(drive_folder=DRIVE_FOLDER)
    except:
        print("\n❌ Authentication required!")
        print("Please run: earthengine authenticate")
        return
    
    # Get training sites
    sites = prepare_sample_training_sites()
    print(f"\n📋 Found {len(sites)} training sites")
    
    # Export data for each site
    all_metadata = []
    all_tasks = []
    
    print(f"\n{'='*60}")
    print("Starting exports to Google Drive...")
    print(f"{'='*60}")
    
    for i, site in enumerate(sites, 1):
        print(f"\n[{i}/{len(sites)}]", end=" ")
        try:
            metadata, tasks = downloader.prepare_site_export(
                roi=site['roi'],
                start_date=site['start_date'],
                end_date=site['end_date'],
                site_name=site['name'],
                scale_factor=scale_factor
            )
            
            if metadata:
                all_metadata.append(metadata)
                all_tasks.extend(tasks)
            
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    # Save metadata
    os.makedirs(LOCAL_DIR, exist_ok=True)
    metadata_path = os.path.join(LOCAL_DIR, 'drive_export_metadata.json')
    
    export_info = {
        'drive_folder': DRIVE_FOLDER,
        'scale_factor': scale_factor,
        'total_sites': len(sites),
        'successful_exports': len(all_metadata),
        'total_files': len(all_tasks),
        'sites': all_metadata
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(export_info, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Export Summary")
    print(f"{'='*60}")
    print(f" Started {len(all_tasks)} export tasks")
    print(f" Exporting to Google Drive folder: '{DRIVE_FOLDER}'")
    print(f" Metadata saved: {metadata_path}")
    print(f"{'='*60}")
    
    # Wait for completion or exit
    if WAIT_FOR_COMPLETION and all_tasks:
        user_input = input("\nWait for exports to complete? (y/n): ").lower()
        if user_input == 'y':
            completed, failed = downloader.wait_for_tasks(all_tasks)
            
            # Update metadata with completion status
            export_info['export_summary'] = {
                'total': len(all_tasks),
                'completed': len(completed),
                'failed': len(failed)
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(export_info, f, indent=2)
        else:
            print("\n💡 Exports will continue in background")
            print(f"   Monitor at: https://code.earthengine.google.com/tasks")
    
    print(f"\n{'='*60}")
    print("Next Steps")
    print(f"{'='*60}")
    print(f"1. Monitor exports: https://code.earthengine.google.com/tasks")
    print(f"2. Once complete, check Google Drive folder: '{DRIVE_FOLDER}'")
    print(f"3. Download files: python download_from_drive.py")
    print(f"4. Process data: python process_downloaded_data.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()