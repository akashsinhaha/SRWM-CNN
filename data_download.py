"""
Data Download and Preprocessing for DeepOSWSRM - FIXED VERSION

This script handles:
1. Downloading Sentinel-1 and Sentinel-2 data from Google Earth Engine
2. Using Landsat-8/9 as a free alternative to PlanetScope for reference data
3. Preprocessing and cloud simulation
"""

import ee
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import os
from datetime import datetime, timedelta
import geemap
from tqdm import tqdm
import json
import cv2


class SentinelDataDownloader:
    """Download Sentinel-1 and Sentinel-2 data from Google Earth Engine"""
    
    def __init__(self, project_id='remote-sensing-469118'):
        """
        Initialize GEE
        
        Args:
            project_id: Your Google Cloud project ID (required for GEE)
        """
        try:
            if project_id:
                ee.Initialize(project=project_id)
            else:
                ee.Initialize()
            print("Google Earth Engine initialized successfully!")
        except Exception as e:
            print(f"Error initializing GEE: {e}")
            print("Please run 'earthengine authenticate' in terminal first")
            raise
    
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
    
    def get_landsat_reference(self, roi, start_date, end_date, cloud_cover=20):
        """
        Get Landsat-8/9 image as high-resolution reference (30m, free alternative to PlanetScope)
        
        Args:
            roi: Region of interest
            start_date: Start date
            end_date: End date
            cloud_cover: Maximum cloud cover
        
        Returns:
            ee.Image with optical bands
        """
        # Try Landsat-9 first (launched 2021), then Landsat-8
        l9_collection = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
                        .filterBounds(roi)
                        .filterDate(start_date, end_date)
                        .filter(ee.Filter.lt('CLOUD_COVER', cloud_cover)))
        
        l8_collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                        .filterBounds(roi)
                        .filterDate(start_date, end_date)
                        .filter(ee.Filter.lt('CLOUD_COVER', cloud_cover)))
        
        # Merge collections
        landsat = l9_collection.merge(l8_collection)
        
        if landsat.size().getInfo() == 0:
            print("Warning: No Landsat images found, trying Sentinel-2 as reference")
            return self.get_sentinel2_image(roi, start_date, end_date, cloud_cover)
        
        # Get median composite
        landsat_image = landsat.median()
        
        # Scale and select bands (SR_B2=Blue, SR_B3=Green, SR_B4=Red, SR_B5=NIR)
        landsat_image = landsat_image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5'])
        
        # Apply scaling factors
        def apply_scale_factors(image):
            optical_bands = image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5']).multiply(0.0000275).add(-0.2)
            return optical_bands
        
        landsat_image = apply_scale_factors(landsat_image)
        
        return landsat_image
    
    def download_image(self, image, roi, filename, scale=10, crs='EPSG:4326'):
        """
        Download an ee.Image to local file
        
        Args:
            image: ee.Image to download
            roi: Region of interest
            filename: Output filename
            scale: Resolution in meters
            crs: Coordinate reference system
        """
        # Create export task
        geemap.ee_export_image(
            image,
            filename=filename,
            scale=scale,
            region=roi,
            file_per_band=False,
            crs=crs
        )
        print(f"Downloaded: {filename}")
    
    def prepare_training_data(self, roi, start_date, end_date, output_dir, 
                             site_name, scale_factor=4):
        """
        Prepare a complete training sample
        
        Args:
            roi: Region of interest
            start_date: Start date
            end_date: End date
            output_dir: Output directory
            site_name: Name for this site
            scale_factor: Super-resolution scale factor
        
        Returns:
            Dictionary with file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nPreparing data for site: {site_name}")
        print(f"Date range: {start_date} to {end_date}")
        
        # Download Sentinel-1 (10m)
        print("Downloading Sentinel-1...")
        s1_image = self.get_sentinel1_image(roi, start_date, end_date)
        s1_path = os.path.join(output_dir, f"{site_name}_S1.tif")
        self.download_image(s1_image, roi, s1_path, scale=10)
        
        # Download Sentinel-2 (10m)
        print("Downloading Sentinel-2...")
        s2_image = self.get_sentinel2_image(roi, start_date, end_date)
        s2_path = os.path.join(output_dir, f"{site_name}_S2.tif")
        self.download_image(s2_image, roi, s2_path, scale=10)
        
        # Download reference data (Sentinel-2 at same resolution for consistency)
        print("Downloading reference data...")
        ref_image = self.get_sentinel2_image(roi, start_date, end_date, cloud_cover=10)
        ref_path = os.path.join(output_dir, f"{site_name}_reference.tif")
        self.download_image(ref_image, roi, ref_path, scale=10)
        
        return {
            'sentinel1': s1_path,
            'sentinel2': s2_path,
            'reference': ref_path,
            'site_name': site_name,
            'roi': roi.getInfo(),
            'dates': {'start': start_date, 'end': end_date},
            'scale_factor': scale_factor
        }


class WaterIndexCalculator:
    """Calculate water indices to create reference water maps"""
    
    @staticmethod
    def ndwi(green, nir):
        """
        Normalized Difference Water Index
        NDWI = (Green - NIR) / (Green + NIR)
        """
        return (green - nir) / (green + nir + 1e-8)
    
    @staticmethod
    def create_water_mask(image_path, method='ndwi', threshold=0, scale_factor=4):
        """
        Create binary water mask from multispectral image at FINE resolution
        
        Args:
            image_path: Path to reference image file
            method: Water index method ('ndwi', 'mndwi', 'awei')
            threshold: Threshold value
            scale_factor: Super-resolution scale factor
        
        Returns:
            Binary water mask at fine resolution (H*scale, W*scale)
        """
        with rasterio.open(image_path) as src:
            # Read at original resolution
            bands = src.read()
            profile = src.profile.copy()
            
            if method == 'ndwi' and bands.shape[0] >= 4:
                green = bands[1].astype(float)
                nir = bands[3].astype(float)
                index = WaterIndexCalculator.ndwi(green, nir)
            else:
                # Default: use NIR threshold
                nir = bands[-1].astype(float)
                index = -nir  # Water has low NIR
            
            # Create binary mask at original resolution
            water_mask_coarse = (index > threshold).astype(np.uint8)
            
            # Now upsample to fine resolution
            h_coarse, w_coarse = water_mask_coarse.shape
            h_fine = h_coarse * scale_factor
            w_fine = w_coarse * scale_factor
            
            # Use nearest neighbor interpolation for binary masks
            water_mask_fine = cv2.resize(
                water_mask_coarse, 
                (w_fine, h_fine), 
                interpolation=cv2.INTER_NEAREST
            )
            
            print(f"  Created water mask: {h_coarse}×{w_coarse} → {h_fine}×{w_fine} (scale {scale_factor}x)")
            
            # Update profile for fine resolution
            profile.update({
                'count': 1,
                'dtype': rasterio.uint8,
                'height': h_fine,
                'width': w_fine,
                'transform': src.transform * src.transform.scale(
                    src.width / w_fine,
                    src.height / h_fine
                )
            })
            
            return water_mask_fine, profile


def prepare_sample_training_sites():
    """
    Prepare sample training sites
    
    Returns list of dictionaries with ROI and date information
    """
    sites = [
        # Kolkata Region
        {
            'name': 'Rabindra_Sarobar',
            'roi': ee.Geometry.Rectangle([88.3445, 22.5085, 88.3555, 22.5185]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'East_Kolkata_Wetlands',
            'roi': ee.Geometry.Rectangle([88.4400, 22.5150, 88.4500, 22.5250]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            # bad quality data - keep for testing
            'name': 'Nalban_Lake',
            'roi': ee.Geometry.Rectangle([88.4160, 22.5740, 88.4280, 22.5840]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Santragachi_Jheel',
            'roi': ee.Geometry.Rectangle([88.2850, 22.5700, 88.2950, 22.5800]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Subhas_Sarobar',
            'roi': ee.Geometry.Rectangle([88.3900, 22.5570, 88.4000, 22.5670]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        
        # West Bengal - Other regions
        {
            'name': 'Digha_Beach',
            'roi': ee.Geometry.Rectangle([87.5200, 21.6200, 87.5400, 21.6400]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Sundarbans_Creek',
            'roi': ee.Geometry.Rectangle([88.9500, 22.0500, 88.9700, 22.0700]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        
        # Major Indian Water Bodies
        {
            'name': 'Dal_Lake_Kashmir',
            'roi': ee.Geometry.Rectangle([74.8600, 34.0800, 74.8900, 34.1100]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Chilika_Lake_Odisha',
            'roi': ee.Geometry.Rectangle([85.3200, 19.6800, 85.3600, 19.7200]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Vembanad_Lake_Kerala',
            'roi': ee.Geometry.Rectangle([76.3500, 9.5800, 76.3900, 9.6200]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Loktak_Lake_Manipur',
            'roi': ee.Geometry.Rectangle([93.7600, 24.5200, 93.8000, 24.5600]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Pulicat_Lake_TN',
            'roi': ee.Geometry.Rectangle([80.3000, 13.4200, 80.3400, 13.4600]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        
        # Reservoirs
        {
            'name': 'Hirakud_Reservoir_Odisha',
            'roi': ee.Geometry.Rectangle([83.8500, 21.5200, 83.9000, 21.5700]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Nagarjuna_Sagar_AP',
            'roi': ee.Geometry.Rectangle([79.3000, 16.5500, 79.3500, 16.6000]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Gobind_Sagar_HP',
            'roi': ee.Geometry.Rectangle([76.4200, 31.4000, 76.4700, 31.4500]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        
        # Rivers
        {
            'name': 'Hooghly_River_Kolkata',
            'roi': ee.Geometry.Rectangle([88.3200, 22.5500, 88.3500, 22.5800]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Brahmaputra_Assam',
            'roi': ee.Geometry.Rectangle([91.7000, 26.1500, 91.7500, 26.2000]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            # bad quality data - keep for testing
            'name': 'Narmada_Gujarat',
            'roi': ee.Geometry.Rectangle([73.0000, 21.8000, 73.0500, 21.8500]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        
        # Coastal areas
        {
            'name': 'Mumbai_Harbor',
            'roi': ee.Geometry.Rectangle([72.8300, 18.9000, 72.8700, 18.9400]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Chennai_Marina',
            'roi': ee.Geometry.Rectangle([80.2700, 13.0400, 80.3100, 13.0800]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        }
    ]
    
    return sites


def main():
    """Main function to download and prepare data"""
    
    # Initialize downloader
    print("Initializing Google Earth Engine...")
    try:
        downloader = SentinelDataDownloader()
    except:
        print("\nPlease authenticate with Google Earth Engine:")
        print("1. Run: pip install earthengine-api")
        print("2. Run: earthengine authenticate")
        print("3. Follow the authentication process")
        return
    
    # Configuration
    scale_factor = 4  # Set your desired scale factor
    output_base_dir = './deeposwsrm_data'
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Get sample sites
    sites = prepare_sample_training_sites()
    
    # Download data for each site
    all_samples = []
    for site in sites:
        try:
            sample_data = downloader.prepare_training_data(
                roi=site['roi'],
                start_date=site['start_date'],
                end_date=site['end_date'],
                output_dir=os.path.join(output_base_dir, site['name']),
                site_name=site['name'],
                scale_factor=scale_factor
            )
            all_samples.append(sample_data)
        except Exception as e:
            print(f"Error processing site {site['name']}: {e}")
            continue
    
    # Create water masks from reference images
    print("\n" + "="*60)
    print("Creating water masks at fine resolution...")
    print("="*60)
    
    for sample in all_samples:
        try:
            ref_path = sample['reference']
            site_name = sample['site_name']
            scale_factor = sample['scale_factor']
            
            print(f"\nProcessing {site_name}...")
            
            # Create fine-resolution water mask
            water_mask_fine, profile = WaterIndexCalculator.create_water_mask(
                ref_path, 
                method='ndwi', 
                threshold=0,
                scale_factor=scale_factor
            )
            
            # Save water mask
            mask_path = ref_path.replace('_reference.tif', '_water_mask.tif')
            with rasterio.open(mask_path, 'w', **profile) as dst:
                dst.write(water_mask_fine, 1)
            
            print(f"  Saved: {mask_path}")
            sample['water_mask'] = mask_path
            
        except Exception as e:
            print(f"  Error creating water mask for {sample['site_name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save metadata
    metadata_path = os.path.join(output_base_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(all_samples, f, indent=2)
    
    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)
    print(f"Downloaded {len(all_samples)} training samples")
    print(f"Metadata saved to: {metadata_path}")
    print(f"\nNext steps:")
    print(f"1. Check your data: ls {output_base_dir}")
    print(f"2. Train the model: python train.py --data_dir {output_base_dir} --scale_factor {scale_factor}")


if __name__ == "__main__":
    main()