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
            'name': 'Rabindra_Sarobar',
            'roi': ee.Geometry.Rectangle([88.282900, 22.444500, 88.417100, 22.582500]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'East_Kolkata_Wetlands',
            'roi': ee.Geometry.Rectangle([88.371957, 22.446957, 88.508043, 22.583043]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Santragachi_Jheel',
            'roi': ee.Geometry.Rectangle([88.216957, 22.501957, 88.353043, 22.638043]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Subhas_Sarobar',
            'roi': ee.Geometry.Rectangle([88.379357, 22.547357, 88.400643, 22.566643]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },

        {
            'name': 'Digha_Beach',
            'roi': ee.Geometry.Rectangle([87.486957, 21.586957, 87.573043, 21.673043]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Sundarbans_Creek',
            'roi': ee.Geometry.Rectangle([88.922957, 22.022957, 88.997043, 22.097043]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },

        {
            'name': 'Dal_Lake_Kashmir',
            'roi': ee.Geometry.Rectangle([74.826657, 34.046657, 74.923343, 34.143343]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Chilika_Lake_Odisha',
            'roi': ee.Geometry.Rectangle([85.319257, 19.679257, 85.340743, 19.700743]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Vembanad_Lake_Kerala',
            'roi': ee.Geometry.Rectangle([76.350357, 9.580357, 76.369643, 9.599643]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Loktak_Lake_Manipur',
            'roi': ee.Geometry.Rectangle([93.740357, 24.500357, 93.799643, 24.559643]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Pulicat_Lake_TN',
            'roi': ee.Geometry.Rectangle([80.294657, 13.414657, 80.325343, 13.445343]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },

        {
            'name': 'Hirakud_Reservoir_Odisha',
            'roi': ee.Geometry.Rectangle([83.838143, 21.533143, 83.911857, 21.606857]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Nagarjuna_Sagar_AP',
            'roi': ee.Geometry.Rectangle([79.250657, 16.550657, 79.349343, 16.649343]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Gobind_Sagar_HP',
            'roi': ee.Geometry.Rectangle([76.339529, 31.369529, 76.500471, 31.530471]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },

        {
            'name': 'Hooghly_River_Kolkata',
            'roi': ee.Geometry.Rectangle([88.313457, 22.508457, 88.356543, 22.631543]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Brahmaputra_Assam',
            'roi': ee.Geometry.Rectangle([91.660779, 26.135779, 91.789221, 26.264221]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },

        {
            'name': 'Mumbai_Harbor',
            'roi': ee.Geometry.Rectangle([72.778143, 18.848143, 72.861857, 18.931857]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Chennai_Marina',
            'roi': ee.Geometry.Rectangle([80.233457, 13.003457, 80.286543, 13.056543]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Visakhapatnam_Bay',
            'roi': ee.Geometry.Rectangle([83.270457, 17.643457, 83.369543, 17.742543]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
        },
        {
            'name': 'Kochi_Coast',
            'roi': ee.Geometry.Rectangle([76.176457, 9.916457, 76.285543, 10.025543]),
            'start_date': '2023-06-01',
            'end_date': '2024-08-31'
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