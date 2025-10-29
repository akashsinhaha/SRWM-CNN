"""
Process Downloaded Data - Create Water Masks

This script processes the downloaded Sentinel data from Google Drive
and creates water masks at fine resolution for training.

Usage:
    python process_downloaded_data.py
"""

import os
from pathlib import Path
import json
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import cv2
from tqdm import tqdm


class WaterMaskProcessor:
    """Create water masks from reference images"""
    
    def __init__(self, scale_factor=4):
        """
        Initialize processor
        
        Args:
            scale_factor: Super-resolution scale factor
        """
        self.scale_factor = scale_factor
    
    @staticmethod
    def calculate_ndwi(green, nir):
        """
        Calculate Normalized Difference Water Index
        NDWI = (Green - NIR) / (Green + NIR)
        
        Args:
            green: Green band array
            nir: NIR band array
        
        Returns:
            NDWI array
        """
        return (green - nir) / (green + nir + 1e-8)
    
    def create_water_mask(self, reference_path, threshold=0.0):
        """
        Create binary water mask from reference image
        
        Args:
            reference_path: Path to reference GeoTIFF
            threshold: NDWI threshold for water classification
        
        Returns:
            Tuple of (water_mask, profile)
        """
        with rasterio.open(reference_path) as src:
            # Read bands (assuming B2=Green, B8=NIR for Sentinel-2)
            bands = src.read()
            profile = src.profile.copy()
            
            # Check number of bands
            if bands.shape[0] >= 4:
                green = bands[1].astype(float)  # B3 (Green)
                nir = bands[3].astype(float)     # B8 (NIR)
                
                # Calculate NDWI
                ndwi = self.calculate_ndwi(green, nir)
            else:
                # Fallback: use last band (NIR) threshold
                nir = bands[-1].astype(float)
                ndwi = -nir  # Water has low NIR
            
            # Create binary mask at coarse resolution
            water_mask_coarse = (ndwi > threshold).astype(np.uint8)
            
            # Upsample to fine resolution (scale_factor × original)
            h_coarse, w_coarse = water_mask_coarse.shape
            h_fine = h_coarse * self.scale_factor
            w_fine = w_coarse * self.scale_factor
            
            # Use nearest neighbor for binary masks
            water_mask_fine = cv2.resize(
                water_mask_coarse,
                (w_fine, h_fine),
                interpolation=cv2.INTER_NEAREST
            )
            
            print(f"    Created mask: {h_coarse}×{w_coarse} → {h_fine}×{w_fine}")
            
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
    
    def process_site(self, site_name, data_dir):
        """
        Process one site: create water mask from reference
        
        Args:
            site_name: Name of the site
            data_dir: Data directory
        
        Returns:
            Path to created water mask or None on error
        """
        # Find reference file
        ref_pattern = f"{site_name}_reference.tif"
        ref_path = os.path.join(data_dir, ref_pattern)
        
        if not os.path.exists(ref_path):
            print(f"    ✗ Reference not found: {ref_pattern}")
            return None
        
        # Create water mask
        try:
            water_mask, profile = self.create_water_mask(ref_path)
            
            # Save mask
            mask_path = os.path.join(data_dir, f"{site_name}_water_mask.tif")
            
            with rasterio.open(mask_path, 'w', **profile) as dst:
                dst.write(water_mask, 1)
            
            print(f"    ✓ Saved: {mask_path}")
            return mask_path
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            return None


def find_downloaded_sites(data_dir):
    """
    Find all downloaded sites by looking for S1, S2, and reference files
    
    Args:
        data_dir: Data directory
    
    Returns:
        List of site names
    """
    sites = set()
    
    # Look for all _reference.tif files
    for file in os.listdir(data_dir):
        if file.endswith('_reference.tif'):
            site_name = file.replace('_reference.tif', '')
            
            # Check if S1 and S2 also exist
            s1_path = os.path.join(data_dir, f"{site_name}_S1.tif")
            s2_path = os.path.join(data_dir, f"{site_name}_S2.tif")
            
            if os.path.exists(s1_path) and os.path.exists(s2_path):
                sites.add(site_name)
    
    return sorted(list(sites))


def create_training_metadata(data_dir, sites, scale_factor):
    """
    Create metadata.json file for training
    
    Args:
        data_dir: Data directory
        sites: List of site names
        scale_factor: Scale factor
    """
    metadata = []
    
    for site_name in sites:
        s1_path = os.path.join(data_dir, f"{site_name}_S1.tif")
        s2_path = os.path.join(data_dir, f"{site_name}_S2.tif")
        ref_path = os.path.join(data_dir, f"{site_name}_reference.tif")
        mask_path = os.path.join(data_dir, f"{site_name}_water_mask.tif")
        
        # Verify all files exist
        if all(os.path.exists(p) for p in [s1_path, s2_path, ref_path, mask_path]):
            metadata.append({
                'site_name': site_name,
                'sentinel1': s1_path,
                'sentinel2': s2_path,
                'reference': ref_path,
                'water_mask': mask_path,
                'scale_factor': scale_factor
            })
    
    # Save metadata
    metadata_path = os.path.join(data_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Created metadata: {metadata_path}")
    print(f"  Total sites: {len(metadata)}")
    
    return metadata_path


def verify_data_integrity(data_dir, sites):
    """
    Verify downloaded data integrity
    
    Args:
        data_dir: Data directory
        sites: List of site names
    
    Returns:
        Dictionary with verification results
    """
    print(f"\n{'='*60}")
    print("Verifying Data Integrity")
    print(f"{'='*60}\n")
    
    results = {
        'complete': [],
        'incomplete': [],
        'corrupted': []
    }
    
    for site_name in tqdm(sites, desc="Verifying files"):
        files = {
            'S1': os.path.join(data_dir, f"{site_name}_S1.tif"),
            'S2': os.path.join(data_dir, f"{site_name}_S2.tif"),
            'Ref': os.path.join(data_dir, f"{site_name}_reference.tif")
        }
        
        # Check if all files exist
        missing = [k for k, v in files.items() if not os.path.exists(v)]
        
        if missing:
            results['incomplete'].append({
                'site': site_name,
                'missing': missing
            })
            continue
        
        # Check if files are valid GeoTIFFs
        corrupted = []
        for file_type, file_path in files.items():
            try:
                with rasterio.open(file_path) as src:
                    # Try to read metadata
                    _ = src.bounds
                    _ = src.crs
                    
                    # Check if file has data
                    if src.width == 0 or src.height == 0:
                        corrupted.append(file_type)
            except Exception as e:
                corrupted.append(file_type)
        
        if corrupted:
            results['corrupted'].append({
                'site': site_name,
                'corrupted': corrupted
            })
        else:
            results['complete'].append(site_name)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Verification Summary")
    print(f"{'='*60}")
    print(f"✓ Complete: {len(results['complete'])} sites")
    print(f"⚠️  Incomplete: {len(results['incomplete'])} sites")
    print(f"✗ Corrupted: {len(results['corrupted'])} sites")
    
    if results['incomplete']:
        print(f"\nIncomplete sites (missing files):")
        for item in results['incomplete'][:5]:
            print(f"  - {item['site']}: missing {', '.join(item['missing'])}")
        if len(results['incomplete']) > 5:
            print(f"  ... and {len(results['incomplete']) - 5} more")
    
    if results['corrupted']:
        print(f"\nCorrupted sites:")
        for item in results['corrupted'][:5]:
            print(f"  - {item['site']}: corrupted {', '.join(item['corrupted'])}")
        if len(results['corrupted']) > 5:
            print(f"  ... and {len(results['corrupted']) - 5} more")
    
    print(f"{'='*60}\n")
    
    return results


def main():
    """Main processing function"""
    
    print("\n" + "="*60)
    print("DeepOSWSRM - Process Downloaded Data")
    print("="*60)
    print("This script creates water masks from downloaded satellite data")
    print("="*60 + "\n")
    
    # Configuration
    DATA_DIR = './deeposwsrm_data'
    SCALE_FACTOR = 4
    NDWI_THRESHOLD = 0.0
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Data directory not found: {DATA_DIR}")
        print("\nPlease run download_from_drive.py first to download data.")
        return
    
    # Find downloaded sites
    print(f"{'='*60}")
    print("Step 1: Finding downloaded sites")
    print(f"{'='*60}")
    
    sites = find_downloaded_sites(DATA_DIR)
    
    if not sites:
        print(f"\n❌ No complete sites found in {DATA_DIR}")
        print("\nA complete site requires:")
        print("  - {site_name}_S1.tif")
        print("  - {site_name}_S2.tif")
        print("  - {site_name}_reference.tif")
        return
    
    print(f"\n✓ Found {len(sites)} sites with complete data")
    
    # Verify data integrity
    verification = verify_data_integrity(DATA_DIR, sites)
    valid_sites = verification['complete']
    
    if not valid_sites:
        print(f"\n❌ No valid sites found. All sites have missing or corrupted files.")
        return
    
    # Create water masks
    print(f"{'='*60}")
    print(f"Step 2: Creating water masks (scale {SCALE_FACTOR}x)")
    print(f"{'='*60}\n")
    
    processor = WaterMaskProcessor(scale_factor=SCALE_FACTOR)
    
    created_masks = []
    failed_masks = []
    
    for i, site_name in enumerate(valid_sites, 1):
        print(f"[{i}/{len(valid_sites)}] Processing: {site_name}")
        
        mask_path = processor.process_site(site_name, DATA_DIR)
        
        if mask_path:
            created_masks.append(site_name)
        else:
            failed_masks.append(site_name)
    
    # Create training metadata
    print(f"\n{'='*60}")
    print("Step 3: Creating training metadata")
    print(f"{'='*60}")
    
    metadata_path = create_training_metadata(DATA_DIR, created_masks, SCALE_FACTOR)
    
    # Final summary
    print(f"\n{'='*60}")
    print("Processing Complete!")
    print(f"{'='*60}")
    print(f"✓ Created water masks: {len(created_masks)}/{len(valid_sites)} sites")
    
    if failed_masks:
        print(f"✗ Failed: {len(failed_masks)} sites")
        for site in failed_masks[:5]:
            print(f"  - {site}")
        if len(failed_masks) > 5:
            print(f"  ... and {len(failed_masks) - 5} more")
    
    print(f"\n📁 Data directory: {DATA_DIR}")
    print(f"📄 Metadata file: {metadata_path}")
    
    # Next steps
    print(f"\n{'='*60}")
    print("Next Steps")
    print(f"{'='*60}")
    print("1. Verify data:")
    print(f"   ls -lh {DATA_DIR}")
    print("\n2. Test dataset:")
    print(f"   python dataset.py")
    print("\n3. Train model:")
    print(f"   python train.py --data_dir {DATA_DIR} --scale_factor {SCALE_FACTOR} --epochs 100")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()