"""
Download DeepOSWSRM Data from Google Drive to Local Directory

This script downloads files exported by data_download.py from Google Drive
to your local machine.

Prerequisites:
1. Create OAuth credentials:
   - Go to: https://console.cloud.google.com/apis/credentials
   - Create "OAuth 2.0 Client ID" (Desktop app type)
   - Download as 'credentials.json' in project directory

2. Install required package:
   pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib

Usage:
    python download_from_drive.py
"""

import os
from pathlib import Path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import pickle
import io
import json
from tqdm import tqdm

# Scopes required for Drive access
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


def authenticate_drive():
    """
    Authenticate with Google Drive
    
    Returns:
        Google Drive service object
    """
    creds = None
    token_path = 'token.pickle'
    
    # Load saved credentials
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
    
    # Refresh or get new credentials
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing access token...")
            creds.refresh(Request())
        else:
            if not os.path.exists('credentials.json'):
                print("\n❌ Error: credentials.json not found!")
                print("\nPlease create OAuth credentials:")
                print("1. Go to: https://console.cloud.google.com/apis/credentials")
                print("2. Click 'Create Credentials' → 'OAuth 2.0 Client ID'")
                print("3. Application type: 'Desktop app'")
                print("4. Download JSON and save as 'credentials.json'")
                print("5. Re-run this script\n")
                raise FileNotFoundError("credentials.json not found")
            
            print("\nAuthenticating with Google Drive...")
            print("A browser window will open. Please authorize access.")
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save credentials for next run
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)
        
        print("✓ Authentication successful!")
    
    return build('drive', 'v3', credentials=creds)


def find_folder(service, folder_name):
    """
    Find Google Drive folder by name
    
    Args:
        service: Google Drive service
        folder_name: Name of folder to find
    
    Returns:
        Folder ID or None
    """
    try:
        results = service.files().list(
            q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
            spaces='drive',
            fields='files(id, name)',
            pageSize=10
        ).execute()
        
        items = results.get('files', [])
        
        if not items:
            return None
        
        if len(items) > 1:
            print(f"\n⚠️  Found {len(items)} folders named '{folder_name}'")
            print("Using the first one. If this is wrong, please rename folders.")
        
        return items[0]['id']
    
    except Exception as e:
        print(f"Error finding folder: {e}")
        return None


def list_files_in_folder(service, folder_id):
    """
    List all files in a Google Drive folder
    
    Args:
        service: Google Drive service
        folder_id: ID of folder
    
    Returns:
        List of file metadata dictionaries
    """
    try:
        files = []
        page_token = None
        
        while True:
            results = service.files().list(
                q=f"'{folder_id}' in parents and trashed=false",
                spaces='drive',
                fields='nextPageToken, files(id, name, size, mimeType)',
                pageSize=100,
                pageToken=page_token
            ).execute()
            
            files.extend(results.get('files', []))
            page_token = results.get('nextPageToken')
            
            if not page_token:
                break
        
        return files
    
    except Exception as e:
        print(f"Error listing files: {e}")
        return []


def get_file_size_mb(size_bytes):
    """Convert bytes to MB"""
    if size_bytes is None:
        return 0
    return int(size_bytes) / (1024 * 1024)


def download_file(service, file_id, file_name, output_dir):
    """
    Download a file from Google Drive
    
    Args:
        service: Google Drive service
        file_id: ID of file to download
        file_name: Name of file
        output_dir: Output directory
    
    Returns:
        Path to downloaded file or None on error
    """
    try:
        output_path = os.path.join(output_dir, file_name)
        
        # Skip if already exists
        if os.path.exists(output_path):
            print(f"⏭️  Skipped (exists): {file_name}")
            return output_path
        
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else output_dir, exist_ok=True)
        
        # Download file
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(output_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request, chunksize=10*1024*1024)  # 10MB chunks
        
        done = False
        pbar = tqdm(total=100, desc=f"📥 {file_name[:40]}", unit="%", leave=False)
        
        while not done:
            status, done = downloader.next_chunk()
            if status:
                progress = int(status.progress() * 100)
                pbar.update(progress - pbar.n)
        
        pbar.close()
        fh.close()
        
        # Verify file exists
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✓ Downloaded: {file_name} ({size_mb:.1f} MB)")
            return output_path
        else:
            print(f"✗ Failed: {file_name} (file not created)")
            return None
    
    except Exception as e:
        print(f"✗ Error downloading {file_name}: {e}")
        return None


def main():
    """Main download function"""
    
    print("\n" + "="*60)
    print("DeepOSWSRM - Download from Google Drive")
    print("="*60)
    print("This script downloads exported satellite data from Google Drive")
    print("="*60 + "\n")
    
    # Configuration
    DRIVE_FOLDER = 'DeepOSWSRM_Data'
    LOCAL_DIR = './deeposwsrm_data'
    METADATA_FILE = './deeposwsrm_data/drive_export_metadata.json'
    
    # Check if metadata exists
    if not os.path.exists(METADATA_FILE):
        print(f"⚠️  Metadata file not found: {METADATA_FILE}")
        print("This is optional. Continuing with download...")
        metadata = None
    else:
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
            DRIVE_FOLDER = metadata.get('drive_folder', DRIVE_FOLDER)
        print(f"✓ Loaded metadata from: {METADATA_FILE}")
    
    # Authenticate
    print(f"\n{'='*60}")
    print("Step 1: Authenticating with Google Drive")
    print(f"{'='*60}")
    
    try:
        service = authenticate_drive()
        print("✓ Successfully authenticated!")
    except Exception as e:
        print(f"\n❌ Authentication failed: {e}")
        return
    
    # Find folder
    print(f"\n{'='*60}")
    print(f"Step 2: Finding folder '{DRIVE_FOLDER}'")
    print(f"{'='*60}")
    
    folder_id = find_folder(service, DRIVE_FOLDER)
    
    if not folder_id:
        print(f"\n❌ Error: Folder '{DRIVE_FOLDER}' not found in Google Drive")
        print("\nPlease check:")
        print(f"1. Folder name is exactly: '{DRIVE_FOLDER}'")
        print("2. You have access to the folder")
        print("3. Exports from data_download.py completed successfully")
        return
    
    print(f"✓ Found folder: {DRIVE_FOLDER}")
    
    # List files
    print(f"\n{'='*60}")
    print("Step 3: Listing files in folder")
    print(f"{'='*60}")
    
    files = list_files_in_folder(service, folder_id)
    
    if not files:
        print(f"\n⚠️  No files found in folder '{DRIVE_FOLDER}'")
        print("\nPossible reasons:")
        print("1. Exports from data_download.py are still running")
        print("2. Exports failed (check https://code.earthengine.google.com/tasks)")
        print("3. Files were deleted or moved")
        return
    
    # Filter for .tif files only
    tif_files = [f for f in files if f['name'].endswith('.tif') or f['name'].endswith('.tiff')]
    
    if not tif_files:
        print(f"\n⚠️  No GeoTIFF files found in folder")
        return
    
    # Calculate total size
    total_size = sum(int(f.get('size', 0)) for f in tif_files)
    total_size_mb = total_size / (1024 * 1024)
    
    print(f"\n📊 Found {len(tif_files)} GeoTIFF files")
    print(f"📦 Total size: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
    
    # Show file list
    print(f"\nFiles to download:")
    for i, f in enumerate(tif_files[:10], 1):  # Show first 10
        size_mb = get_file_size_mb(f.get('size'))
        print(f"  {i}. {f['name']} ({size_mb:.1f} MB)")
    
    if len(tif_files) > 10:
        print(f"  ... and {len(tif_files) - 10} more files")
    
    # Confirm download
    print(f"\n{'='*60}")
    response = input(f"Download {len(tif_files)} files to '{LOCAL_DIR}'? (y/n): ").lower()
    
    if response != 'y':
        print("Download cancelled.")
        return
    
    # Download files
    print(f"\n{'='*60}")
    print(f"Step 4: Downloading files")
    print(f"{'='*60}\n")
    
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    downloaded = []
    skipped = []
    failed = []
    
    for i, file_info in enumerate(tif_files, 1):
        print(f"\n[{i}/{len(tif_files)}]", end=" ")
        
        result = download_file(
            service,
            file_info['id'],
            file_info['name'],
            LOCAL_DIR
        )
        
        if result:
            if os.path.exists(result) and os.path.getsize(result) > 0:
                downloaded.append(result)
            else:
                skipped.append(file_info['name'])
        else:
            failed.append(file_info['name'])
    
    # Summary
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")
    print(f"✓ Downloaded: {len(downloaded)} files")
    print(f"⏭️  Skipped: {len(skipped)} files (already existed)")
    print(f"✗ Failed: {len(failed)} files")
    print(f"📁 Location: {LOCAL_DIR}")
    print(f"{'='*60}")
    
    if failed:
        print(f"\n⚠️  Failed downloads:")
        for name in failed[:5]:
            print(f"  - {name}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
    
    # Save download log
    download_log = {
        'total_files': len(tif_files),
        'downloaded': len(downloaded),
        'skipped': len(skipped),
        'failed': len(failed),
        'downloaded_files': downloaded,
        'failed_files': failed
    }
    
    log_path = os.path.join(LOCAL_DIR, 'download_log.json')
    with open(log_path, 'w') as f:
        json.dump(download_log, f, indent=2)
    
    print(f"\n✓ Download log saved: {log_path}")
    
    # Next steps
    print(f"\n{'='*60}")
    print("Next Steps")
    print(f"{'='*60}")
    print("1. Verify downloaded files in:", LOCAL_DIR)
    print("2. Process data to create water masks:")
    print("   python process_downloaded_data.py")
    print("3. Train model:")
    print("   python train.py --data_dir", LOCAL_DIR)
    print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()