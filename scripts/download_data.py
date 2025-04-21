import os
import zipfile
import tempfile
import shutil
import urllib.request
from pathlib import Path

def download_building_data():
    """Download building data and organize into appropriate folders."""
    url = "https://drive.usercontent.google.com/download?id=12RsrfU8m-cvtONohD6FBcF21OPoEA_Mf&export=download&authuser=0&confirm=t&uuid=c515a571-22b7-4069-b9d1-dae146f45284&at=APcmpoyHMsiI4_i7w-zeq_eGoV9B%3A1745163787593"
    
    print("Downloading building data...")
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download the zip file to the temporary directory
        zip_path = os.path.join(temp_dir, "building_data.zip")
        
        try:
            urllib.request.urlretrieve(url, zip_path)
            print(f"Downloaded zip file to {zip_path}")
            
            # Get the project root directory
            project_root = Path(__file__).parent.parent.absolute()
            
            # Setup target directories
            footprints_dir = os.path.join(project_root, "data", "raw", "building_data", "footprints")
            grid_dir = os.path.join(project_root, "data", "raw", "building_data", "grid")
            
            # Extract files directly to appropriate directories
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get a list of all files in the zip
                all_files = [f for f in zip_ref.namelist() 
                            if not f.startswith('__MACOSX') and f.endswith('.csv')]
                
                # Separate footprint and grid files
                footprint_files = [f for f in all_files if f.endswith('_footprints.csv')]
                grid_files = [f for f in all_files if f.endswith('_grid.csv')]
                
                # Display file counts
                print(f"Found {len(footprint_files)} footprint files and {len(grid_files)} grid files")
                
                # Extract footprint files
                print("Extracting footprint files...")
                for file in footprint_files:
                    # Extract file content
                    content = zip_ref.read(file)
                    
                    # Get just the filename without path
                    filename = os.path.basename(file)
                    
                    # Write to destination
                    with open(os.path.join(footprints_dir, filename), 'wb') as f:
                        f.write(content)
                    print(f" - Extracted: {filename}")
                
                # Extract grid files
                print("Extracting grid files...")
                for file in grid_files:
                    # Extract file content
                    content = zip_ref.read(file)
                    
                    # Get just the filename without path
                    filename = os.path.basename(file)
                    
                    # Write to destination
                    with open(os.path.join(grid_dir, filename), 'wb') as f:
                        f.write(content)
                    print(f" - Extracted: {filename}")
            
            # List files in each directory for verification
            print("\nFootprints files:")
            for file in os.listdir(footprints_dir):
                print(f" - {file}")
            
            print("\nGrid files:")
            for file in os.listdir(grid_dir):
                print(f" - {file}")
                
            # Count files to verify
            footprint_count = len(os.listdir(footprints_dir))
            grid_count = len(os.listdir(grid_dir))
            print(f"\nTotal: {footprint_count} footprint files and {grid_count} grid files extracted")
        
        except Exception as e:
            print(f"Error downloading or extracting building data: {str(e)}")
            return False
    
    return True

unosat_urls = {
    "07_11_2023": "https://unosat.org/static/unosat_filesystem/3734/UNOSAT_GazaStrip_CDA_07Nov2023_GDB.zip", 
    "26_11_2023": "https://unosat.org/static/unosat_filesystem/3769/UNOSAT_GazaStrip_CDA_26November2023_GDB.zip", 
    "06_01_2024": "https://unosat.org/static/unosat_filesystem/3793/UNOSAT_GazaStrip_CDA_January2024_GDB_V2.zip", 
    "01_04_2024": "https://unosat.org/static/unosat_filesystem/3824/OCHA-OPT_013_UNOSAT_GazaStrip_CDA_01Apr2024_GDB.zip", 
    "03_05_2024": "https://unosat.org/static/unosat_filesystem/3861/OCHA_OPT-014_UNOSAT_GazaStrip_OPT_CDA_03May2024_GDB_v2.zip", 
    "06_07_2024": "https://unosat.org/static/unosat_filesystem/3904/OCHA_OPT-015_UNOSAT_GazaStrip_OPT_CDA_06July2024_GDB.zip",
    "06_09_2024": "https://unosat.org/static/unosat_filesystem/3984/OCHA-OPT-017_UNOSAT_A3_Gaza_Strip_OPT_CDA_GDB_06092024.zip",
    "01_12_2024": "https://unosat.org/static/unosat_filesystem/4047/OCHA-OPT_019_UNOSAT_GazaStrip_CDA_GDB_01December2024.zip",
    "25_02_2025": "https://gaza-unosat.docs.cern.ch/CE20231007PSE_UNOSAT_GazaStrip_ComprehensiveDamageAssessment_20250225.zip"
}


def download_unosat_data(date_key=None):
    """
    Download UNOSAT damage assessment data for Gaza.
    
    Args:
        date_key: Specific date key to download. If None, will download all datasets.
    
    Returns:
        Boolean indicating success or failure
    """
    # Dictionary of available UNOSAT datasets with their URLs
    unosat_urls = {
        "07_11_2023": "https://unosat.org/static/unosat_filesystem/3734/UNOSAT_GazaStrip_CDA_07Nov2023_GDB.zip", 
        "26_11_2023": "https://unosat.org/static/unosat_filesystem/3769/UNOSAT_GazaStrip_CDA_26November2023_GDB.zip", 
        "06_01_2024": "https://unosat.org/static/unosat_filesystem/3793/UNOSAT_GazaStrip_CDA_January2024_GDB_V2.zip", 
        "01_04_2024": "https://unosat.org/static/unosat_filesystem/3824/OCHA-OPT_013_UNOSAT_GazaStrip_CDA_01Apr2024_GDB.zip", 
        "03_05_2024": "https://unosat.org/static/unosat_filesystem/3861/OCHA_OPT-014_UNOSAT_GazaStrip_OPT_CDA_03May2024_GDB_v2.zip", 
        "06_07_2024": "https://unosat.org/static/unosat_filesystem/3904/OCHA_OPT-015_UNOSAT_GazaStrip_OPT_CDA_06July2024_GDB.zip",
        "06_09_2024": "https://unosat.org/static/unosat_filesystem/3984/OCHA-OPT-017_UNOSAT_A3_Gaza_Strip_OPT_CDA_GDB_06092024.zip",
        "01_12_2024": "https://unosat.org/static/unosat_filesystem/4047/OCHA-OPT_019_UNOSAT_GazaStrip_CDA_GDB_01December2024.zip",
        "25_02_2025": "https://gaza-unosat.docs.cern.ch/CE20231007PSE_UNOSAT_GazaStrip_ComprehensiveDamageAssessment_20250225.zip"
    }
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent.absolute()
    
    # Set the target directory
    labels_dir = os.path.join(project_root, "data", "raw", "labels", "unosat")
    os.makedirs(labels_dir, exist_ok=True)
    
    # Determine which datasets to download
    if date_key is not None:
        # Download a specific dataset
        if date_key not in unosat_urls:
            print(f"Error: Invalid date key '{date_key}'. Available keys: {', '.join(unosat_urls.keys())}")
            return False
        dates_to_download = {date_key: unosat_urls[date_key]}
    else:
        # Download all datasets
        dates_to_download = unosat_urls
    
    success_count = 0
    fail_count = 0
    
    # Download each dataset
    for date, url in dates_to_download.items():
        print(f"\nDownloading UNOSAT Gaza damage assessment data for {date}...")
        
        # Create a subdirectory for this date
        date_dir = os.path.join(labels_dir, date)
        os.makedirs(date_dir, exist_ok=True)
        
        # Download the zip file to a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, f"unosat_gaza_{date}.zip")
            
            try:
                # Download the file
                urllib.request.urlretrieve(url, zip_path)
                print(f"Downloaded zip file to {zip_path}")
                
                # Extract zip file contents
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    print(f"Extracting files to {date_dir}...")
                    zip_ref.extractall(date_dir)
                
                # Check for GDB files in the extracted directory
                gdb_paths = []
                for root, dirs, files in os.walk(date_dir):
                    for dir_name in dirs:
                        if dir_name.endswith('.gdb'):
                            gdb_path = os.path.join(root, dir_name)
                            gdb_paths.append(gdb_path)
                
                if gdb_paths:
                    print(f"Found {len(gdb_paths)} GDB file(s) for {date}:")
                    for gdb_path in gdb_paths:
                        print(f" - {gdb_path}")
                    success_count += 1
                else:
                    print(f"No GDB files found in the extracted data for {date}.")
                    fail_count += 1
                    
            except Exception as e:
                print(f"Error downloading or extracting UNOSAT data for {date}: {str(e)}")
                fail_count += 1
    
    # Report summary
    print(f"\nDownload summary: {success_count} successful, {fail_count} failed")
    return success_count > 0

if __name__ == "__main__":
    # Get the project root directory
    project_root = Path(__file__).parent.parent.absolute()
    
    # Ask user which data to download
    print("What data would you like to download?")
    print("1. Building footprints data")
    print("2. UNOSAT damage assessment data")
    print("3. Both")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1' or choice == '3':
        # Download building data
        success_building = download_building_data()
        if success_building:
            print("Building data download and extraction completed successfully.")
        else:
            print("Building data download or extraction failed.")
    
    if choice == '2' or choice == '3':
        # For UNOSAT data, ask which dataset to download
        print("\nUNOSAT damage assessment datasets available:")
        unosat_urls = {
            "07_11_2023": "https://unosat.org/static/unosat_filesystem/3734/UNOSAT_GazaStrip_CDA_07Nov2023_GDB.zip", 
            "26_11_2023": "https://unosat.org/static/unosat_filesystem/3769/UNOSAT_GazaStrip_CDA_26November2023_GDB.zip", 
            "06_01_2024": "https://unosat.org/static/unosat_filesystem/3793/UNOSAT_GazaStrip_CDA_January2024_GDB_V2.zip", 
            "01_04_2024": "https://unosat.org/static/unosat_filesystem/3824/OCHA-OPT_013_UNOSAT_GazaStrip_CDA_01Apr2024_GDB.zip", 
            "03_05_2024": "https://unosat.org/static/unosat_filesystem/3861/OCHA_OPT-014_UNOSAT_GazaStrip_OPT_CDA_03May2024_GDB_v2.zip", 
            "06_07_2024": "https://unosat.org/static/unosat_filesystem/3904/OCHA_OPT-015_UNOSAT_GazaStrip_OPT_CDA_06July2024_GDB.zip",
            "06_09_2024": "https://unosat.org/static/unosat_filesystem/3984/OCHA-OPT-017_UNOSAT_A3_Gaza_Strip_OPT_CDA_GDB_06092024.zip",
            "01_12_2024": "https://unosat.org/static/unosat_filesystem/4047/OCHA-OPT_019_UNOSAT_GazaStrip_CDA_GDB_01December2024.zip",
            "25_02_2025": "https://gaza-unosat.docs.cern.ch/CE20231007PSE_UNOSAT_GazaStrip_ComprehensiveDamageAssessment_20250225.zip"
        }
        
        print("Available datasets:")
        for i, date in enumerate(unosat_urls.keys()):
            print(f"{i+1}. {date}")
        print(f"{len(unosat_urls)+1}. All datasets")
        
        dataset_choice = input(f"Enter your choice (1-{len(unosat_urls)+1}): ")
        
        try:
            dataset_idx = int(dataset_choice) - 1
            if dataset_idx >= 0 and dataset_idx < len(unosat_urls):
                # Download a specific dataset
                date_key = list(unosat_urls.keys())[dataset_idx]
                success_unosat = download_unosat_data(date_key)
            elif dataset_idx == len(unosat_urls):
                # Download all datasets
                success_unosat = download_unosat_data()
            else:
                print("Invalid choice. No UNOSAT data will be downloaded.")
                success_unosat = False
        except ValueError:
            print("Invalid input. No UNOSAT data will be downloaded.")
            success_unosat = False
        
        if success_unosat:
            print("UNOSAT data download and extraction completed successfully.")
        else:
            print("UNOSAT data download or extraction failed.")