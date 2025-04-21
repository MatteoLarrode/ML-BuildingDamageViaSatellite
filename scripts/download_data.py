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

def download_unosat_data():
    """Download UNOSAT damage assessment data for Gaza."""
    url = "https://unosat.org/static/unosat_filesystem/4047/OCHA-OPT_019_UNOSAT_GazaStrip_CDA_GDB_01December2024.zip"
    
    print("Downloading UNOSAT Gaza damage assessment data...")
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent.absolute()
    
    # Set the target directory
    labels_dir = os.path.join(project_root, "data", "raw", "labels")
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download the zip file to the temporary directory
        zip_path = os.path.join(temp_dir, "unosat_gaza.zip")
        
        try:
            urllib.request.urlretrieve(url, zip_path)
            print(f"Downloaded zip file to {zip_path}")
            
            # Extract zip file contents
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                print("Files in the zip archive:")
                for file in zip_ref.namelist():
                    print(f" - {file}")
                
                # Extract all files
                print(f"Extracting files to {labels_dir}...")
                zip_ref.extractall(labels_dir)
                
        except Exception as e:
            print(f"Error downloading or extracting UNOSAT data: {str(e)}")
            return False

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
        # Download UNOSAT data
        success_unosat = download_unosat_data()
        if success_unosat:
            print("UNOSAT data download and extraction completed successfully.")
        else:
            print("UNOSAT data download or extraction failed.")