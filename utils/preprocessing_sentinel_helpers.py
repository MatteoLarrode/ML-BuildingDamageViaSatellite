import os
import subprocess
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
import zipfile

def get_aoi_wkt(aoi_path):
    """Get WKT representation of AOI from GeoJSON file"""
    aoi = gpd.read_file(aoi_path)
    if len(aoi) > 0:
        return aoi.geometry.iloc[0].wkt
    else:
        raise ValueError("AOI file contains no geometries")

def create_preprocessing_graph(aoi_wkt, output_path="single_file_graph.xml"):
    """Create a SNAP graph for preprocessing with AOI subsetting"""
    xml_content = f"""<graph id="Graph">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>INPUTFILE</file>
      <formatName>SENTINEL-1</formatName>
    </parameters>
  </node>
  <node id="Apply-Orbit-File">
    <operator>Apply-Orbit-File</operator>
    <sources>
      <sourceProduct refid="Read"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <orbitType>Sentinel Precise (Auto Download)</orbitType>
      <polyDegree>3</polyDegree>
      <continueOnFail>false</continueOnFail>
    </parameters>
  </node>
  <node id="Subset">
    <operator>Subset</operator>
    <sources>
      <sourceProduct refid="Apply-Orbit-File"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <geoRegion>{aoi_wkt}</geoRegion>
      <copyMetadata>true</copyMetadata>
    </parameters>
  </node>
  <node id="ThermalNoiseRemoval">
    <operator>ThermalNoiseRemoval</operator>
    <sources>
      <sourceProduct refid="Subset"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <removeThermalNoise>true</removeThermalNoise>
    </parameters>
  </node>
  <node id="Remove-GRD-Border-Noise">
    <operator>Remove-GRD-Border-Noise</operator>
    <sources>
      <sourceProduct refid="ThermalNoiseRemoval"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <borderLimit>500</borderLimit>
      <trimThreshold>0.5</trimThreshold>
    </parameters>
  </node>
  <node id="Calibration">
    <operator>Calibration</operator>
    <sources>
      <sourceProduct refid="Remove-GRD-Border-Noise"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <outputImageInComplex>false</outputImageInComplex>
      <outputImageScaleInDb>false</outputImageScaleInDb>
      <selectedPolarisations>VV,VH</selectedPolarisations>
      <outputSigmaBand>true</outputSigmaBand>
    </parameters>
  </node>
  <node id="Terrain-Correction">
    <operator>Terrain-Correction</operator>
    <sources>
      <sourceProduct refid="Calibration"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <demName>SRTM 3Sec</demName>
      <imgResamplingMethod>NEAREST_NEIGHBOUR</imgResamplingMethod>
      <pixelSpacingInMeter>10.0</pixelSpacingInMeter>
      <nodataValueAtSea>true</nodataValueAtSea>
      <saveSelectedSourceBand>true</saveSelectedSourceBand>
    </parameters>
  </node>
  <node id="BandSelect">
    <operator>BandSelect</operator>
    <sources>
      <sourceProduct refid="Terrain-Correction"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <selectedPolarisations>VV,VH</selectedPolarisations>
      <sourceBands>Sigma0_VV,Sigma0_VH</sourceBands>
    </parameters>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="BandSelect"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>OUTPUTFILE</file>
      <formatName>GeoTIFF</formatName>
    </parameters>
  </node>
</graph>
    """
    
    with open(output_path, 'w') as f:
        f.write(xml_content)
    
    print(f"Created processing graph at {output_path}")
    return output_path

def update_graph_paths(graph_path, input_file, output_file):
    """Update the input and output file paths in the graph"""
    tree = ET.parse(graph_path)
    root = tree.getroot()
    
    # Update input file
    for read_node in root.findall(".//node[@id='Read']"):
        for params in read_node.findall("./parameters"):
            for file_elem in params.findall("./file"):
                file_elem.text = input_file
    
    # Update output file
    for write_node in root.findall(".//node[@id='Write']"):
        for params in write_node.findall("./parameters"):
            for file_elem in params.findall("./file"):
                file_elem.text = output_file
    
    # Save the modified graph
    modified_path = graph_path.replace('.xml', '_modified.xml')
    tree.write(modified_path)
    
    return modified_path

def extract_date_from_filename(filename):
    """
    Extract acquisition date from Sentinel-1 filename.
    
    Parameters:
    - filename: Sentinel-1 product filename
    
    Returns:
    - Date string in YYYYMMDD format
    """
    parts = os.path.basename(filename).split('_')
    if len(parts) >= 6:
        date_time_str = parts[5]  # Format: YYYYMMDDTHHMMSS
        return date_time_str[:8]  # Return just YYYYMMDD
    else:
        # Handle cases where filename doesn't follow expected pattern
        print(f"Couldn't extract date from filename: {filename}")
        return "unknown_date"

def process_single_file(zip_path, aoi_path, output_dir, snap_gpt_path="gpt"):
    """Process a single Sentinel-1 file with SNAP and convert to dB"""
    print(f"Processing file: {os.path.basename(zip_path)}")
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Get AOI as WKT
    aoi_wkt = get_aoi_wkt(aoi_path)
    
    # Extract filename parts for metadata
    filename = os.path.basename(zip_path)
    date_str = extract_date_from_filename(filename)
    
    # Create output subdirectory with date for organization
    date_dir = os.path.join(output_dir, date_str)
    os.makedirs(date_dir, exist_ok=True)
    
    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create processing graph
        graph_path = create_preprocessing_graph(aoi_wkt, os.path.join(temp_dir, "process_graph.xml"))
        
        # Set output paths for VV and VH bands
        vv_output = os.path.join(date_dir, f"{date_str}_sigma0_vv_db.tif")
        vh_output = os.path.join(date_dir, f"{date_str}_sigma0_vh_db.tif")
        
        # Set temporary output for SNAP processing
        temp_output = os.path.join(temp_dir, "processed_output.tif")
        
        # Update graph with input/output paths
        modified_graph = update_graph_paths(graph_path, zip_path, temp_output)
        
        # Run SNAP processing
        print(f"Starting SNAP processing. This may take a while...")
        cmd = [snap_gpt_path, modified_graph]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("SNAP processing completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error in SNAP processing: {e}")
            print(f"SNAP Error Output: {e.stderr}")
            return None
        
        # Extract VV and VH bands from the output and convert to dB
        if os.path.exists(temp_output):
            with rasterio.open(temp_output) as src:
                # Print information about the output file
                print(f"Output file has {src.count} bands")
                print(f"Band names information available: {src.descriptions is not None}")
                
                # Extract metadata from the source file
                metadata = src.meta.copy()
                tags = src.tags().copy() if hasattr(src, 'tags') else {}
                
                # Add original filename to metadata
                tags['original_file'] = filename
                tags['processing_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Find band indices - use more robust approach
                vv_idx = None
                vh_idx = None
                
                # If descriptions are available, use them
                if src.descriptions:
                    for i, description in enumerate(src.descriptions):
                        if description and "Sigma0_VV" in description:
                            vv_idx = i + 1
                        elif description and "Sigma0_VH" in description:
                            vh_idx = i + 1
                
                # If descriptions don't work, try band names
                if (vv_idx is None or vh_idx is None) and hasattr(src, 'tags'):
                    for i in range(1, src.count + 1):
                        band_name = src.tags(i).get('name', '')
                        if "Sigma0_VV" in band_name:
                            vv_idx = i
                        elif "Sigma0_VH" in band_name:
                            vh_idx = i
                
                # If still not found, use heuristic for 2-band images
                if (vv_idx is None or vh_idx is None) and src.count == 2:
                    print("Using band position heuristic (assuming band 1 is VV, band 2 is VH)")
                    vv_idx = 1
                    vh_idx = 2
                
                # Extract, convert to dB, and save each band
                if vv_idx:
                    print(f"Found VV band at index {vv_idx}")
                    vv_data = src.read(vv_idx)
                    
                    # Convert to dB: 10 * log10(value)
                    # Handle zero/negative values by setting a floor
                    vv_data_db = 10 * np.log10(np.maximum(vv_data, 0.0000001))
                    
                    # Update metadata
                    meta = metadata.copy()
                    meta.update(count=1, dtype='float32')
                    
                    # Add band-specific metadata
                    band_tags = src.tags(vv_idx).copy() if hasattr(src, 'tags') else {}
                    band_tags.update({
                        'units': 'dB',
                        'original_polarization': 'VV',
                        'conversion': 'sigma0_linear_to_dB'
                    })
                    
                    with rasterio.open(vv_output, 'w', **meta) as dst:
                        dst.write(vv_data_db, 1)
                        dst.update_tags(**tags)  # File-level tags
                        dst.update_tags(1, **band_tags)  # Band-level tags
                    
                    print(f"Saved VV band (dB) to {vv_output}")
                else:
                    print("Warning: Could not identify VV band")
                
                if vh_idx:
                    print(f"Found VH band at index {vh_idx}")
                    vh_data = src.read(vh_idx)
                    
                    # Convert to dB: 10 * log10(value)
                    # Handle zero/negative values by setting a floor
                    vh_data_db = 10 * np.log10(np.maximum(vh_data, 0.0000001))
                    
                    # Update metadata
                    meta = metadata.copy()
                    meta.update(count=1, dtype='float32')
                    
                    # Add band-specific metadata
                    band_tags = src.tags(vh_idx).copy() if hasattr(src, 'tags') else {}
                    band_tags.update({
                        'units': 'dB',
                        'original_polarization': 'VH',
                        'conversion': 'sigma0_linear_to_dB'
                    })
                    
                    with rasterio.open(vh_output, 'w', **meta) as dst:
                        dst.write(vh_data_db, 1)
                        dst.update_tags(**tags)  # File-level tags
                        dst.update_tags(1, **band_tags)  # Band-level tags
                    
                    print(f"Saved VH band (dB) to {vh_output}")
                else:
                    print("Warning: Could not identify VH band")
            
            # Create a metadata JSON file with extended information
            metadata_file = os.path.join(date_dir, f"{date_str}_metadata.json")
            try:
                import json
                # Extract additional metadata from filename
                file_parts = filename.split('_')
                metadata_dict = {
                    'original_file': filename,
                    'acquisition_date': date_str,
                    'satellite': file_parts[0] if len(file_parts) > 0 else 'Unknown',
                    'mode': file_parts[1] if len(file_parts) > 1 else 'Unknown',
                    'product_type': file_parts[2] if len(file_parts) > 2 else 'Unknown',
                    'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'output_files': {
                        'VV': os.path.basename(vv_output) if vv_idx else None,
                        'VH': os.path.basename(vh_output) if vh_idx else None
                    }
                }
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata_dict, f, indent=2)
                
                print(f"Saved metadata to {metadata_file}")
            except Exception as e:
                print(f"Error saving metadata: {e}")
            
            return {
                "VV": vv_output if vv_idx else None,
                "VH": vh_output if vh_idx else None,
                "metadata": metadata_file
            }
        else:
            print(f"Error: Expected output file {temp_output} was not created")
            return None