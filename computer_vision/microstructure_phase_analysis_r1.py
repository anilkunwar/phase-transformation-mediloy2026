import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageOps
from skimage.measure import regionprops, label
import io
import os
import re
from pathlib import Path
import traceback
import sys
import json
from datetime import datetime

# Enable PIL to load truncated or partially corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None

# Page Configuration
st.set_page_config(page_title="Microstructure Phase Analyzer", layout="wide", page_icon="🔬")

st.title("🔬 Microstructure Phase Analyzer (HCP ε / FCC γ)")
st.markdown("""
This tool analyzes two-phase microstructure images with automatic experimental condition extraction.
It calculates **phase fractions**, **area fractions**, and morphological metrics for **HCP epsilon (ε)** and **FCC gamma (γ)** phases.

**Supported filename pattern:** `ABN.ext` where:
- `A` = C (Continuous Wave) or P (Pulse Laser)
- `B` = NH (Not Heated, t=0) or H (Heated, t=35 min)  
- `N` = 0 (0°) or 45 (45° orientation)
""")

# --- Helper Functions ---

def parse_filename(filename):
    """Parse filename according to convention: ABN.ext"""
    name_without_ext = Path(filename).stem
    pattern = r'^([CP])(NH|H)(\d+)$'
    match = re.match(pattern, name_without_ext, re.IGNORECASE)
    
    if match:
        laser_type = match.group(1).upper()
        heating = match.group(2).upper()
        orientation = match.group(3)
        
        laser_desc = "Continuous Wave Laser" if laser_type == 'C' else "Pulse Type Laser" if laser_type == 'P' else "Unknown"
        
        if heating == 'NH':
            heating_desc, time_desc = "Not Heated (Pre-heated)", "t = 0 min"
        elif heating == 'H':
            heating_desc, time_desc = "Heated Isothermally", "t = 35 min"
        else:
            heating_desc, time_desc = "Unknown", "Unknown"
        
        orientation_desc = "0° (Reference)" if orientation == '0' else "45° (Rotated)" if orientation == '45' else f"{orientation}°"
        
        return {
            'laser_type': laser_type,
            'laser_description': laser_desc,
            'heating_code': heating,
            'heating_description': heating_desc,
            'time': time_desc,
            'orientation': orientation,
            'orientation_description': orientation_desc,
            'valid': True
        }
    else:
        return {
            'valid': False,
            'message': f"Filename '{filename}' does not match expected pattern (e.g., CNH0.bmp, PH45.bmp)"
        }

def get_script_directory():
    """Get the directory where the script is located"""
    try:
        if getattr(sys, 'frozen', False):
            script_dir = os.path.dirname(sys.executable)
        elif '__file__' in globals():
            script_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            script_dir = os.getcwd()
        return script_dir
    except Exception as e:
        st.warning(f"Could not determine script directory: {e}")
        return os.getcwd()

def scan_images_folder(folder_path="./images"):
    """Scan the images folder and return list of valid image files"""
    supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    images = []
    abs_folder_path = os.path.abspath(folder_path)
    
    st.sidebar.write(f"**🔍 Searching:** `{abs_folder_path}`")
    st.sidebar.write(f"**✅ Folder exists:** {os.path.exists(abs_folder_path)}")
    
    if os.path.exists(abs_folder_path):
        try:
            all_files = os.listdir(abs_folder_path)
            st.sidebar.write(f"**📁 Total items:** {len(all_files)}")
            
            if len(all_files) > 0:
                st.sidebar.write("**📋 Sample files:**")
                for idx, f in enumerate(all_files[:8]):
                    ext = Path(f).suffix.lower()
                    icon = "🖼️" if ext in supported_extensions else "📄"
                    st.sidebar.text(f"  {icon} {f}")
                if len(all_files) > 8:
                    st.sidebar.text(f"  ... and {len(all_files) - 8} more")
            
            for filename in all_files:
                file_path = os.path.join(abs_folder_path, filename)
                if os.path.isfile(file_path):
                    ext = Path(filename).suffix.lower()
                    if ext in supported_extensions:
                        images.append(filename)
                        st.sidebar.success(f"✓ Found: {filename}")
            
            if len(images) == 0:
                st.sidebar.warning("⚠️ No image files found with supported extensions")
                st.sidebar.write(f"**Supported:** {', '.join(sorted(supported_extensions))}")
        except PermissionError:
            st.sidebar.error("❌ Permission denied to read folder")
        except Exception as e:
            st.sidebar.error(f"❌ Error reading folder: {e}")
            st.sidebar.code(traceback.format_exc())
    else:
        st.sidebar.error(f"❌ Folder not found: {abs_folder_path}")
        try:
            os.makedirs(abs_folder_path, exist_ok=True)
            st.sidebar.info(f"📁 Created folder: {abs_folder_path}")
            st.sidebar.warning("Please add image files and refresh the page")
        except Exception as e:
            st.sidebar.error(f"❌ Could not create folder: {e}")
    
    return sorted(images)

def load_image_robust(image_path):
    """Ultra-robust image loading with multiple fallbacks"""
    st.info(f"🔄 Loading: {os.path.basename(image_path)}")
    st.write(f"📍 Path: `{image_path}`")
    
    # Method 1: PIL with maximum tolerance
    try:
        st.write("  → Method 1: PIL direct load...")
        img = Image.open(image_path)
        img.load()
        if img.mode in ('RGBA', 'LA', 'P', 'PA', '1', 'L'):
            if img.mode == 'P':
                img = img.convert('RGBA')
            if img.mode in ('RGBA', 'LA', 'PA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'PA':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
                img = background
            else:
                img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        st.success("  ✓ PIL load successful")
        return img
    except Exception as e1:
        st.warning(f"  ✗ PIL failed: {type(e1).__name__}: {e1}")
    
    # Method 2: OpenCV with aggressive flags
    try:
        st.write("  → Method 2: OpenCV with IMREAD flags...")
        for flag in [cv2.IMREAD_COLOR, cv2.IMREAD_UNCHANGED, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH]:
            img_cv = cv2.imread(image_path, flag)
            if img_cv is not None and img_cv.size > 0:
                if len(img_cv.shape) == 2:
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
                elif img_cv.shape[2] == 4:
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGB)
                elif img_cv.shape[2] == 3:
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img_cv)
                st.success(f"  ✓ OpenCV load successful (flag: {flag})")
                return img
        st.warning("  ✗ OpenCV returned None for all flags")
    except Exception as e2:
        st.warning(f"  ✗ OpenCV failed: {type(e2).__name__}: {e2}")
    
    # Method 3: Raw bytes + PIL from memory
    try:
        st.write("  → Method 3: Raw bytes + PIL memory load...")
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        if len(img_bytes) == 0:
            st.warning("  ✗ File is empty")
        else:
            buffer = io.BytesIO(img_bytes)
            img = Image.open(buffer)
            img.load()
            if img.mode in ('RGBA', 'LA', 'P', 'PA', '1', 'L'):
                if img.mode == 'P':
                    img = img.convert('RGBA')
                if img.mode in ('RGBA', 'LA', 'PA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'PA':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
                    img = background
                else:
                    img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            st.success("  ✓ Bytes + PIL successful")
            return img
    except Exception as e3:
        st.warning(f"  ✗ Bytes load failed: {type(e3).__name__}: {e3}")
    
    # Method 4: BMP → PNG conversion in memory
    file_ext = Path(image_path).suffix.lower()
    if file_ext == '.bmp':
        try:
            st.write("  → Method 4: BMP → PNG conversion in memory...")
            img_cv = None
            for flag in [cv2.IMREAD_COLOR, cv2.IMREAD_UNCHANGED]:
                img_cv = cv2.imread(image_path, flag)
                if img_cv is not None and img_cv.size > 0:
                    break
            
            if img_cv is not None and img_cv.size > 0:
                if len(img_cv.shape) == 2:
                    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
                elif img_cv.shape[2] == 4:
                    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGB)
                else:
                    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                
                pil_img = Image.fromarray(img_rgb)
                png_buffer = io.BytesIO()
                pil_img.save(png_buffer, format='PNG', compress_level=6)
                png_buffer.seek(0)
                
                final_img = Image.open(png_buffer)
                if final_img.mode != 'RGB':
                    final_img = final_img.convert('RGB')
                st.success("  ✓ BMP → PNG conversion successful")
                return final_img
            else:
                st.warning("  ✗ OpenCV could not read BMP file")
        except Exception as e4:
            st.warning(f"  ✗ BMP conversion failed: {type(e4).__name__}: {e4}")
    
    # Method 5: imageio fallback
    try:
        import imageio
        st.write("  → Method 5: imageio fallback...")
        img_array = imageio.imread(image_path)
        if img_array is not None:
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)
            elif img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_array)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            st.success("  ✓ imageio fallback successful")
            return img
    except ImportError:
        st.write("  → imageio not available")
    except Exception as e5:
        st.warning(f"  ✗ imageio failed: {type(e5).__name__}: {e5}")
    
    # All methods failed
    st.error(f"❌ CRITICAL: All loading methods failed for: {os.path.basename(image_path)}")
    st.error(f"📍 Full path: {image_path}")
    st.error(f"📏 File size: {os.path.getsize(image_path) if os.path.exists(image_path) else 'N/A'} bytes")
    
    with st.expander("🔍 Full Error Traceback"):
        st.code(traceback.format_exc())
    
    with st.expander("💡 Troubleshooting"):
        st.write("""
        1. **Convert BMP to PNG locally**:
           ```python
           from PIL import Image
           img = Image.open('CH0.bmp')
           img.save('CH0.png')
           ```
        2. **Check file integrity** in an image viewer
        3. **Use upload option** as workaround
        """)
    
    return None

def create_overlay_safe(img_np, mask_red, mask_green):
    """Create segmentation overlay with proper dtype handling - FIXES THE VALUE ERROR"""
    # Convert to float32 for safe arithmetic operations
    overlay = img_np.astype(np.float32).copy()
    
    # Create 3D masks for broadcasting
    red_mask_3d = np.stack([mask_red] * 3, axis=-1).astype(bool)
    green_mask_3d = np.stack([mask_green] * 3, axis=-1).astype(bool)
    
    # Apply red tint for HCP phase (with proper float arithmetic)
    if np.any(red_mask_3d):
        red_tint = np.array([255, 80, 80], dtype=np.float32)
        overlay[red_mask_3d] = overlay[red_mask_3d] * 0.6 + red_tint * 0.4
    
    # Apply green tint for FCC phase
    if np.any(green_mask_3d):
        green_tint = np.array([80, 255, 80], dtype=np.float32)
        overlay[green_mask_3d] = overlay[green_mask_3d] * 0.6 + green_tint * 0.4
    
    # Clip to valid range and convert back to uint8
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return overlay

def generate_radar_chart_data(df_all, area_per_pixel):
    """Generate data for radar chart comparing phase metrics"""
    if df_all.empty:
        return None
    
    radar_data = []
    
    for phase in df_all['Phase'].unique():
        phase_df = df_all[df_all['Phase'] == phase]
        
        metrics = {
            'Phase': phase.replace(' (Red)', '').replace(' (Green)', ''),
            'Grain Count': len(phase_df),
            'Avg Area (µm²)': phase_df['Area (µm²)'].mean(),
            'Avg ECD (µm)': phase_df['ECD (µm)'].mean(),
            'Avg Circularity': phase_df['Circularity'].mean(),
            'Avg Aspect Ratio': phase_df['Aspect Ratio'].mean(),
            'Total Area (µm²)': phase_df['Area (µm²)'].sum()
        }
        radar_data.append(metrics)
    
    return pd.DataFrame(radar_data)

def generate_chord_data(df_all, red_fraction, green_fraction, boundary_fraction):
    """Generate data for chord diagram showing phase relationships"""
    # Nodes: phases and their interactions
    nodes = [
        {"name": "HCP ε", "color": "#e74c3c"},
        {"name": "FCC γ", "color": "#2ecc71"},
        {"name": "Boundaries", "color": "#95a5a6"}
    ]
    
    # Links: represent area fractions and morphological relationships
    links = []
    
    # Add fraction-based links (normalized to 100)
    total = red_fraction + green_fraction + boundary_fraction
    if total > 0:
        # Self-loops representing phase abundance
        links.append({"source": 0, "target": 0, "value": red_fraction / total * 100, "label": f"HCP ε: {red_fraction:.1f}%"})
        links.append({"source": 1, "target": 1, "value": green_fraction / total * 100, "label": f"FCC γ: {green_fraction:.1f}%"})
        links.append({"source": 2, "target": 2, "value": boundary_fraction / total * 100, "label": f"Boundaries: {boundary_fraction:.1f}%"})
        
        # Cross-phase interaction links (representing interface area)
        interface_strength = min(red_fraction, green_fraction) * 0.5  # Simplified metric
        if interface_strength > 0:
            links.append({"source": 0, "target": 1, "value": interface_strength, "label": "HCP-FCC Interface"})
            links.append({"source": 1, "target": 0, "value": interface_strength, "label": "FCC-HCP Interface"})
    
    # Add morphological correlation links if data available
    if not df_all.empty:
        hcp_df = df_all[df_all['Phase'].str.contains('HCP', case=False, na=False)]
        fcc_df = df_all[df_all['Phase'].str.contains('FCC', case=False, na=False)]
        
        if not hcp_df.empty and not fcc_df.empty:
            # Correlation based on circularity similarity
            hcp_circ = hcp_df['Circularity'].mean()
            fcc_circ = fcc_df['Circularity'].mean()
            circ_similarity = 1 - abs(hcp_circ - fcc_circ)
            
            if circ_similarity > 0.3:
                links.append({
                    "source": 0, 
                    "target": 1, 
                    "value": circ_similarity * 30, 
                    "label": f"Shape Similarity: {circ_similarity:.2f}"
                })
    
    return {"nodes": nodes, "links": links}

def create_comprehensive_csv(df_all, red_fraction, green_fraction, boundary_fraction, 
                            red_area_abs, green_area_abs, boundary_area_abs,
                            image_info, filename, domain_size_um, area_per_pixel):
    """Create comprehensive CSV with all analysis results"""
    results = []
    
    # Metadata section
    results.append({"Category": "METADATA", "Parameter": "Filename", "Value": filename, "Unit": ""})
    results.append({"Category": "METADATA", "Parameter": "Analysis Timestamp", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Unit": ""})
    results.append({"Category": "METADATA", "Parameter": "Domain Size", "Value": domain_size_um, "Unit": "µm"})
    results.append({"Category": "METADATA", "Parameter": "Area per Pixel", "Value": area_per_pixel, "Unit": "µm²/px"})
    
    # Experimental conditions if parsed
    if image_info and image_info.get('valid'):
        results.append({"Category": "EXPERIMENT", "Parameter": "Laser Type", "Value": image_info['laser_description'], "Unit": ""})
        results.append({"Category": "EXPERIMENT", "Parameter": "Heating Condition", "Value": image_info['heating_description'], "Unit": ""})
        results.append({"Category": "EXPERIMENT", "Parameter": "Isothermal Time", "Value": image_info['time'], "Unit": ""})
        results.append({"Category": "EXPERIMENT", "Parameter": "Orientation", "Value": f"{image_info['orientation']}°", "Unit": "degrees"})
    
    # Phase fractions
    results.append({"Category": "PHASE_FRACTIONS", "Parameter": "HCP ε Fraction", "Value": red_fraction, "Unit": "%"})
    results.append({"Category": "PHASE_FRACTIONS", "Parameter": "FCC γ Fraction", "Value": green_fraction, "Unit": "%"})
    results.append({"Category": "PHASE_FRACTIONS", "Parameter": "Boundary Fraction", "Value": boundary_fraction, "Unit": "%"})
    
    # Absolute areas
    results.append({"Category": "ABSOLUTE_AREAS", "Parameter": "HCP ε Area", "Value": red_area_abs, "Unit": "µm²"})
    results.append({"Category": "ABSOLUTE_AREAS", "Parameter": "FCC γ Area", "Value": green_area_abs, "Unit": "µm²"})
    results.append({"Category": "ABSOLUTE_AREAS", "Parameter": "Boundary Area", "Value": boundary_area_abs, "Unit": "µm²"})
    
    # Summary statistics per phase
    if not df_all.empty:
        for phase in df_all['Phase'].unique():
            phase_df = df_all[df_all['Phase'] == phase]
            phase_name = phase.replace(' (Red)', '').replace(' (Green)', '')
            
            results.append({"Category": f"SUMMARY_{phase_name}", "Parameter": "Grain Count", "Value": len(phase_df), "Unit": "count"})
            results.append({"Category": f"SUMMARY_{phase_name}", "Parameter": "Mean Area", "Value": phase_df['Area (µm²)'].mean(), "Unit": "µm²"})
            results.append({"Category": f"SUMMARY_{phase_name}", "Parameter": "Std Area", "Value": phase_df['Area (µm²)'].std(), "Unit": "µm²"})
            results.append({"Category": f"SUMMARY_{phase_name}", "Parameter": "Mean ECD", "Value": phase_df['ECD (µm)'].mean(), "Unit": "µm"})
            results.append({"Category": f"SUMMARY_{phase_name}", "Parameter": "Mean Circularity", "Value": phase_df['Circularity'].mean(), "Unit": ""})
            results.append({"Category": f"SUMMARY_{phase_name}", "Parameter": "Mean Aspect Ratio", "Value": phase_df['Aspect Ratio'].mean(), "Unit": ""})
    
    # Individual grain data
    for _, row in df_all.iterrows():
        results.append({
            "Category": "GRAIN_DATA",
            "Parameter": f"Grain_{row['Grain ID']}",
            "Value": json.dumps({
                "phase": row['Phase'],
                "area_um2": row['Area (µm²)'],
                "ecd_um": row['ECD (µm)'],
                "circularity": row['Circularity'],
                "aspect_ratio": row['Aspect Ratio']
            }),
            "Unit": "JSON"
        })
    
    return pd.DataFrame(results)

# --- Sidebar Controls ---
st.sidebar.header("📁 Image Source")

# Directory information
st.sidebar.subheader("📍 Environment Info")
script_dir = get_script_directory()
cwd = os.getcwd()
st.sidebar.write(f"**📜 Script dir:** `{script_dir}`")
st.sidebar.write(f"**🔄 Working dir:** `{cwd}`")
st.sidebar.write(f"**🎯 Same location:** {script_dir == cwd}")
st.sidebar.write(f"**☁️ Streamlit Cloud:** {'Yes' if '/mount/src/' in cwd else 'No (Local)'}")

# Image source selection
source_option = st.sidebar.radio(
    "Select Image Source:",
    ["📂 From ./images/ Folder", "📤 Upload Custom Image"]
)

uploaded_file = None
selected_folder_image = None
image_info = None
images_folder_path = None
selected_filename = None

if source_option == "📂 From ./images/ Folder":
    st.sidebar.subheader("🔍 Searching for Images")
    
    possible_paths = [
        "./images",
        "images", 
        os.path.join(script_dir, "images"),
        os.path.join(cwd, "images"),
        "/mount/src/phase-transformation-mediloy2026/computer_vision/images",
    ]
    
    available_images = []
    images_path_used = None
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and os.path.isdir(abs_path):
            st.sidebar.success(f"✅ Found: `{path}`")
            images_path_used = path
            available_images = scan_images_folder(path)
            images_folder_path = abs_path
            break
        else:
            st.sidebar.text(f"❌ Not found: `{path}`")
    
    if not available_images:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 Troubleshooting")
        
        st.sidebar.write("**🗂️ Checked paths:**")
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            exists = os.path.exists(abs_path)
            is_dir = os.path.isdir(abs_path) if exists else False
            status = "✅" if (exists and is_dir) else "❌"
            st.sidebar.text(f"  {status} {abs_path}")
        
        st.sidebar.write("**📋 Script directory contents:**")
        try:
            script_contents = os.listdir(script_dir)
            for idx, item in enumerate(sorted(script_contents)[:12]):
                item_path = os.path.join(script_dir, item)
                is_dir = os.path.isdir(item_path)
                icon = "📁" if is_dir else "📄"
                st.sidebar.text(f"  {icon} {item}")
            if len(script_contents) > 12:
                st.sidebar.text(f"  ... and {len(script_contents) - 12} more")
        except Exception as e:
            st.sidebar.error(f"Cannot list: {e}")
        
        st.sidebar.write("**➕ Create images folder?**")
        if st.sidebar.button("📁 Create ./images folder now"):
            try:
                new_folder = os.path.join(script_dir, "images")
                os.makedirs(new_folder, exist_ok=True)
                st.sidebar.success(f"✅ Created: {new_folder}")
                st.sidebar.info("🔄 Refresh page after adding images")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Failed: {e}")
    
    if available_images:
        st.sidebar.markdown("---")
        st.sidebar.success(f"✅ Found {len(available_images)} image(s)")
        
        selected_filename = st.sidebar.selectbox(
            "🖼️ Select an image:",
            available_images,
            index=0 if len(available_images) == 1 else None
        )
        
        if selected_filename:
            image_info = parse_filename(selected_filename)
            
            if images_folder_path:
                image_path = os.path.join(images_folder_path, selected_filename)
            elif images_path_used:
                image_path = os.path.join(images_path_used, selected_filename)
            else:
                image_path = os.path.abspath(os.path.join("./images", selected_filename))
            
            st.sidebar.write(f"**🔗 Loading:** `{selected_filename}`")
            
            selected_folder_image = load_image_robust(image_path)
            
            if selected_folder_image is not None:
                class MockUploadedFile:
                    def __init__(self, name):
                        self.name = name
                uploaded_file = MockUploadedFile(selected_filename)
                st.sidebar.success(f"✅ Loaded successfully!")
                st.sidebar.write(f"**🎨 Mode:** {selected_folder_image.mode}")
                st.sidebar.write(f"**📐 Size:** {selected_folder_image.size[0]} × {selected_folder_image.size[1]} px")
            else:
                st.sidebar.error(f"❌ Failed to load: {selected_filename}")
                st.sidebar.info("💡 Try: Upload manually or convert BMP→PNG locally")
else:
    uploaded_file = st.sidebar.file_uploader(
        "📤 Upload Microstructure Image", 
        type=["png", "jpg", "jpeg", "bmp", "tiff", "tif"],
        help="Supports PNG, JPG, JPEG, BMP, TIFF formats"
    )
    
    if uploaded_file:
        image_info = parse_filename(uploaded_file.name)
        
        try:
            selected_folder_image = Image.open(uploaded_file)
            if selected_folder_image.mode in ('RGBA', 'LA', 'P', 'PA'):
                if selected_folder_image.mode == 'P':
                    selected_folder_image = selected_folder_image.convert('RGBA')
                if selected_folder_image.mode in ('RGBA', 'LA', 'PA'):
                    background = Image.new('RGB', selected_folder_image.size, (255, 255, 255))
                    if selected_folder_image.mode == 'PA':
                        selected_folder_image = selected_folder_image.convert('RGBA')
                    background.paste(selected_folder_image, mask=selected_folder_image.split()[3] if selected_folder_image.mode == 'RGBA' else None)
                    selected_folder_image = background
                else:
                    selected_folder_image = selected_folder_image.convert('RGB')
            elif selected_folder_image.mode != 'RGB':
                selected_folder_image = selected_folder_image.convert('RGB')
            st.sidebar.success(f"✅ Uploaded: {uploaded_file.name}")
        except Exception as e:
            st.sidebar.error(f"❌ Upload failed: {e}")
            try:
                uploaded_file.seek(0)
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img_cv is not None:
                    if len(img_cv.shape) == 2:
                        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
                    elif img_cv.shape[2] == 4:
                        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGB)
                    else:
                        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                    selected_folder_image = Image.fromarray(img_cv)
                    st.sidebar.success("✅ Loaded via OpenCV fallback")
            except Exception as e2:
                st.sidebar.error(f"❌ Fallback also failed: {e2}")

# --- Display Experimental Conditions ---
if image_info and image_info.get('valid'):
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 Experimental Conditions")
    
    col_exp1, col_exp2 = st.sidebar.columns(2)
    
    with col_exp1:
        st.metric("🔦 Laser Type", image_info['laser_type'], image_info['laser_description'])
        st.metric("🔥 Heating", image_info['heating_code'], image_info['heating_description'])
    
    with col_exp2:
        st.metric("📐 Orientation", f"{image_info['orientation']}°", image_info['orientation_description'])
        st.metric("⏱️ Time", image_info['time'])
    
    with st.sidebar.expander("📋 Full Experimental Details"):
        filename_display = uploaded_file.name if uploaded_file else (selected_filename if selected_filename else 'N/A')
        st.write(f"""
        | Parameter | Value | Description |
        |-----------|-------|-------------|
        | **Laser Type** | {image_info['laser_type']} | {image_info['laser_description']} |
        | **Heating** | {image_info['heating_code']} | {image_info['heating_description']} |
        | **Time** | {image_info['time']} | Isothermal hold duration |
        | **Orientation** | {image_info['orientation']}° | {image_info['orientation_description']} |
        | **Filename** | `{filename_display}` | Source image file |
        """)

elif image_info and not image_info.get('valid'):
    st.sidebar.warning(f"⚠️ {image_info.get('message', 'Could not parse filename')}")
    st.sidebar.info("💡 Expected pattern: `CNH0.bmp`, `PH45.png`, etc.")

# --- Main Configuration ---
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Analysis Configuration")

st.sidebar.subheader("📏 Physical Calibration")
domain_size_um = st.sidebar.number_input(
    "Total Domain Length (µm)", 
    value=250.0,
    min_value=10.0,
    max_value=10000.0,
    step=10.0,
    help="Physical length of the square image side in micrometers"
)
st.sidebar.info(f"📐 Domain: {domain_size_um} × {domain_size_um} µm = {domain_size_um**2:,.0f} µm²")

st.sidebar.subheader("🔬 Analysis Options")
exclude_boundaries = st.sidebar.checkbox(
    "🚫 Exclude grain boundaries from area fraction", 
    value=False,
    help="When checked: HCP ε + FCC γ = 100% (boundaries excluded from normalization)"
)

st.sidebar.subheader("🎨 Segmentation Settings")
use_auto_threshold = st.sidebar.checkbox("🤖 Use automatic HSV color detection", value=True)

st.sidebar.subheader("🔧 Mask Refinement")
apply_morphology = st.sidebar.checkbox("✨ Apply morphological cleaning", value=True, help="Remove noise and smooth grain boundaries")

if apply_morphology:
    kernel_size = st.sidebar.slider("🔲 Kernel size", 1, 21, 3, step=2, help="Odd values recommended for symmetry")
    morphology_iterations = st.sidebar.slider("🔄 Iterations", 1, 5, 1, help="Number of open/close operations")

# --- Main Processing Logic ---

if uploaded_file is not None and selected_folder_image is not None:
    # Convert PIL to numpy array
    img_np = np.array(selected_folder_image)
    
    # Handle alpha channel
    if img_np.ndim == 3 and img_np.shape[2] == 4:
        st.info("🎨 Detected alpha channel - converting to RGB")
        img_np = img_np[:, :, :3]
    
    # Handle grayscale
    if img_np.ndim == 2:
        st.info("⚫ Detected grayscale - converting to RGB")
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    h, w, _ = img_np.shape
    
    # Calculate calibration
    total_physical_area = domain_size_um ** 2
    total_pixels = h * w
    area_per_pixel = total_physical_area / total_pixels
    
    # Display image metrics
    col_img1, col_img2, col_img3 = st.columns(3)
    with col_img1:
        st.metric("🖼️ Resolution", f"{w} × {h}", "pixels")
    with col_img2:
        st.metric("📐 Calibration", f"{area_per_pixel:.4f}", "µm²/pixel")
    with col_img3:
        st.metric("🌐 Total Area", f"{total_physical_area:,.0f}", "µm²")
    
    # Show original image in expander
    with st.expander("🖼️ View Original Image"):
        st.image(img_np, caption=f"Original: {uploaded_file.name if uploaded_file else selected_filename}", use_column_width=True)
    
    # Convert to HSV for segmentation
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    
    # HSV ranges for HCP epsilon (Red phase)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask1, mask2)
    
    # HSV ranges for FCC gamma (Green phase)
    lower_green = np.array([40, 70, 50])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations if enabled
    if apply_morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        for _ in range(morphology_iterations):
            mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
            mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
            mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
            mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
        
        st.sidebar.info(f"✨ Applied {morphology_iterations}× morphology with {kernel_size}×{kernel_size} kernel")
    
    # --- Quantification Calculations ---
    
    red_pixels = int(cv2.countNonZero(mask_red))
    green_pixels = int(cv2.countNonZero(mask_green))
    boundary_pixels = total_pixels - red_pixels - green_pixels
    
    red_area_abs = red_pixels * area_per_pixel
    green_area_abs = green_pixels * area_per_pixel
    boundary_area_abs = boundary_pixels * area_per_pixel
    
    # Area fractions based on mode
    if exclude_boundaries:
        total_phase_pixels = red_pixels + green_pixels
        if total_phase_pixels > 0:
            red_fraction = (red_pixels / total_phase_pixels) * 100
            green_fraction = (green_pixels / total_phase_pixels) * 100
        else:
            red_fraction = green_fraction = 0
        boundary_fraction = 0
        st.info("📊 **Mode:** Normalized fractions (HCP ε + FCC γ = 100%)")
    else:
        red_fraction = (red_pixels / total_pixels) * 100
        green_fraction = (green_pixels / total_pixels) * 100
        boundary_fraction = (boundary_pixels / total_pixels) * 100
        st.info("📊 **Mode:** Absolute fractions (includes boundaries)")
    
    # Morphological analysis function
    def analyze_morphology(mask, phase_name, area_per_pixel):
        labeled = label(mask)
        props = regionprops(labeled)
        
        if not props:
            return pd.DataFrame()
        
        data = []
        for prop in props:
            if prop.area < 5:
                continue
            
            area_um2 = prop.area * area_per_pixel
            ecd_um = np.sqrt(4 * area_um2 / np.pi)
            
            perimeter = max(prop.perimeter, 0.001)
            circularity = (4 * np.pi * prop.area) / (perimeter ** 2)
            
            if prop.major_axis_length > 0 and prop.minor_axis_length > 0:
                aspect_ratio = prop.major_axis_length / prop.minor_axis_length
            else:
                aspect_ratio = 1.0
            
            data.append({
                "Phase": phase_name,
                "Grain ID": prop.label,
                "Area (µm²)": round(area_um2, 3),
                "ECD (µm)": round(ecd_um, 3),
                "Circularity": round(circularity, 4),
                "Aspect Ratio": round(aspect_ratio, 3)
            })
        
        return pd.DataFrame(data)
    
    df_red = analyze_morphology(mask_red, "HCP ε (Red)", area_per_pixel)
    df_green = analyze_morphology(mask_green, "FCC γ (Green)", area_per_pixel)
    df_all = pd.concat([df_red, df_green], ignore_index=True)
    
    # --- Results Display ---
    
    st.markdown("---")
    st.subheader("📊 Phase Quantification Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔢 Absolute Areas")
        k1, k2, k3 = st.columns(3)
        k1.metric("🔴 HCP ε", f"{red_area_abs:.1f} µm²", f"{red_pixels:,} px")
        k2.metric("🟢 FCC γ", f"{green_area_abs:.1f} µm²", f"{green_pixels:,} px")
        k3.metric("⚪ Boundary", f"{boundary_area_abs:.1f} µm²", f"{boundary_pixels:,} px")
        
        st.divider()
        
        st.markdown("### 📈 Area Fractions")
        if exclude_boundaries:
            f1, f2 = st.columns(2)
            f1.metric("🔴 HCP ε", f"{red_fraction:.2f}%", "Normalized")
            f2.metric("🟢 FCC γ", f"{green_fraction:.2f}%", "Normalized")
            st.caption("✅ HCP ε + FCC γ = 100%")
        else:
            f1, f2, f3 = st.columns(3)
            f1.metric("🔴 HCP ε", f"{red_fraction:.1f}%")
            f2.metric("🟢 FCC γ", f"{green_fraction:.1f}%")
            f3.metric("⚪ Boundary", f"{boundary_fraction:.1f}%")
            st.caption("📏 Includes grain boundaries")
        
        # Bar chart
        st.markdown("### 📊 Phase Distribution")
        if exclude_boundaries:
            chart_df = pd.DataFrame({
                "Phase": ["HCP ε", "FCC γ"],
                "Area Fraction (%)": [red_fraction, green_fraction]
            })
        else:
            chart_df = pd.DataFrame({
                "Phase": ["HCP ε", "FCC γ", "Boundaries"],
                "Area Fraction (%)": [red_fraction, green_fraction, boundary_fraction]
            })
        st.bar_chart(chart_df.set_index("Phase"), use_container_width=True)
    
    with col2:
        st.markdown("### 🎭 Segmentation Masks")
        c1, c2 = st.columns(2)
        with c1:
            st.image(mask_red, caption="🔴 HCP ε Mask", use_column_width=True)
        with c2:
            st.image(mask_green, caption="🟢 FCC γ Mask", use_column_width=True)
        
        # Overlay visualization - FIXED VERSION
        with st.expander("🔍 View Segmentation Overlay"):
            try:
                overlay = create_overlay_safe(img_np, mask_red, mask_green)
                st.image(overlay, caption="Overlay: 🔴 HCP ε + 🟢 FCC γ", use_column_width=True)
            except Exception as e:
                st.error(f"❌ Overlay creation failed: {e}")
                st.code(traceback.format_exc())
                # Fallback: show masks side by side
                st.info("Showing individual masks instead")
    
    # --- Advanced Visualizations ---
    
    st.markdown("---")
    st.subheader("📈 Advanced Analytics")
    
    # Radar Chart for Phase Comparison
    st.markdown("### 🎯 Radar Chart: Phase Metrics Comparison")
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        plotly_ok = True
    except ImportError:
        plotly_ok = False
        st.warning("💡 Install plotly for interactive charts: `pip install plotly`")
    
    if plotly_ok and not df_all.empty:
        radar_df = generate_radar_chart_data(df_all, area_per_pixel)
        
        if radar_df is not None and len(radar_df) >= 2:
            # Normalize metrics for radar chart (0-1 scale)
            metrics_to_plot = ['Avg Area (µm²)', 'Avg ECD (µm)', 'Avg Circularity', 'Avg Aspect Ratio', 'Grain Count']
            
            fig_radar = go.Figure()
            
            colors = {'HCP ε': '#e74c3c', 'FCC γ': '#2ecc71'}
            
            for idx, row in radar_df.iterrows():
                phase_name = row['Phase']
                color = colors.get(phase_name, '#95a5a6')
                
                # Normalize values for radar display
                values = []
                for metric in metrics_to_plot:
                    val = row[metric]
                    # Simple normalization (adjust ranges as needed for your data)
                    if metric == 'Grain Count':
                        norm_val = min(val / 100, 1.0)
                    elif metric == 'Avg Circularity':
                        norm_val = val  # Already 0-1
                    elif metric == 'Avg Aspect Ratio':
                        norm_val = min(val / 5, 1.0)
                    else:
                        norm_val = min(val / 100, 1.0)  # Generic normalization
                    values.append(norm_val)
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics_to_plot,
                    fill='toself',
                    name=phase_name,
                    line_color=color,
                    fillcolor=color.replace(')', ', 0.2)').replace('rgb', 'rgba') if 'rgb' in color else color + '33'
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                ),
                showlegend=True,
                title="Phase Metrics Comparison (Normalized)",
                height=400
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("Need at least 2 phases with data for radar chart")
    
    # Chord Diagram for Phase Relationships
    st.markdown("### 🔗 Chord Diagram: Phase Relationships")
    
    if plotly_ok:
        chord_data = generate_chord_data(df_all, red_fraction, green_fraction, boundary_fraction)
        
        if chord_data and chord_data['links']:
            # Create interactive chord diagram using plotly
            nodes = chord_data['nodes']
            links = chord_data['links']
            
            # Create node positions in a circle
            n = len(nodes)
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            
            fig_chord = go.Figure()
            
            # Add nodes as points
            for i, node in enumerate(nodes):
                x, y = np.cos(angles[i]), np.sin(angles[i])
                fig_chord.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers+text',
                    marker=dict(size=20, color=node['color'], line=dict(width=2, color='white')),
                    text=[node['name']],
                    textposition='top center',
                    name=node['name'],
                    hoverinfo='name',
                    showlegend=False
                ))
            
            # Add links as curved lines
            for link in links:
                source_idx = link['source']
                target_idx = link['target']
                
                if source_idx == target_idx:
                    # Self-loop: small circle around node
                    x0, y0 = np.cos(angles[source_idx]), np.sin(angles[source_idx])
                    theta = np.linspace(0, 2*np.pi, 50)
                    r = 0.15 * link['value'] / 100
                    x = x0 + r * np.cos(theta)
                    y = y0 + r * np.sin(theta)
                else:
                    # Connection between nodes
                    x0, y0 = np.cos(angles[source_idx]), np.sin(angles[source_idx])
                    x1, y1 = np.cos(angles[target_idx]), np.sin(angles[target_idx])
                    
                    # Create curved path
                    mid_angle = (angles[source_idx] + angles[target_idx]) / 2
                    mid_dist = 0.5 + link['value'] / 200
                    xm, ym = mid_dist * np.cos(mid_angle), mid_dist * np.sin(mid_angle)
                    
                    # Quadratic bezier curve
                    t = np.linspace(0, 1, 50)
                    x = (1-t)**2 * x0 + 2*(1-t)*t * xm + t**2 * x1
                    y = (1-t)**2 * y0 + 2*(1-t)*t * ym + t**2 * y1
                
                fig_chord.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='lines',
                    line=dict(width=link['value']/20 + 1, color='rgba(100,100,100,0.5)'),
                    hoverinfo='skip',
                    showlegend=False,
                    name=link.get('label', '')
                ))
            
            fig_chord.update_layout(
                title="Phase Relationship Network",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
                width=600,
                height=500,
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig_chord, use_container_width=True)
        else:
            st.info("Add phase data to generate chord diagram")
    
    # --- Data Export Section ---
    
    st.markdown("---")
    st.subheader("📥 Export Results")
    
    # Create comprehensive CSV
    filename_base = Path(uploaded_file.name if uploaded_file else selected_filename).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # CSV Download - Grain morphology data
    csv_morphology = df_all.to_csv(index=False) if not df_all.empty else ""
    st.download_button(
        label="📊 Download Grain Morphology Data (CSV)",
        data=csv_morphology.encode('utf-8'),
        file_name=f"grain_morphology_{filename_base}_{timestamp}.csv",
        mime="text/csv",
        help="Download detailed metrics for each detected grain"
    )
    
    # CSV Download - Comprehensive results
    comprehensive_df = create_comprehensive_csv(
        df_all, red_fraction, green_fraction, boundary_fraction,
        red_area_abs, green_area_abs, boundary_area_abs,
        image_info, 
        uploaded_file.name if uploaded_file else selected_filename,
        domain_size_um, area_per_pixel
    )
    csv_comprehensive = comprehensive_df.to_csv(index=False)
    st.download_button(
        label="📋 Download Comprehensive Analysis Report (CSV)",
        data=csv_comprehensive.encode('utf-8'),
        file_name=f"analysis_report_{filename_base}_{timestamp}.csv",
        mime="text/csv",
        help="Download complete analysis including metadata, fractions, and grain data"
    )
    
    # JSON Download - For programmatic use
    if not df_all.empty:
        json_data = {
            "metadata": {
                "filename": uploaded_file.name if uploaded_file else selected_filename,
                "timestamp": datetime.now().isoformat(),
                "domain_size_um": domain_size_um,
                "area_per_pixel": area_per_pixel
            },
            "experimental": image_info if image_info and image_info.get('valid') else None,
            "phase_fractions": {
                "HCP_epsilon_percent": red_fraction,
                "FCC_gamma_percent": green_fraction,
                "boundary_percent": boundary_fraction
            },
            "absolute_areas_um2": {
                "HCP_epsilon": red_area_abs,
                "FCC_gamma": green_area_abs,
                "boundary": boundary_area_abs
            },
            "morphology_summary": {
                phase: {
                    "grain_count": len(df_all[df_all['Phase'] == phase]),
                    "mean_area": df_all[df_all['Phase'] == phase]['Area (µm²)'].mean(),
                    "mean_ecd": df_all[df_all['Phase'] == phase]['ECD (µm)'].mean(),
                    "mean_circularity": df_all[df_all['Phase'] == phase]['Circularity'].mean()
                }
                for phase in df_all['Phase'].unique()
            },
            "grains": df_all.to_dict('records')
        }
        
        json_str = json.dumps(json_data, indent=2)
        st.download_button(
            label="🔧 Download Analysis Data (JSON)",
            data=json_str.encode('utf-8'),
            file_name=f"analysis_data_{filename_base}_{timestamp}.json",
            mime="application/json",
            help="Download structured JSON for programmatic analysis"
        )
    
    # Summary display
    with st.expander("📄 View Analysis Summary"):
        st.write(f"""
        **Analysis Summary for:** `{uploaded_file.name if uploaded_file else selected_filename}`
        
        | Metric | HCP ε | FCC γ | Boundaries |
        |--------|-------|-------|------------|
        | **Area Fraction** | {red_fraction:.2f}% | {green_fraction:.2f}% | {boundary_fraction:.2f}% |
        | **Absolute Area** | {red_area_abs:.1f} µm² | {green_area_abs:.1f} µm² | {boundary_area_abs:.1f} µm² |
        | **Pixel Count** | {red_pixels:,} | {green_pixels:,} | {boundary_pixels:,} |
        | **Grain Count** | {len(df_red)} | {len(df_green)} | - |
        
        **Experimental Conditions:**
        - Laser: {image_info['laser_description'] if image_info and image_info.get('valid') else 'Unknown'}
        - Heating: {image_info['heating_description'] if image_info and image_info.get('valid') else 'Unknown'}
        - Time: {image_info['time'] if image_info and image_info.get('valid') else 'Unknown'}
        - Orientation: {image_info['orientation_description'] if image_info and image_info.get('valid') else 'Unknown'}
        """)

else:
    # Welcome/instructions state
    st.info("👈 Select an image from the sidebar to begin analysis")
    
    with st.expander("📖 How to Use"):
        st.write("""
        ### 🏷️ Filename Convention
        Images should follow: **ABN.ext**
        
        | Position | Values | Meaning |
        |----------|--------|---------|
        | **A** | C, P | Laser: Continuous Wave or Pulse |
        | **B** | NH, H | Heating: Not Heated (t=0) or Heated (t=35 min) |
        | **N** | 0, 45 | Orientation: 0° or 45° |
        
        **Examples:** `CNH0.bmp`, `PH45.png`, `CH0.jpg`
        
        ### 🔄 Analysis Workflow
        1. Select image source (folder or upload)
        2. Set physical calibration (domain size in µm)
        3. Configure analysis options
        4. Review phase fractions and morphology results
        5. Download data in CSV or JSON format
        
        ### 📊 Advanced Visualizations
        - **Radar Chart**: Compare morphological metrics between phases
        - **Chord Diagram**: Visualize phase relationships and interfaces
        - **Histograms**: Grain size and shape distributions
        
        ### ⚠️ BMP File Issues on Streamlit Cloud
        If BMP files fail to load:
        - Convert to PNG locally: `Image.open('file.bmp').save('file.png')`
        - Use the upload option as workaround
        - Check sidebar debug output for specific errors
        """)
    
    with st.expander("📦 Requirements"):
        st.code("""
# requirements.txt
streamlit>=1.28.0
opencv-python-headless>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-image>=0.21.0
plotly>=5.17.0
        """)
    
    with st.expander("🔧 Advanced: HSV Threshold Tuning"):
        st.write("""
        If automatic segmentation isn't accurate, adjust these values in code:
        
        **HCP ε (Red phase) - HSV ranges:**
        ```python
        lower_red1 = [0, 70, 50]    # Hue: 0-15 (red wraps around)
        upper_red1 = [15, 255, 255] # Sat: 70-255, Val: 50-255
        lower_red2 = [160, 70, 50]  # Second red range
        upper_red2 = [180, 255, 255]
        ```
        
        **FCC γ (Green phase) - HSV ranges:**
        ```python
        lower_green = [40, 70, 50]  # Hue: 40-80 for green
        upper_green = [80, 255, 255]
        ```
        
        **Tips:**
        - Lower saturation threshold to detect faded colors
        - Adjust value threshold for brightness variations
        - Use morphology to clean noisy masks
        """)

# Footer
st.markdown("---")
st.caption("🔬 Microstructure Phase Analyzer | HCP ε / FCC γ Quantification | Built with Streamlit + OpenCV + scikit-image + Plotly")
