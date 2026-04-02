import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from skimage.measure import regionprops, label
import io
import os
import re
from pathlib import Path
import traceback

# Enable PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Page Configuration
st.set_page_config(page_title="Microstructure Analyzer", layout="wide")

st.title("🔬 Microstructure Phase Analyzer (HCP ε / FCC γ)")
st.markdown("""
This tool analyzes two-phase microstructure images with automatic experimental condition extraction.
It calculates **phase fractions**, **area fractions**, and morphological metrics for **HCP epsilon (ε)** and **FCC gamma (γ)** phases.
""")

# --- Helper Functions ---

def parse_filename(filename):
    """
    Parse filename according to convention: ABN.ext
    A: C (Continuous wave) or P (Pulse laser)
    B: NH (Not heated, t=0) or H (Heated, t=35 min)
    N: 0 (0 degree) or 45 (45 degree orientation)
    """
    # Remove extension
    name_without_ext = Path(filename).stem
    
    # Pattern: Letter + (NH or H) + Number
    pattern = r'^([CP])(NH|H)(\d+)$'
    match = re.match(pattern, name_without_ext, re.IGNORECASE)
    
    if match:
        laser_type = match.group(1).upper()
        heating = match.group(2).upper()
        orientation = match.group(3)
        
        # Decode laser type
        if laser_type == 'C':
            laser_desc = "Continuous Wave Laser"
        elif laser_type == 'P':
            laser_desc = "Pulse Type Laser"
        else:
            laser_desc = "Unknown"
        
        # Decode heating condition
        if heating == 'NH':
            heating_desc = "Not Heated (Pre-heated)"
            time_desc = "t = 0 min"
        elif heating == 'H':
            heating_desc = "Heated Isothermally"
            time_desc = "t = 35 min"
        else:
            heating_desc = "Unknown"
            time_desc = "Unknown"
        
        # Decode orientation
        if orientation == '0':
            orientation_desc = "0° (Reference)"
        elif orientation == '45':
            orientation_desc = "45° (Rotated)"
        else:
            orientation_desc = f"{orientation}°"
        
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
        # Get the directory of the current file
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        return script_dir
    except Exception as e:
        # Fallback to current working directory
        st.warning(f"Could not determine script directory: {e}")
        return os.getcwd()

def scan_images_folder(folder_path="./images"):
    """Scan the images folder and return list of valid image files"""
    supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    images = []
    
    # Get the absolute path
    abs_folder_path = os.path.abspath(folder_path)
    
    st.sidebar.write(f"**Searching in:** `{abs_folder_path}`")
    st.sidebar.write(f"**Folder exists:** {os.path.exists(abs_folder_path)}")
    
    if os.path.exists(abs_folder_path):
        try:
            all_files = os.listdir(abs_folder_path)
            st.sidebar.write(f"**All files in folder:** {len(all_files)} items")
            
            # Show first few files for debugging
            if len(all_files) > 0:
                st.sidebar.write("**Sample files:**")
                for f in all_files[:5]:  # Show first 5 files
                    st.sidebar.text(f"  - {f}")
                if len(all_files) > 5:
                    st.sidebar.text(f"  ... and {len(all_files) - 5} more")
            
            for filename in all_files:
                file_path = os.path.join(abs_folder_path, filename)
                # Only process files (not directories)
                if os.path.isfile(file_path):
                    ext = Path(filename).suffix.lower()
                    if ext in supported_extensions:
                        images.append(filename)
                        st.sidebar.success(f"✓ Found image: {filename}")
            
            if len(images) == 0:
                st.sidebar.warning("No image files found with supported extensions")
                st.sidebar.write(f"**Supported extensions:** {', '.join(supported_extensions)}")
        except Exception as e:
            st.sidebar.error(f"Error reading folder: {e}")
            st.sidebar.code(traceback.format_exc())
    else:
        st.sidebar.error(f"Folder does not exist: {abs_folder_path}")
        # Try to create it
        try:
            os.makedirs(abs_folder_path, exist_ok=True)
            st.sidebar.info(f"Created folder: {abs_folder_path}")
            st.sidebar.warning("Please add image files to this folder and refresh the page")
        except Exception as e:
            st.sidebar.error(f"Could not create folder: {e}")
    
    return sorted(images)

def load_image_robust(image_path):
    """
    Robust image loading with multiple fallback methods
    Handles BMP, JPG, PNG, TIFF with conversion if needed
    """
    st.info(f"🔄 Loading image: {image_path}")
    
    # Method 1: Direct PIL loading
    try:
        st.write("  → Trying direct PIL load...")
        img = Image.open(image_path)
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        st.success("  ✓ Direct PIL load successful")
        return img
    except Exception as e1:
        st.warning(f"  ✗ Direct PIL load failed: {e1}")
    
    # Method 2: Load with OpenCV then convert to PIL
    try:
        st.write("  → Trying OpenCV load...")
        img_cv = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_cv is not None:
            # OpenCV uses BGR, convert to RGB
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            st.success("  ✓ OpenCV load successful")
            return img
        else:
            st.warning("  ✗ OpenCV returned None")
    except Exception as e2:
        st.warning(f"  ✗ OpenCV load failed: {e2}")
    
    # Method 3: Read file bytes and load with PIL from memory
    try:
        st.write("  → Trying memory buffer load...")
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        buffer = io.BytesIO(img_bytes)
        img = Image.open(buffer)
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        st.success("  ✓ Memory buffer load successful")
        return img
    except Exception as e3:
        st.warning(f"  ✗ Memory buffer load failed: {e3}")
    
    # Method 4: Convert BMP to JPG using OpenCV if it's a BMP file
    file_ext = Path(image_path).suffix.lower()
    if file_ext == '.bmp':
        try:
            st.write("  → Trying BMP to JPG conversion via OpenCV...")
            img_cv = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img_cv is not None:
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                # Save to memory as JPG
                jpg_buffer = io.BytesIO()
                img_pil = Image.fromarray(img_rgb)
                img_pil.save(jpg_buffer, format='JPEG', quality=95)
                jpg_buffer.seek(0)
                # Reload from JPG buffer
                img_converted = Image.open(jpg_buffer)
                if img_converted.mode != 'RGB':
                    img_converted = img_converted.convert('RGB')
                st.success("  ✓ BMP to JPG conversion successful")
                return img_converted
        except Exception as e4:
            st.warning(f"  ✗ BMP to JPG conversion failed: {e4}")
    
    # All methods failed
    st.error(f"❌ Could not load image: {image_path}")
    st.error("Full traceback:")
    st.code(traceback.format_exc())
    return None

def convert_image_format(img, output_format='JPEG', quality=95):
    """Convert PIL image to specified format in memory"""
    try:
        buffer = io.BytesIO()
        if output_format.upper() == 'JPEG' and img.mode in ('RGBA', 'LA', 'P'):
            # Create white background for JPEG (doesn't support transparency)
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            background.save(buffer, format=output_format, quality=quality)
        else:
            if img.mode in ('RGBA', 'LA') and output_format.upper() == 'JPEG':
                img = img.convert('RGB')
            img.save(buffer, format=output_format, quality=quality)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error converting image format: {e}")
        return None

# --- Sidebar Controls ---
st.sidebar.header("📁 Image Source")

# Show current directory information
st.sidebar.subheader("📍 Directory Information")
script_dir = get_script_directory()
cwd = os.getcwd()
st.sidebar.write(f"**Script location:** `{script_dir}`")
st.sidebar.write(f"**Current working dir:** `{cwd}`")
st.sidebar.write(f"**Same directory:** {script_dir == cwd}")

# Option to choose between folder images or upload
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
    # Try multiple possible paths for the images folder
    st.sidebar.subheader("🔍 Searching for Images")
    
    # Possible paths to check
    possible_paths = [
        "./images",
        "images",
        os.path.join(script_dir, "images"),
        os.path.join(cwd, "images"),
    ]
    
    available_images = []
    images_path_used = None
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            st.sidebar.success(f"Found images folder at: `{path}`")
            images_path_used = path
            available_images = scan_images_folder(path)
            images_folder_path = abs_path
            break
        else:
            st.sidebar.text(f"❌ Not found: `{path}`")
    
    if not available_images:
        # If still no images found, show detailed debugging
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 Troubleshooting")
        st.sidebar.write("**Checked paths:**")
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            exists = os.path.exists(abs_path)
            status = "✓" if exists else "✗"
            st.sidebar.text(f"  {status} {abs_path}")
        
        # List what's in the script directory
        st.sidebar.write("**Contents of script directory:**")
        try:
            script_contents = os.listdir(script_dir)
            for item in script_contents[:10]:  # Show first 10 items
                item_path = os.path.join(script_dir, item)
                is_dir = os.path.isdir(item_path)
                st.sidebar.text(f"  {'📁' if is_dir else '📄'} {item}")
            if len(script_contents) > 10:
                st.sidebar.text(f"  ... and {len(script_contents) - 10} more items")
        except Exception as e:
            st.sidebar.error(f"Cannot list directory: {e}")
        
        # Offer to create the folder
        st.sidebar.write("**Create images folder?**")
        if st.sidebar.button("Create ./images folder"):
            try:
                new_folder_path = os.path.join(script_dir, "images")
                os.makedirs(new_folder_path, exist_ok=True)
                st.sidebar.success(f"Created: {new_folder_path}")
                st.sidebar.info("Please add your image files to this folder and refresh the page")
                st.experimental_rerun()
            except Exception as e:
                st.sidebar.error(f"Failed to create folder: {e}")
    
    if available_images:
        st.sidebar.markdown("---")
        st.sidebar.success(f"✅ Found {len(available_images)} image(s)")
        
        selected_filename = st.sidebar.selectbox(
            "Select an image:",
            available_images
        )
        
        if selected_filename:
            # Parse the filename
            image_info = parse_filename(selected_filename)
            
            # Load the image
            if images_folder_path:
                image_path = os.path.join(images_folder_path, selected_filename)
            else:
                image_path = os.path.join(images_path_used, selected_filename)
            
            st.sidebar.write(f"**Loading:** `{image_path}`")
            
            # Use robust loading function
            selected_folder_image = load_image_robust(image_path)
            
            if selected_folder_image is not None:
                # Create a mock uploaded_file object for compatibility
                class MockUploadedFile:
                    def __init__(self, name):
                        self.name = name
                    def getvalue(self):
                        return None
                
                uploaded_file = MockUploadedFile(selected_filename)
                st.sidebar.success(f"✅ Loaded: {selected_filename}")
                st.sidebar.write(f"**Image mode:** {selected_folder_image.mode}")
                st.sidebar.write(f"**Image size:** {selected_folder_image.size}")
            else:
                st.sidebar.error(f"❌ Failed to load: {selected_filename}")
                st.sidebar.info("Try uploading the file manually using the upload option")
else:
    # File upload option
    uploaded_file = st.sidebar.file_uploader(
        "Upload Microstructure Image", 
        type=["png", "jpg", "jpeg", "bmp", "tiff"]
    )
    
    if uploaded_file:
        # Parse uploaded filename
        image_info = parse_filename(uploaded_file.name)
        
        # Load uploaded image
        try:
            selected_folder_image = Image.open(uploaded_file)
            if selected_folder_image.mode in ('RGBA', 'LA', 'P'):
                selected_folder_image = selected_folder_image.convert('RGB')
            st.sidebar.success(f"✅ Uploaded: {uploaded_file.name}")
        except Exception as e:
            st.sidebar.error(f"Error loading uploaded file: {e}")
            # Try OpenCV fallback for uploaded files
            try:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img_cv is not None:
                    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                    selected_folder_image = Image.fromarray(img_rgb)
                    st.sidebar.success("✅ Loaded via OpenCV fallback")
            except Exception as e2:
                st.sidebar.error(f"OpenCV fallback also failed: {e2}")

# --- Display Experimental Conditions ---
if image_info and image_info.get('valid'):
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 Experimental Conditions")
    
    col_exp1, col_exp2 = st.sidebar.columns(2)
    
    with col_exp1:
        st.metric("Laser Type", image_info['laser_type'], image_info['laser_description'])
        st.metric("Heating", image_info['heating_code'], image_info['heating_description'])
    
    with col_exp2:
        st.metric("Orientation", f"{image_info['orientation']}°", image_info['orientation_description'])
        st.metric("Time", image_info['time'])
    
    # Show full details in expander
    with st.sidebar.expander("📋 View Full Experimental Details"):
        filename_display = uploaded_file.name if uploaded_file else (selected_filename if selected_filename else 'N/A')
        st.write(f"""
        - **Laser Type:** {image_info['laser_description']} ({image_info['laser_type']})
        - **Heating Condition:** {image_info['heating_description']} ({image_info['heating_code']})
        - **Isothermal Time:** {image_info['time']}
        - **Sample Orientation:** {image_info['orientation_description']} ({image_info['orientation']}°)
        - **Filename:** {filename_display}
        """)

elif image_info and not image_info.get('valid'):
    st.sidebar.warning(f"⚠️ {image_info.get('message', 'Could not parse filename')}")

# --- Main Configuration ---
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Analysis Configuration")

# Domain Calibration
st.sidebar.subheader("Physical Calibration")
domain_size_um = st.sidebar.number_input(
    "Total Domain Length (µm)", 
    value=250.0, 
    help="The physical length of the square image side."
)
st.sidebar.info(f"Assuming a square domain of {domain_size_um} x {domain_size_um} µm")

# Analysis Options
st.sidebar.subheader("Analysis Options")
exclude_boundaries = st.sidebar.checkbox(
    "Exclude Grain Boundaries from Area Fraction", 
    value=False,
    help="When checked, area fractions are normalized so that HCP + FCC = 100% (boundaries excluded)"
)

# Segmentation Settings
st.sidebar.subheader("Segmentation Settings")
use_auto_threshold = st.sidebar.checkbox("Use Automatic Color Detection", value=True)
if not use_auto_threshold:
    st.sidebar.warning("Manual HSV ranges not implemented for this demo. Using fixed robust ranges.")

# --- Main Processing Logic ---

if uploaded_file is not None and selected_folder_image is not None:
    # Get the numpy array from PIL image
    img_np = np.array(selected_folder_image)
    
    # Remove Alpha channel if present
    if img_np.ndim == 3 and img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]
        st.info("Removed alpha channel from image")
    
    # Ensure we have a 3-channel image
    if img_np.ndim == 2:
        # Grayscale to RGB
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        st.info("Converted grayscale to RGB")
    
    h, w, _ = img_np.shape
    
    # Calculate Pixel Resolution
    total_physical_area = domain_size_um ** 2
    total_pixels = h * w
    area_per_pixel = total_physical_area / total_pixels
    
    # Display image info at top
    col_img1, col_img2, col_img3 = st.columns(3)
    with col_img1:
        st.metric("Image Resolution", f"{w} x {h}")
    with col_img2:
        st.metric("Calibration", f"{area_per_pixel:.4f} µm²/px")
    with col_img3:
        st.metric("Total Area", f"{total_physical_area:.0f} µm²")
    
    # Display the original image
    with st.expander("🖼️ View Original Image"):
        st.image(img_np, caption=f"Original: {uploaded_file.name if uploaded_file else selected_filename}", use_column_width=True)
    
    # Convert to HSV for color segmentation
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    # Define Range for Red (HCP epsilon phase)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask1, mask2)

    # Define Range for Green (FCC gamma phase)
    lower_green = np.array([40, 70, 50])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Optional: Morphological operations to clean masks
    st.sidebar.subheader("Mask Refinement")
    apply_morphology = st.sidebar.checkbox("Apply morphological cleaning", value=True)
    
    if apply_morphology:
        kernel_size = st.sidebar.slider("Kernel size for morphology", 1, 15, 3, step=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Clean red mask (HCP)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
        
        # Clean green mask (FCC)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
        
        st.sidebar.info(f"Applied opening/closing with {kernel_size}x{kernel_size} kernel")

    # --- Calculations ---
    
    # Area Calculations
    red_pixels = cv2.countNonZero(mask_red)
    green_pixels = cv2.countNonZero(mask_green)
    boundary_pixels = total_pixels - red_pixels - green_pixels
    
    red_area_abs = red_pixels * area_per_pixel
    green_area_abs = green_pixels * area_per_pixel
    boundary_area_abs = boundary_pixels * area_per_pixel
    
    # Area Fraction Calculations
    if exclude_boundaries:
        total_phase_pixels = red_pixels + green_pixels
        
        if total_phase_pixels > 0:
            red_fraction = (red_pixels / total_phase_pixels) * 100
            green_fraction = (green_pixels / total_phase_pixels) * 100
        else:
            red_fraction = 0
            green_fraction = 0
            
        boundary_fraction = 0
        
        st.info("📊 **Mode:** Area fractions normalized to exclude grain boundaries (HCP ε + FCC γ = 100%)")
    else:
        red_fraction = (red_pixels / total_pixels) * 100
        green_fraction = (green_pixels / total_pixels) * 100
        boundary_fraction = (boundary_pixels / total_pixels) * 100
        
        st.info("📊 **Mode:** Area fractions include grain boundaries in total area")

    # Morphology Analysis
    def analyze_morphology(mask, phase_name, area_per_pixel):
        labeled_mask = label(mask)
        props = regionprops(labeled_mask)
        
        if not props:
            return pd.DataFrame()
            
        data = []
        for prop in props:
            if prop.area < 5:
                continue
                
            area_um2 = prop.area * area_per_pixel
            ecd_um = np.sqrt(4 * area_um2 / np.pi)
            
            perimeter = prop.perimeter if prop.perimeter > 0 else 0.001
            circularity = (4 * np.pi * prop.area) / (perimeter ** 2)
            
            if prop.major_axis_length > 0 and prop.minor_axis_length > 0:
                aspect_ratio = prop.major_axis_length / prop.minor_axis_length
            else:
                aspect_ratio = 1.0
                
            data.append({
                "Phase": phase_name,
                "Grain ID": prop.label,
                "Area (µm²)": area_um2,
                "ECD (µm)": ecd_um,
                "Circularity": circularity,
                "Aspect Ratio": aspect_ratio
            })
            
        return pd.DataFrame(data)

    df_red_morph = analyze_morphology(mask_red, "HCP ε (Red)", area_per_pixel)
    df_green_morph = analyze_morphology(mask_green, "FCC γ (Green)", area_per_pixel)
    df_all_morph = pd.concat([df_red_morph, df_green_morph], ignore_index=True)

    # --- Display Results ---

    st.markdown("---")
    st.subheader("📊 Phase Quantification Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Absolute Areas")
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("HCP ε Area", f"{red_area_abs:.1f} µm²", f"{red_pixels} pixels")
        kpi2.metric("FCC γ Area", f"{green_area_abs:.1f} µm²", f"{green_pixels} pixels")
        kpi3.metric("Boundary Area", f"{boundary_area_abs:.1f} µm²", f"{boundary_pixels} pixels")
        
        st.divider()
        
        st.markdown("### Area Fractions")
        if exclude_boundaries:
            kpi4, kpi5 = st.columns(2)
            kpi4.metric("HCP ε Fraction", f"{red_fraction:.2f}%", "Normalized")
            kpi5.metric("FCC γ Fraction", f"{green_fraction:.2f}%", "Normalized")
            st.caption("✅ HCP ε + FCC γ = 100% (boundaries excluded)")
        else:
            kpi4, kpi5, kpi6 = st.columns(3)
            kpi4.metric("HCP ε Fraction", f"{red_fraction:.1f}%")
            kpi5.metric("FCC γ Fraction", f"{green_fraction:.1f}%")
            kpi6.metric("Boundaries", f"{boundary_fraction:.1f}%")
            st.caption("📏 Fractions include grain boundaries")
        
        # Chart
        st.markdown("### Phase Distribution")
        if exclude_boundaries:
            chart_data = pd.DataFrame({
                "Phase": ["HCP ε", "FCC γ"],
                "Area Fraction (%)": [red_fraction, green_fraction]
            })
        else:
            chart_data = pd.DataFrame({
                "Phase": ["HCP ε", "FCC γ", "Boundaries"],
                "Area Fraction (%)": [red_fraction, green_fraction, boundary_fraction]
            })
        st.bar_chart(chart_data.set_index("Phase"))

    with col2:
        st.markdown("### Segmentation Preview")
        c1, c2 = st.columns(2)
        with c1:
            st.image(mask_red, caption="HCP ε Phase (Red)", use_column_width=True)
        with c2:
            st.image(mask_green, caption="FCC γ Phase (Green)", use_column_width=True)
        
        # Overlay visualization
        with st.expander("🔍 View Segmentation Overlay"):
            overlay = img_np.copy()
            # Add red tint for HCP
            overlay[mask_red > 0] = overlay[mask_red > 0] * 0.7 + np.array([255, 100, 100]) * 0.3
            # Add green tint for FCC
            overlay[mask_green > 0] = overlay[mask_green > 0] * 0.7 + np.array([100, 255, 100]) * 0.3
            st.image(overlay, caption="Overlay: Red=HCP ε, Green=FCC γ", use_column_width=True)

    st.markdown("---")
    st.subheader("📐 Morphological Metrics")
    
    if not df_all_morph.empty:
        # Summary Statistics
        st.markdown("### Summary Statistics per Phase")
        summary = df_all_morph.groupby("Phase")[["Area (µm²)", "ECD (µm)", "Circularity", "Aspect Ratio"]].mean()
        summary['Grain Count'] = df_all_morph.groupby("Phase").size()
        summary = summary[['Grain Count', 'Area (µm²)', 'ECD (µm)', 'Circularity', 'Aspect Ratio']]
        st.dataframe(summary.style.format("{:.2f}"))
        
        # Download button
        filename_base = uploaded_file.name if uploaded_file else (selected_filename if selected_filename else 'analysis')
        csv_filename = f'grain_morphology_{Path(filename_base).stem}.csv'
        st.download_button(
            label="📥 Download Detailed Grain Data (CSV)",
            data=df_all_morph.to_csv(index=False).encode('utf-8'),
            file_name=csv_filename,
            mime='text/csv',
        )
        
        # Histograms
        st.markdown("### Grain Size Distribution (ECD)")
        
        try:
            import plotly.express as px
            plotly_available = True
        except ImportError:
            plotly_available = False
            st.warning("Install plotly for interactive histograms: `pip install plotly`")
        
        col_h1, col_h2 = st.columns(2)
        
        with col_h1:
            if not df_red_morph.empty:
                st.markdown("**HCP ε Phase**")
                if plotly_available:
                    fig_red = px.histogram(df_red_morph, x="ECD (µm)", nbins=20, 
                                          title="HCP ε Grain Size Distribution",
                                          labels={"ECD (µm)": "Equivalent Diameter (µm)"},
                                          color_discrete_sequence=['#e74c3c'])
                    fig_red.update_layout(bargap=0.1)
                    st.plotly_chart(fig_red, use_container_width=True)
                else:
                    st.write(f"- Count: {len(df_red_morph)} grains")
                    st.write(f"- Mean ECD: {df_red_morph['ECD (µm)'].mean():.2f} µm")
                    st.write(f"- Std Dev: {df_red_morph['ECD (µm)'].std():.2f} µm")
                    st.write(f"- Min: {df_red_morph['ECD (µm)'].min():.2f} µm")
                    st.write(f"- Max: {df_red_morph['ECD (µm)'].max():.2f} µm")
        
        with col_h2:
            if not df_green_morph.empty:
                st.markdown("**FCC γ Phase**")
                if plotly_available:
                    fig_green = px.histogram(df_green_morph, x="ECD (µm)", nbins=20,
                                            title="FCC γ Grain Size Distribution",
                                            labels={"ECD (µm)": "Equivalent Diameter (µm)"},
                                            color_discrete_sequence=['#2ecc71'])
                    fig_green.update_layout(bargap=0.1)
                    st.plotly_chart(fig_green, use_container_width=True)
                else:
                    st.write(f"- Count: {len(df_green_morph)} grains")
                    st.write(f"- Mean ECD: {df_green_morph['ECD (µm)'].mean():.2f} µm")
                    st.write(f"- Std Dev: {df_green_morph['ECD (µm)'].std():.2f} µm")
                    st.write(f"- Min: {df_green_morph['ECD (µm)'].min():.2f} µm")
                    st.write(f"- Max: {df_green_morph['ECD (µm)'].max():.2f} µm")
        
        st.markdown("### Circularity Distribution")
        if not df_all_morph.empty:
            if plotly_available:
                fig_circ = px.histogram(df_all_morph, x="Circularity", color="Phase", 
                                       nbins=20, title="Circularity by Phase (1.0 = Perfect Circle)",
                                       labels={"Circularity": "Circularity", "Phase": "Phase"},
                                       color_discrete_map={"HCP ε (Red)": "#e74c3c", "FCC γ (Green)": "#2ecc71"})
                fig_circ.update_layout(bargap=0.1)
                st.plotly_chart(fig_circ, use_container_width=True)
            else:
                st.write("**HCP Circularity:**")
                if not df_red_morph.empty:
                    st.write(f"- Mean: {df_red_morph['Circularity'].mean():.3f}")
                    st.write(f"- Std: {df_red_morph['Circularity'].std():.3f}")
                st.write("**FCC Circularity:**")
                if not df_green_morph.empty:
                    st.write(f"- Mean: {df_green_morph['Circularity'].mean():.3f}")
                    st.write(f"- Std: {df_green_morph['Circularity'].std():.3f}")

    else:
        st.warning("⚠️ No grains detected. Try adjusting the image or segmentation settings.")
        st.info("Tips: Check if your image has clear red/green phases, adjust HSV thresholds, or enable morphological cleaning")

else:
    st.info("👈 Please select an image from the sidebar to begin analysis.")
    
    with st.expander("📖 How to Use"):
        st.write("""
        ### Image Naming Convention
        Images should follow the pattern: **ABN.ext** where:
        - **A** = Laser Type: **C** (Continuous Wave) or **P** (Pulse)
        - **B** = Heating: **NH** (Not Heated, t=0) or **H** (Heated, t=35 min)
        - **N** = Orientation: **0** (0°) or **45** (45°)
        
        **Examples:**
        - `CNH0.bmp` - Continuous wave, not heated, 0° orientation
        - `PH45.png` - Pulse laser, heated 35 min, 45° orientation
        - `CH0.jpg` - Continuous wave, heated, 0° orientation
        
        ### Analysis Steps
        1. Select image source (folder or upload)
        2. Configure physical calibration (domain size)
        3. Choose analysis options (boundary exclusion, morphology)
        4. View phase fractions and morphological data
        
        ### Troubleshooting BMP Loading Issues
        If you see "cannot identify image file" errors:
        - The BMP file might be corrupted or use an uncommon format
        - Try converting the BMP to PNG/JPG using any image editor
        - The app will automatically try multiple loading methods
        - Check the sidebar debug output for specific error messages
        
        ### Supported Formats
        - ✅ PNG, JPG, JPEG (best compatibility)
        - ✅ BMP (with automatic conversion fallbacks)
        - ✅ TIFF, TIF
        """)
    
    with st.expander("📦 Required Packages"):
        st.write("""
        ```bash
        pip install streamlit opencv-python-headless numpy pandas scikit-image pillow plotly
        ```
        
        For BMP support, ensure Pillow is up to date:
        ```bash
        pip install --upgrade Pillow
        ```
        """)
    
    with st.expander("🔧 Advanced: Manual HSV Threshold Adjustment"):
        st.write("""
        If automatic color detection isn't working well:
        
        **For HCP (Red phase):**
        - Hue: 0-15 or 160-180 (red appears at both ends of HSV hue)
        - Saturation: 70-255 (avoid gray/white)
        - Value: 50-255 (avoid very dark pixels)
        
        **For FCC (Green phase):**
        - Hue: 40-80 (green range)
        - Saturation: 70-255
        - Value: 50-255
        
        You can modify the `lower_red1`, `upper_red1`, etc. arrays in the code
        to fine-tune detection for your specific images.
        """)
