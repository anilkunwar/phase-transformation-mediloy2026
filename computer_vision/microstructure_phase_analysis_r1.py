import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import regionprops, label
import io

# Page Configuration
st.set_page_config(page_title="Microstructure Analyzer", layout="wide")

st.title("🔬 Microstructure Phase Analyzer (HCP/FCC)")
st.markdown("""
This tool analyzes two-phase microstructure images. 
It calculates absolute areas, area fractions, and morphological metrics based on a defined domain size.
""")

# --- Sidebar Controls ---
st.sidebar.header("Configuration")

# 1. Image Upload
uploaded_file = st.sidebar.file_uploader("Upload Microstructure Image", type=["png", "jpg", "jpeg", "bmp", "tiff"])

# 2. Domain Calibration
st.sidebar.subheader("Physical Calibration")
domain_size_um = st.sidebar.number_input("Total Domain Length (µm)", value=250.0, help="The physical length of the square image side.")
st.sidebar.info(f"Assuming a square domain of {domain_size_um} x {domain_size_um} µm")

# 3. Area Fraction Calculation Mode
st.sidebar.subheader("Analysis Options")
exclude_boundaries = st.sidebar.checkbox(
    "Exclude Grain Boundaries from Area Fraction", 
    value=False,
    help="When checked, area fractions are normalized so that HCP + FCC = 100% (boundaries excluded)"
)

# 4. Color Thresholding Tolerance (Advanced)
st.sidebar.subheader("Segmentation Settings")
use_auto_threshold = st.sidebar.checkbox("Use Automatic Color Detection", value=True)
if not use_auto_threshold:
    st.sidebar.warning("Manual HSV ranges not implemented for this demo to keep it simple. Using fixed robust ranges.")

# --- Helper Function to Convert BMP ---
def convert_bmp_to_png(img_file):
    """Convert BMP image to PNG format in memory"""
    try:
        # Open the BMP file with PIL
        img = Image.open(img_file)
        
        # Convert to RGB if necessary (handle different modes)
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save to bytes buffer as PNG
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        return img, buffer
    except Exception as e:
        st.error(f"Error converting image: {e}")
        return None, None

# --- Main Processing Logic ---

if uploaded_file is not None:
    # Check file extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Convert BMP to PNG if necessary
    if file_extension == 'bmp':
        st.info("🔄 Converting BMP to PNG format...")
        img, converted_buffer = convert_bmp_to_png(uploaded_file)
        
        if img is None:
            st.error("Failed to convert BMP file. Please ensure the file is a valid image.")
            st.stop()
        
        st.success("✅ BMP converted successfully!")
        img_np = np.array(img)
    else:
        # Load image normally for other formats
        image = Image.open(uploaded_file)
        img_np = np.array(image)
    
    # Remove Alpha channel if present
    if img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]
        
    h, w, _ = img_np.shape
    
    # Calculate Pixel Resolution
    # Physical Area = domain_size^2
    # Pixel Area = (Physical Area) / (Total Pixels)
    total_physical_area = domain_size_um ** 2
    total_pixels = h * w
    area_per_pixel = total_physical_area / total_pixels
    
    st.metric("Image Resolution", f"{w} x {h} pixels")
    st.metric("Calibration Factor", f"{area_per_pixel:.4f} µm²/pixel")

    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    # Define Range for Red (HCP)
    # Red is usually around 0 or 180 hue. We check both ranges.
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask1, mask2)

    # Define Range for Green (FCC)
    # Green is usually around 60 hue.
    lower_green = np.array([40, 70, 50])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # --- Calculations ---
    
    # 1. Area Calculations (Absolute - these don't change)
    red_pixels = cv2.countNonZero(mask_red)
    green_pixels = cv2.countNonZero(mask_green)
    boundary_pixels = total_pixels - red_pixels - green_pixels
    
    red_area_abs = red_pixels * area_per_pixel
    green_area_abs = green_pixels * area_per_pixel
    boundary_area_abs = boundary_pixels * area_per_pixel
    
    # 2. Area Fraction Calculations
    if exclude_boundaries:
        # Normalize to exclude boundaries (HCP + FCC = 100%)
        total_phase_pixels = red_pixels + green_pixels
        
        if total_phase_pixels > 0:
            red_fraction = (red_pixels / total_phase_pixels) * 100
            green_fraction = (green_pixels / total_phase_pixels) * 100
        else:
            red_fraction = 0
            green_fraction = 0
            
        boundary_fraction = 0  # Set to zero since we're excluding it
        
        st.info("📊 **Mode:** Area fractions normalized to exclude grain boundaries (HCP + FCC = 100%)")
    else:
        # Include boundaries in total area
        red_fraction = (red_pixels / total_pixels) * 100
        green_fraction = (green_pixels / total_pixels) * 100
        boundary_fraction = (boundary_pixels / total_pixels) * 100
        
        st.info("📊 **Mode:** Area fractions include grain boundaries in total area")

    # 3. Morphology Analysis using Scikit-Image
    def analyze_morphology(mask, phase_name, area_per_pixel):
        # Label connected components
        labeled_mask = label(mask)
        props = regionprops(labeled_mask)
        
        if not props:
            return pd.DataFrame()
            
        data = []
        for prop in props:
            # Filter noise (grains smaller than 5 pixels)
            if prop.area < 5:
                continue
                
            area_um2 = prop.area * area_per_pixel
            # Equivalent Circle Diameter: sqrt(4*Area/pi)
            ecd_um = np.sqrt(4 * area_um2 / np.pi)
            
            # Circularity: 4*pi*Area / Perimeter^2 (1 = perfect circle)
            # Adding small epsilon to avoid division by zero
            perimeter = prop.perimeter if prop.perimeter > 0 else 0.001
            circularity = (4 * np.pi * prop.area) / (perimeter ** 2)
            
            # Aspect Ratio
            if prop.major_axis_length > 0:
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

    df_red_morph = analyze_morphology(mask_red, "HCP (Red)", area_per_pixel)
    df_green_morph = analyze_morphology(mask_green, "FCC (Green)", area_per_pixel)
    df_all_morph = pd.concat([df_red_morph, df_green_morph], ignore_index=True)

    # --- Display Results ---

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Phase Quantification")
        
        # Show absolute areas (these never change)
        st.write("**Absolute Areas:**")
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("HCP Area", f"{red_area_abs:.1f} µm²", f"{red_pixels} pixels")
        kpi2.metric("FCC Area", f"{green_area_abs:.1f} µm²", f"{green_pixels} pixels")
        kpi3.metric("Boundary Area", f"{boundary_area_abs:.1f} µm²", f"{boundary_pixels} pixels")
        
        st.divider()
        
        # Show area fractions (these change based on mode)
        st.write("**Area Fractions:**")
        if exclude_boundaries:
            kpi4, kpi5 = st.columns(2)
            kpi4.metric("HCP Fraction", f"{red_fraction:.2f}%", "Normalized")
            kpi5.metric("FCC Fraction", f"{green_fraction:.2f}%", "Normalized")
            st.caption("✅ HCP + FCC = 100% (boundaries excluded)")
        else:
            kpi4, kpi5, kpi6 = st.columns(3)
            kpi4.metric("HCP Fraction", f"{red_fraction:.1f}%")
            kpi5.metric("FCC Fraction", f"{green_fraction:.1f}%")
            kpi6.metric("Boundaries", f"{boundary_fraction:.1f}%")
            st.caption("📏 Fractions include grain boundaries")
        
        st.write("**Area Fraction Chart:**")
        if exclude_boundaries:
            chart_data = pd.DataFrame({
                "Phase": ["HCP (Red)", "FCC (Green)"],
                "Area Fraction (%)": [red_fraction, green_fraction]
            })
        else:
            chart_data = pd.DataFrame({
                "Phase": ["HCP (Red)", "FCC (Green)", "Boundaries"],
                "Area Fraction (%)": [red_fraction, green_fraction, boundary_fraction]
            })
        st.bar_chart(chart_data.set_index("Phase"))

    with col2:
        st.subheader("🖼️ Segmentation Preview")
        c1, c2 = st.columns(2)
        with c1:
            st.image(mask_red, caption="Detected HCP (Red)", use_column_width=True)
        with c2:
            st.image(mask_green, caption="Detected FCC (Green)", use_column_width=True)

    st.divider()

    # --- Morphology Section ---
    st.subheader("📐 Morphological Metrics")
    
    if not df_all_morph.empty:
        # Summary Statistics
        st.write("**Summary Statistics per Phase:**")
        summary = df_all_morph.groupby("Phase")[["Area (µm²)", "ECD (µm)", "Circularity", "Aspect Ratio"]].mean()
        summary['Grain Count'] = df_all_morph.groupby("Phase").size()
        # Reorder columns
        summary = summary[['Grain Count', 'Area (µm²)', 'ECD (µm)', 'Circularity', 'Aspect Ratio']]
        st.dataframe(summary.style.format("{:.2f}"))
        
        # Detailed Data Download
        st.download_button(
            label="📥 Download Detailed Grain Data (CSV)",
            data=df_all_morph.to_csv(index=False).encode('utf-8'),
            file_name='grain_morphology_data.csv',
            mime='text/csv',
        )
        
        # Histograms - Fixed version using proper binning
        st.write("**Grain Size Distribution (ECD):**")
        
        try:
            import plotly.express as px
            plotly_available = True
        except ImportError:
            plotly_available = False
            st.warning("Install plotly for better histograms: `pip install plotly`")
        
        col_h1, col_h2 = st.columns(2)
        
        with col_h1:
            if not df_red_morph.empty:
                st.write("**HCP Phase**")
                if plotly_available:
                    fig_red = px.histogram(df_red_morph, x="ECD (µm)", nbins=20, 
                                          title="HCP Grain Size Distribution",
                                          labels={"ECD (µm)": "Equivalent Diameter (µm)"})
                    st.plotly_chart(fig_red, use_container_width=True)
                else:
                    # Fallback: simple stats without plot
                    st.write(f"- Count: {len(df_red_morph)} grains")
                    st.write(f"- Mean ECD: {df_red_morph['ECD (µm)'].mean():.2f} µm")
                    st.write(f"- Std Dev: {df_red_morph['ECD (µm)'].std():.2f} µm")
                    st.write(f"- Min: {df_red_morph['ECD (µm)'].min():.2f} µm")
                    st.write(f"- Max: {df_red_morph['ECD (µm)'].max():.2f} µm")
        
        with col_h2:
            if not df_green_morph.empty:
                st.write("**FCC Phase**")
                if plotly_available:
                    fig_green = px.histogram(df_green_morph, x="ECD (µm)", nbins=20,
                                            title="FCC Grain Size Distribution",
                                            labels={"ECD (µm)": "Equivalent Diameter (µm)"})
                    st.plotly_chart(fig_green, use_container_width=True)
                else:
                    # Fallback: simple stats without plot
                    st.write(f"- Count: {len(df_green_morph)} grains")
                    st.write(f"- Mean ECD: {df_green_morph['ECD (µm)'].mean():.2f} µm")
                    st.write(f"- Std Dev: {df_green_morph['ECD (µm)'].std():.2f} µm")
                    st.write(f"- Min: {df_green_morph['ECD (µm)'].min():.2f} µm")
                    st.write(f"- Max: {df_green_morph['ECD (µm)'].max():.2f} µm")
        
        st.write("**Circularity Distribution (Shape):**")
        if not df_all_morph.empty:
            if plotly_available:
                fig_circ = px.histogram(df_all_morph, x="Circularity", color="Phase", 
                                       nbins=20, title="Circularity by Phase (1.0 = Perfect Circle)",
                                       labels={"Circularity": "Circularity", "Phase": "Phase"})
                st.plotly_chart(fig_circ, use_container_width=True)
            else:
                # Fallback: simple stats
                st.write("**HCP Circularity:**")
                st.write(f"- Mean: {df_red_morph['Circularity'].mean():.3f}" if not df_red_morph.empty else "- No data")
                st.write("**FCC Circularity:**")
                st.write(f"- Mean: {df_green_morph['Circularity'].mean():.3f}" if not df_green_morph.empty else "- No data")

    else:
        st.warning("No grains detected. Try adjusting the image or segmentation settings.")

else:
    st.info("👈 Please upload an image from the sidebar to begin analysis.")
    
    # Display example logic explanation
    with st.expander("How the calculation works"):
        st.write("""
        1. **Pixel Counting:** The app counts red and green pixels using HSV color thresholding.
        2. **Calibration:** 
           - Total Domain = 250 µm x 250 µm = 62,500 µm².
           - Pixel Area = 62,500 / (Image Width * Image Height).
        3. **Absolute Area:** Pixel Count * Pixel Area.
        4. **Area Fractions:**
           - **With boundaries:** Fraction of total image area
           - **Without boundaries:** Normalized so HCP + FCC = 100%
        5. **Morphology:** Uses connected component labeling to identify individual grains and calculate shape factors.
        """)
    
    with st.expander("Supported File Formats"):
        st.write("""
        - ✅ PNG
        - ✅ JPG/JPEG
        - ✅ BMP (automatically converted to PNG)
        - ✅ TIFF
        """)
    
    with st.expander("Required Packages"):
        st.write("""
        ```bash
        pip install streamlit opencv-python-headless numpy pandas scikit-image pillow plotly
        ```
        """)
