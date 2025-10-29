# MIRI JWST Data Analysis - NGC 7469
# Streamlit Version

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="MIRI JWST Analyzer",
    page_icon="🔭",
    layout="wide"
)

class NGC7469Analyzer:
    def __init__(self):
        self.object_info = {
            'name': "NGC 7469",
            'ra': "23h 03m 15.61s",
            'dec': "+08d 52m 26.0s",
            'distance_mpc': 62.1,
            'redshift': 0.016335,
            'category': "Galaxy",
            'subcategory': "Seyfert 1 Galaxy, Starburst Galaxy"
        }
        
    def create_simulated_data(self, shape=(200, 200)):
        """Create realistic simulated MIRI data"""
        y, x = np.indices(shape)
        center_y, center_x = shape[0]//2, shape[1]//2
        
        # Create galaxy profile
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Bright AGN core
        agn_core = 10 * np.exp(-(r**2) / (2 * 5**2))
        
        # Spiral arm structure
        theta = np.arctan2(y - center_y, x - center_x)
        spiral_arms = 2 * np.exp(-r/50) * (1 + 0.5 * np.sin(2 * theta + 0.5 * np.log(1 + r/10)))
        
        # Background noise
        rng = np.random.RandomState(42)
        noise = 0.1 * rng.normal(size=shape)
        
        # Combine components
        simulated_data = agn_core + spiral_arms + noise
        
        return simulated_data
    
    def create_visualizations(self, data):
        """Create visualizations and return matplotlib figures"""
        figures = {}
        
        # Main analysis figure
        fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig1.suptitle(f'MIRI JWST Analysis: {self.object_info["name"]}', fontsize=16, fontweight='bold')
        
        # Original data
        im1 = ax1.imshow(data, cmap='viridis', origin='lower')
        ax1.set_title('Original Data - Viridis')
        ax1.set_xlabel('Pixel X')
        ax1.set_ylabel('Pixel Y')
        plt.colorbar(im1, ax=ax1)
        
        # Log scale
        log_data = np.log10(data + 1e-10)
        im2 = ax2.imshow(log_data, cmap='plasma', origin='lower')
        ax2.set_title('Log Scale - Plasma')
        ax2.set_xlabel('Pixel X')
        ax2.set_ylabel('Pixel Y')
        plt.colorbar(im2, ax=ax2)
        
        # Histogram
        ax3.hist(data.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_xlabel('Flux Value')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Flux Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Radial profile
        center_y, center_x = np.array(data.shape) // 2
        y, x = np.indices(data.shape)
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
        radial_profile = np.bincount(r.ravel(), data.ravel()) / np.bincount(r.ravel())
        
        ax4.plot(radial_profile, color='red', linewidth=2)
        ax4.set_xlabel('Radius (pixels)')
        ax4.set_ylabel('Average Flux')
        ax4.set_title('Radial Profile')
        ax4.grid(True, alpha=0.3)
        ax4.fill_between(range(len(radial_profile)), radial_profile, alpha=0.3, color='red')
        
        plt.tight_layout()
        figures['main_analysis'] = fig1
        
        # Individual figures for separate display
        # Spectral analysis simulation
        fig2, ax = plt.subplots(figsize=(10, 6))
        wavelength = np.linspace(5, 28, 1000)
        continuum = 1e-15 * wavelength**-2
        pah_features = (0.5e-15 * np.exp(-(wavelength - 6.2)**2 / 0.1) +
                       0.8e-15 * np.exp(-(wavelength - 7.7)**2 / 0.15))
        total_flux = continuum + pah_features
        
        ax.plot(wavelength, total_flux, color='navy', linewidth=2, label='Simulated Spectrum')
        ax.set_xlabel('Wavelength (μm)')
        ax.set_ylabel('Flux [erg/s/cm²/Å]')
        ax.set_title('Simulated MIR Spectrum')
        ax.grid(True, alpha=0.3)
        ax.legend()
        figures['spectrum'] = fig2
        
        return figures

def main():
    """Main Streamlit app"""
    st.title("🔭 MIRI JWST Data Analysis")
    st.subheader("NGC 7469 - Seyfert 1 Galaxy")
    
    # Initialize analyzer
    analyzer = NGC7469Analyzer()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Analysis Controls")
        image_size = st.slider("Image Size", 100, 300, 200)
        noise_level = st.slider("Noise Level", 0.0, 1.0, 0.1)
        generate_btn = st.button("Generate Analysis")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Object Information")
        st.write(f"**Name:** {analyzer.object_info['name']}")
        st.write(f"**Coordinates:** {analyzer.object_info['ra']}, {analyzer.object_info['dec']}")
        st.write(f"**Distance:** {analyzer.object_info['distance_mpc']} Mpc")
        st.write(f"**Redshift:** z = {analyzer.object_info['redshift']}")
        st.write(f"**Type:** {analyzer.object_info['subcategory']}")
        
        st.header("Scientific Context")
        st.info("""
        NGC 7469 is a remarkable Seyfert 1 galaxy with concurrent starburst activity, 
        making it an ideal laboratory for studying AGN-starburst connections.
        
        **MIRI Advantages:**
        - Penetrates dusty regions
        - Reveals PAH features
        - Maps star formation and AGN activity
        """)
    
    with col2:
        if generate_btn:
            with st.spinner("Generating MIRI data and analysis..."):
                # Create simulated data
                data = analyzer.create_simulated_data(shape=(image_size, image_size))
                
                # Create visualizations
                figures = analyzer.create_visualizations(data)
                
                # Display main analysis figure
                st.subheader("Comprehensive Analysis")
                st.pyplot(figures['main_analysis'])
                
                # Display spectrum
                st.subheader("Spectral Analysis")
                st.pyplot(figures['spectrum'])
                
                # Data statistics
                st.subheader("Data Statistics")
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Mean Flux", f"{np.mean(data):.4f}")
                with col_stat2:
                    st.metric("Max Flux", f"{np.max(data):.4f}")
                with col_stat3:
                    st.metric("Data Points", f"{data.size:,}")
                
                # Download option
                st.subheader("Export Results")
                if st.button("Download Analysis Report"):
                    # Create a simple text report
                    report = f"""
MIRI JWST ANALYSIS REPORT
Target: {analyzer.object_info['name']}
Generated: {np.datetime64('now')}

Data Statistics:
- Mean Flux: {np.mean(data):.4f}
- Max Flux: {np.max(data):.4f}
- Standard Deviation: {np.std(data):.4f}
- Image Size: {image_size} x {image_size}

Object Properties:
- Distance: {analyzer.object_info['distance_mpc']} Mpc
- Redshift: {analyzer.object_info['redshift']}
- Type: {analyzer.object_info['subcategory']}
"""
                    st.download_button(
                        label="Download Report as TXT",
                        data=report,
                        file_name="ngc7469_analysis_report.txt",
                        mime="text/plain"
                    )
        else:
            st.info("👈 Adjust parameters in the sidebar and click 'Generate Analysis' to start")
            
            # Show sample image
            st.subheader("About MIRI JWST")
            st.image("https://www.jwst.nasa.gov/content/webbLaunch/assets/images/miri/MIRI-1.jpg", 
                    caption="MIRI Instrument on JWST", use_column_width=True)

    # Footer
    st.markdown("---")
    st.caption("MIRI JWST Data Analysis Tool | NGC 7469 Analysis Pipeline")

if __name__ == "__main__":
    main()
