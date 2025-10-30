# MIRI JWST Data Analysis - NGC 7469
# Streamlit Version (Fixed for Streamlit 1.40+)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Configure Streamlit page
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

    def create_simulated_data(self, shape=(200, 200), noise_level=0.1):
        y, x = np.indices(shape)
        center_y, center_x = shape[0] // 2, shape[1] // 2
        r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        agn_core = 10 * np.exp(-(r ** 2) / (2 * 5 ** 2))
        theta = np.arctan2(y - center_y, x - center_x)
        spiral_arms = 2 * np.exp(-r / 50) * (1 + 0.5 * np.sin(2 * theta + 0.5 * np.log(1 + r / 10)))

        rng = np.random.default_rng(42)
        noise = noise_level * rng.normal(size=shape)

        return agn_core + spiral_arms + noise

    def create_visualizations(self, data):
        figures = {}

        # Main 2x2 analysis plot
        fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig1.suptitle('MIRI JWST Analysis: NGC 7469', fontsize=16, fontweight='bold')

        im1 = ax1.imshow(data, cmap='viridis', origin='lower')
        ax1.set_title('Original Data - Viridis')
        plt.colorbar(im1, ax=ax1)

        log_data = np.log10(data + 1e-10)
        im2 = ax2.imshow(log_data, cmap='plasma', origin='lower')
        ax2.set_title('Log Scale - Plasma')
        plt.colorbar(im2, ax=ax2)

        ax3.hist(data.flatten(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax3.set_title('Flux Distribution')
        ax3.grid(True, alpha=0.3)

        center_y, center_x = np.array(data.shape) // 2
        y, x = np.indices(data.shape)
        r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2).astype(int)
        radial_profile = np.bincount(r.ravel(), data.ravel()) / np.bincount(r.ravel())

        ax4.plot(radial_profile, color='red', linewidth=2)
        ax4.fill_between(range(len(radial_profile)), radial_profile, color='red', alpha=0.3)
        ax4.set_title('Radial Profile')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        figures['main_analysis'] = fig1

        # Simulated spectrum plot
        fig2, ax = plt.subplots(figsize=(10, 6))
        wavelength = np.linspace(5, 28, 1000)
        continuum = 1e-15 * wavelength ** -2
        pah_features = (0.5e-15 * np.exp(-(wavelength - 6.2) ** 2 / 0.1) +
                        0.8e-15 * np.exp(-(wavelength - 7.7) ** 2 / 0.15))
        total_flux = continuum + pah_features

        ax.plot(wavelength, total_flux, color='navy', linewidth=2, label='Simulated Spectrum')
        ax.set_xlabel('Wavelength (μm)')
        ax.set_ylabel('Flux [erg/s/cm²/Å]')
        ax.set_title('Simulated MIR Spectrum')
        ax.legend()
        ax.grid(True, alpha=0.3)

        figures['spectrum'] = fig2
        return figures


def main():
    st.title("🔭 MIRI JWST Data Analysis")
    st.subheader("NGC 7469 - Seyfert 1 Galaxy")

    analyzer = NGC7469Analyzer()

    with st.sidebar:
        st.header("Analysis Controls")
        image_size = st.slider("Image Size", 100, 300, 200)
        noise_level = st.slider("Noise Level", 0.0, 1.0, 0.1)
        generate_btn = st.button("Generate Analysis")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Object Information")
        st.write(f"**Name:** {analyzer.object_info['name']}")
        st.write(f"**Coordinates:** {analyzer.object_info['ra']}, {analyzer.object_info['dec']}")
        st.write(f"**Distance:** {analyzer.object_info['distance_mpc']} Mpc")
        st.write(f"**Redshift:** z = {analyzer.object_info['redshift']}")
        st.write(f"**Type:** {analyzer.object_info['subcategory']}")

        st.header("Scientific Context")
        st.info(
            """
            NGC 7469 is a Seyfert 1 galaxy exhibiting starburst activity, making it a key subject
            for studying AGN-starburst connections using MIRI’s mid-infrared capabilities.

            **MIRI Advantages:**
            - Penetrates dusty regions  
            - Reveals PAH emission features  
            - Maps star formation & AGN activity
            """
        )

    with col2:
        if generate_btn:
            with st.spinner("Generating MIRI data and analysis..."):
                data = analyzer.create_simulated_data(shape=(image_size, image_size),
                                                     noise_level=noise_level)
                figures = analyzer.create_visualizations(data)

                # Display figures
                st.pyplot(figures['main_analysis'])
                st.pyplot(figures['spectrum'])
        else:
            st.info("Click **Generate Analysis** to begin simulation.")

if __name__ == "__main__":
    main()
