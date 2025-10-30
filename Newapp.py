"""
JWST MIRI MIR Analysis — NGC 7469 (Real FITS or Synthetic Cube)
Author: Sankalp Sharma | Version 2.1 (Streamlit Cloud Safe)
--------------------------------------------------------------
Features:
- Upload or simulate JWST/MIRI spectral cubes
- Cosmology-based scale conversion
- Interactive slice & ROI visualization
- Spectral extraction (center vs. ring)
- Live plots (Matplotlib + Plotly)
- Export CSV & PNG outputs
--------------------------------------------------------------
"""

# -----------------------------
# Imports and configuration
# -----------------------------
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless backend for Streamlit Cloud
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import plotly.express as px
import streamlit as st
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# -----------------------------
# Streamlit settings
# -----------------------------
st.set_page_config(page_title="MIRI — NGC 7469 Explorer", layout="wide")

# Title
st.title("🔭 JWST MIRI Explorer — NGC 7469 (Real or Simulated)")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("📁 Data Source")
use_sample = st.sidebar.checkbox("Use built-in simulated cube", value=True)
uploaded_file = st.sidebar.file_uploader("Or upload a MIRI FITS cube (.fits)", type=["fits", "fz"])

st.sidebar.markdown("---")
st.sidebar.header("🌌 Cosmology & Scale")
H0 = st.sidebar.number_input("H₀ (km/s/Mpc)", value=70.0)
Om0 = st.sidebar.number_input("Ω₀", value=0.3)
redshift_z = st.sidebar.number_input("Redshift (z)", value=0.016335)

cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
ang_diam_dist = cosmo.angular_diameter_distance(redshift_z)

st.sidebar.markdown("---")
st.sidebar.header("🎨 Visualization")
show_log = st.sidebar.checkbox("Log scale for images", value=False)
cmap = st.sidebar.selectbox("Colormap", ["magma", "inferno", "viridis", "plasma", "cividis"], index=1)

# -----------------------------
# Helper functions
# -----------------------------
def generate_synthetic_cube(nx=100, ny=100, nwav=20, wav_min=5.0, wav_max=12.0, seed=42):
    """Generate a synthetic JWST/MIRI-like data cube"""
    np.random.seed(seed)
    wavelengths = np.linspace(wav_min, wav_max, nwav)
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    ring = np.exp(-((R - 0.6) / 0.2) ** 2)
    center = np.exp(-((R) / 0.25) ** 2)
    sci = np.zeros((nwav, ny, nx), dtype=np.float32)
    for i, w in enumerate(wavelengths):
        sci[i] = (
            center * (1 + 0.4 * np.sin(2 * np.pi * (w - wav_min) / (wav_max - wav_min)))
            + 0.6 * ring * (1 + 0.3 * np.cos(2 * np.pi * (w - wav_min) / 3))
            + 0.05 * np.random.randn(ny, nx)
        )
    return sci, wavelengths


def load_fits_cube(fobj):
    """Load FITS data cube"""
    with fits.open(fobj) as hdul:
        if "SCI" in hdul:
            hdu = hdul["SCI"]
        else:
            hdu = hdul[0]
        data = np.array(hdu.data, dtype=np.float32)
        header = dict(hdu.header)
    return data, header


def header_pixel_scale(header):
    """Extract pixel scale from FITS header if available"""
    cdelt = header.get("CDELT1") or header.get("CD1_1") or header.get("PC1_1")
    if cdelt is None:
        return None
    pixel_scale_deg = float(cdelt) * u.deg
    pixel_scale_arcsec = pixel_scale_deg.to(u.arcsec)
    theta_rad = pixel_scale_arcsec.to(u.rad)
    return pixel_scale_deg, pixel_scale_arcsec, theta_rad

# -----------------------------
# Data loading
# -----------------------------
try:
    if uploaded_file and not use_sample:
        sci_data, header = load_fits_cube(uploaded_file)
        using_real = True
    else:
        sci_data, wavelengths = generate_synthetic_cube()
        header = {}
        using_real = False
except Exception as e:
    st.error(f"⚠️ Failed to load FITS: {e}")
    sci_data, wavelengths = generate_synthetic_cube()
    header = {}
    using_real = False

# Ensure correct shape (λ, y, x)
if sci_data.ndim == 4:
    sci_data = np.squeeze(sci_data)
if sci_data.ndim == 3 and sci_data.shape[-1] > sci_data.shape[0]:
    pass
elif sci_data.ndim == 3:
    sci_data = np.transpose(sci_data, (2, 0, 1))

nwav, ny, nx = sci_data.shape
if "wavelengths" not in locals():
    wavelengths = np.linspace(5.0, 12.0, nwav)

# Pixel scale
ps = header_pixel_scale(header) if using_real else None
if ps:
    pixel_scale_deg, pixel_scale_arcsec, theta_rad = ps
else:
    pixel_scale_deg = 0.00015 * u.deg
    pixel_scale_arcsec = pixel_scale_deg.to(u.arcsec)
    theta_rad = pixel_scale_arcsec.to(u.rad)

pixel_scale_pc = (ang_diam_dist * theta_rad).to(u.pc, equivalencies=u.dimensionless_angles())

# -----------------------------
# Layout: metadata + visuals
# -----------------------------
left_col, right_col = st.columns([2, 1])

with right_col:
    st.subheader("📊 Dataset & Metadata")
    st.markdown(f"**Source:** {'Uploaded FITS' if using_real else 'Simulated Cube'}")
    st.markdown(f"**Shape:** {sci_data.shape}")
    st.markdown(f"**λ Range:** {wavelengths[0]:.2f}–{wavelengths[-1]:.2f} µm")
    st.markdown(f"**Angular Pixel Scale:** {str(pixel_scale_arcsec)}")
    st.markdown(f"**Physical Scale (pc/pixel):** {str(pixel_scale_pc):s}")

    st.markdown("---")
    st.subheader("ROI Controls")
    center_radius = st.slider("Center radius (px)", 2, min(nx, ny)//4, value=8)
    ring_inner = st.slider("Ring inner radius (px)", center_radius+1, min(nx, ny)//2, value=center_radius+6)
    ring_outer = st.slider("Ring outer radius (px)", ring_inner+1, min(nx, ny)//2+10, value=ring_inner+6)

with left_col:
    st.subheader("📸 Wavelength Slice Viewer")
    slice_idx = st.slider("Select wavelength slice", 0, nwav-1, value=nwav//2)
    slice_wav = wavelengths[slice_idx]
    img = sci_data[slice_idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    if show_log:
        im = ax.imshow(img, origin="lower", cmap=cmap, norm=LogNorm(vmin=max(img.min(), 1e-6), vmax=img.max()))
    else:
        im = ax.imshow(img, origin="lower", cmap=cmap)
    ax.set_title(f"Slice {slice_idx} — {slice_wav:.2f} µm")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Flux (arb. units)")
    st.pyplot(fig)
    plt.close(fig)

    hist_fig, hist_ax = plt.subplots(figsize=(6, 2.5))
    hist_ax.hist(img.flatten(), bins=50, color="orange", alpha=0.7)
    hist_ax.set_title("Flux Distribution")
    hist_ax.set_xlabel("Flux (arb. units)")
    hist_ax.set_ylabel("Count")
    st.pyplot(hist_fig)
    plt.close(hist_fig)

    avg_map = np.mean(sci_data, axis=0).astype(np.float32)
    st.subheader("Average Flux Map")
    heat_fig = px.imshow(avg_map, origin="lower", labels=dict(x="X pixel", y="Y pixel", color="Avg Flux"))
    st.plotly_chart(heat_fig, use_container_width=True)

# -----------------------------
# Spectral extraction
# -----------------------------
st.markdown("---")
st.subheader("📈 Spectral Extraction — Center vs Ring")

cy, cx = ny//2, nx//2
Y, X = np.indices((ny, nx))
R = np.sqrt((X-cx)**2 + (Y-cy)**2)
center_mask = R <= center_radius
ring_mask = (R >= ring_inner) & (R <= ring_outer)

center_spec = np.array([sci_data[i][center_mask].mean() for i in range(nwav)])
ring_spec = np.array([sci_data[i][ring_mask].mean() for i in range(nwav)])
ratio = center_spec / (ring_spec + 1e-12)

spec_df = pd.DataFrame({
    "Wavelength (µm)": wavelengths,
    "Center Flux": center_spec,
    "Ring Flux": ring_spec,
    "Flux Ratio (C/R)": ratio
})

fig_spec = px.line(spec_df, x="Wavelength (µm)", y=["Center Flux", "Ring Flux"], title="Center vs Ring Spectra")
fig_ratio = px.line(spec_df, x="Wavelength (µm)", y="Flux Ratio (C/R)", title="Flux Ratio (Center/Ring)")
fig_ratio.add_hline(y=1.0, line_dash="dash")

st.plotly_chart(fig_spec, use_container_width=True)
st.plotly_chart(fig_ratio, use_container_width=True)

# -----------------------------
# Downloads
# -----------------------------
st.subheader("📂 Export Results")

col1, col2, col3 = st.columns(3)
col1.download_button("Download Center Spectrum (CSV)", spec_df[["Wavelength (µm)", "Center Flux"]].to_csv(index=False), file_name="center_spectrum.csv")
col2.download_button("Download Ring Spectrum (CSV)", spec_df[["Wavelength (µm)", "Ring Flux"]].to_csv(index=False), file_name="ring_spectrum.csv")
col3.download_button("Download Flux Ratio (CSV)", spec_df[["Wavelength (µm)", "Flux Ratio (C/R)"]].to_csv(index=False), file_name="flux_ratio.csv")

# PNG spectrum download
buffer = io.BytesIO()
plt.figure(figsize=(6, 4))
plt.plot(wavelengths, center_spec, label="Center")
plt.plot(wavelengths, ring_spec, label="Ring")
plt.legend()
plt.xlabel("Wavelength (µm)")
plt.ylabel("Flux (arb. units)")
plt.title("Center vs Ring Spectrum")
plt.tight_layout()
plt.savefig(buffer, format="png", dpi=150)
buffer.seek(0)
st.download_button("Download Spectrum PNG", buffer.getvalue(), file_name="spectrum_plot.png", mime="image/png")
plt.close()

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("🪐 JWST MIRI Analysis Dashboard — Sankalp Sharma | v2.1 (Optimized for Streamlit Cloud)")
