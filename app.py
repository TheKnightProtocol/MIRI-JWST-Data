"""
Streamlit app: JWST MIRI MIR analysis for NGC 7469 (real FITS or synthetic)
Single-file Streamlit application that mirrors and extends the supplied script
- Upload a FITS cube or use simulated data
- Interactive slice selector, ROI controls, spectrum extraction (center vs ring)
- Live plots, download buttons for CSV/PNG, metadata display
- Uses astropy, numpy, pandas, matplotlib, plotly, streamlit
"""

import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import plotly.express as px
import streamlit as st
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="MIRI — NGC 7469 Explorer", layout="wide")
TITLE = "JWST MIRI Explorer — NGC 7469 (Real or Simulated)"
st.title(TITLE)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Data Source")
use_sample = st.sidebar.checkbox("Use built-in simulated cube", value=True)
uploaded_file = st.sidebar.file_uploader("Or upload a MIRI FITS cube (.fits)", type=["fits", "fz"], accept_multiple_files=False)

st.sidebar.markdown("---")
st.sidebar.header("Cosmology & Scale")
H0 = st.sidebar.number_input("H0 (km/s/Mpc)", value=70.0)
Om0 = st.sidebar.number_input("Om0", value=0.3)
redshift_z = st.sidebar.number_input("Redshift (z)", value=0.016335)

cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
ang_diam_dist = cosmo.angular_diameter_distance(redshift_z)

st.sidebar.markdown("---")
st.sidebar.header("Visualization Options")
show_log = st.sidebar.checkbox("Log scale for images", value=False)
cmap = st.sidebar.selectbox("Colormap", options=["magma", "inferno", "viridis", "plasma", "cividis"], index=0)

# -----------------------------
# Helper utilities
# -----------------------------
@st.cache_data
def generate_synthetic_cube(nx=150, ny=150, nwav=30, wav_min=5.0, wav_max=12.0, seed=42):
    np.random.seed(seed)
    wavelengths = np.linspace(wav_min, wav_max, nwav)
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    ring = np.exp(-((R - 0.6) / 0.2) ** 2)
    center = np.exp(-((R) / 0.25) ** 2)
    sci = np.zeros((nwav, ny, nx))
    for i, w in enumerate(wavelengths):
        sci[i] = (
            center * (1 + 0.4 * np.sin(2 * np.pi * (w - wav_min) / (wav_max - wav_min)))
            + 0.6 * ring * (1 + 0.3 * np.cos(2 * np.pi * (w - wav_min) / ((wav_max-wav_min)/3)))
            + 0.05 * np.random.randn(ny, nx)
        )
    return sci, wavelengths

@st.cache_data
def load_fits_cube(fobj):
    # Accept file-like (uploaded) or filepath
    with fits.open(fobj) as hdul:
        # common HLSP cubes store data in 'SCI' extension; fallback to [0]
        if "SCI" in hdul:
            hdu = hdul["SCI"]
        else:
            hdu = hdul[0]
        data = np.array(hdu.data)
        header = hdu.header
    return data, header

def header_pixel_scale(header):
    # try common header keys
    cdelt = header.get("CDELT1") or header.get("CD1_1") or header.get("PC1_1")
    if cdelt is None:
        return None
    try:
        pixel_scale_deg = float(cdelt) * u.deg
        pixel_scale_arcsec = pixel_scale_deg.to(u.arcsec)
        theta_rad = pixel_scale_arcsec.to(u.rad)
        return pixel_scale_deg, pixel_scale_arcsec, theta_rad
    except Exception:
        return None

# -----------------------------
# Load data: real or synthetic
# -----------------------------
if (uploaded_file is not None) and (not use_sample):
    st.info("Loading uploaded FITS cube...")
    try:
        sci_data, header = load_fits_cube(uploaded_file)
        using_real = True
    except Exception as e:
        st.error(f"Failed to read FITS: {e}\nFalling back to simulated cube.")
        sci_data, wavelengths = generate_synthetic_cube()
        header = {}
        using_real = False
else:
    sci_data, wavelengths = generate_synthetic_cube()
    header = {}
    using_real = False

# If FITS cube has shape (ny, nx, nwav) or (nwav, ny, nx) — try to detect
if using_real:
    # many JWST HLSP cubes are (lambda, y, x)
    if sci_data.ndim == 4:
        # sometimes extra singleton axes exist, try to reduce
        sci_data = np.squeeze(sci_data)
    if sci_data.ndim == 3:
        if sci_data.shape[0] < 10 and sci_data.shape[-1] > 10:
            # likely (y, x, lambda) -> transpose
            sci_data = np.transpose(sci_data, (2, 0, 1))
    elif sci_data.ndim == 2:
        # single image -> make dummy spectral axis
        sci_data = sci_data[None, ...]
else:
    # already (nwav, ny, nx)
    pass

nwav, ny, nx = sci_data.shape
# If wavelengths were not loaded from file, make a default vector
if 'wavelengths' not in locals():
    wavelengths = np.linspace(5.0, 12.0, nwav)

# Pixel scale from header if possible
pix_info = None
if using_real and isinstance(header, dict):
    ps = header_pixel_scale(header)
    if ps is not None:
        pixel_scale_deg, pixel_scale_arcsec, theta_rad = ps
        pix_info = (pixel_scale_deg, pixel_scale_arcsec, theta_rad)
else:
    # default simulated pixel scale (deg)
    pixel_scale_deg = 0.00015 * u.deg
    pixel_scale_arcsec = pixel_scale_deg.to(u.arcsec)
    theta_rad = pixel_scale_arcsec.to(u.rad)
    pix_info = (pixel_scale_deg, pixel_scale_arcsec, theta_rad)

# Physical pixel scale in pc
pixel_scale_pc = (ang_diam_dist * theta_rad).to(u.pc, equivalencies=u.dimensionless_angles())

# -----------------------------
# Page layout: main columns
# -----------------------------
left_col, right_col = st.columns([2, 1])

with right_col:
    st.subheader("Dataset & Metadata")
    st.markdown(f"**Source:** {'Uploaded FITS' if using_real else 'Simulated cube (built-in)'}")
    st.markdown(f"**Shape (nwav, ny, nx):** {sci_data.shape}")
    st.markdown(f"**Wavelength range (um):** {wavelengths[0]:.2f} — {wavelengths[-1]:.2f}")
    st.markdown(f"**Angular pixel scale:** {pixel_scale_arcsec:.4f}")
    st.markdown(f"**Physical scale (pc/pixel)** (z={redshift_z}): {pixel_scale_pc:.3f}")
    if using_real and header:
        st.download_button("Download FITS header (txt)", data=str(dict(header)), file_name="fits_header.txt")

    st.markdown("---")
    st.subheader("ROI Controls")
    center_radius = st.slider("Center aperture radius (pixels)", 1, min(nx, ny)//4, value=8)
    ring_inner = st.slider("Ring inner radius (pixels)", center_radius+1, min(nx, ny)//2, value=center_radius+6)
    ring_outer = st.slider("Ring outer radius (pixels)", ring_inner+1, min(nx, ny)//2+10, value=ring_inner+6)
    st.markdown("(Apertures are circular, centered on image center.)")

with left_col:
    st.subheader("Interactive Visuals")
    # Slice selector
    slice_idx = st.slider("Wavelength slice index", 0, nwav-1, value=nwav//2)
    slice_wav = wavelengths[slice_idx]

    img = sci_data[slice_idx]
    fig, ax = plt.subplots(figsize=(6,6))
    if show_log:
        im = ax.imshow(img, origin='lower', cmap=cmap, norm=LogNorm(vmin=max(img.min(), 1e-6), vmax=img.max()))
    else:
        im = ax.imshow(img, origin='lower', cmap=cmap)
    ax.set_title(f"Slice {slice_idx} — {slice_wav:.3f} µm")
    ax.set_xlabel('X pixel')
    ax.set_ylabel('Y pixel')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Flux (arb. units)')
    st.pyplot(fig)

    # Histogram
    hist_fig, hist_ax = plt.subplots(figsize=(6,2.5))
    hist_ax.hist(img.flatten(), bins=60)
    hist_ax.set_title('Flux distribution — selected slice')
    hist_ax.set_xlabel('Flux (arb. units)')
    hist_ax.set_ylabel('Pixel count')
    st.pyplot(hist_fig)

    # Average flux map over wavelengths
    st.subheader('Average Flux Map (wavelength-averaged)')
    avg_map = np.mean(sci_data, axis=0)
    heat_fig = px.imshow(avg_map, origin='lower', labels=dict(x='X pixel', y='Y pixel', color='Avg flux'))
    st.plotly_chart(heat_fig, use_container_width=True)

# -----------------------------
# Spectral extraction: center vs ring
# -----------------------------
st.markdown('---')
st.subheader('Spectral Extraction — Center vs Ring')

# build masks
cy, cx = ny//2, nx//2
Y, X = np.indices((ny, nx))
R = np.sqrt((X-cx)**2 + (Y-cy)**2)
center_mask = R <= center_radius
ring_mask = (R >= ring_inner) & (R <= ring_outer)

center_spec = np.array([sci_data[i][center_mask].mean() for i in range(nwav)])
ring_spec = np.array([sci_data[i][ring_mask].mean() for i in range(nwav)])

spec_df = pd.DataFrame({"wavelength_um": wavelengths, "center_flux": center_spec, "ring_flux": ring_spec})
ratio = spec_df["center_flux"] / (spec_df["ring_flux"] + 1e-12)
spec_df["flux_ratio_center_ring"] = ratio

# Plot interactive spectra with plotly
fig_spec = px.line(spec_df, x='wavelength_um', y=['center_flux', 'ring_flux'], labels={'value':'Flux (arb. units)', 'wavelength_um':'Wavelength (µm)'}, title='Extracted Spectra')
st.plotly_chart(fig_spec, use_container_width=True)

fig_ratio = px.line(spec_df, x='wavelength_um', y='flux_ratio_center_ring', title='Center / Ring Flux Ratio')
fig_ratio.add_hline(y=1.0, line_dash='dash')
st.plotly_chart(fig_ratio, use_container_width=True)

# Download CSVs
csv_center = spec_df[['wavelength_um','center_flux']].to_csv(index=False).encode('utf-8')
csv_ring = spec_df[['wavelength_um','ring_flux']].to_csv(index=False).encode('utf-8')
csv_ratio = spec_df[['wavelength_um','flux_ratio_center_ring']].to_csv(index=False).encode('utf-8')

col1, col2, col3 = st.columns(3)
col1.download_button('Download center spectrum (CSV)', csv_center, file_name='center_spectrum.csv')
col2.download_button('Download ring spectrum (CSV)', csv_ring, file_name='ring_spectrum.csv')
col3.download_button('Download flux ratio (CSV)', csv_ratio, file_name='flux_ratio.csv')

# Save PNGs on demand
buffer = io.BytesIO()
plt.figure(figsize=(6,4))
plt.plot(spec_df['wavelength_um'], spec_df['center_flux'], label='Center')
plt.plot(spec_df['wavelength_um'], spec_df['ring_flux'], label='Ring')
plt.legend()
plt.xlabel('Wavelength (µm)')
plt.ylabel('Flux (arb. units)')
plt.title('Center vs Ring (Matplotlib)')
plt.tight_layout()
plt.savefig(buffer, format='png', dpi=150)
buffer.seek(0)

st.download_button('Download Matplotlib spectrum PNG', buffer.getvalue(), file_name='spectrum_matplotlib.png', mime='image/png')

# -----------------------------
# Footer & help
# -----------------------------
st.markdown('---')
with st.expander('Notes & How to use'):
    st.markdown(
        """
        - Use the sidebar to upload a FITS cube or check 'Use built-in simulated cube'.
        - ROI controls define the center aperture and the annular ring (in image pixels).
        - The app supports common HLSP-style FITS cubes where the spectral axis is first.
        - If the uploaded FITS lacks header pixel scale keys, the app falls back to a simulated scale.
        - Use the Download buttons to export CSV or PNG for publication.
        """
    )

st.success('Ready — interact with the panels and download results.')

# Optional: expose outputs in a local Outputs/ directory for debugging
if st.button('Save outputs to Outputs/ (server)'):
    os.makedirs('Outputs', exist_ok=True)
    spec_df.to_csv('Outputs/center_ring_fluxes.csv', index=False)
    avg_map = np.mean(sci_data, axis=0)
    np.savetxt('Outputs/avg_map.txt', avg_map)
    st.write('Saved Outputs/center_ring_fluxes.csv and Outputs/avg_map.txt')


