# app.py
"""
Streamlit Dashboard — MIRI JWST Analysis: NGC 7469
Author: Sankalp Sharma | v2.0
Features:
- Tabs: Overview | Image Visualization | Spectral Analysis | Flux Map | Exports
- Upload real JWST MIRI FITS cube (auto-detect SCI extension) or use simulated cube
- Cosmology inputs and pixel -> physical scale conversion using astropy equivalencies
- Interactive slice selector, ROI controls, log scaling, colormap choices
- Spectral extraction (center vs ring), flux-ratio, radial profile
- Download CSV / PNG / text report
"""

import io
import os
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

# -----------------------------
# Utility functions
# -----------------------------
@st.cache_data
def generate_synthetic_cube(nx=150, ny=150, nwav=30, wav_min=5.0, wav_max=12.0, seed=42):
    """Return synthetic cube (nwav, ny, nx) and wavelengths (um)."""
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(wav_min, wav_max, nwav)
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    ring = np.exp(-((R - 0.6) / 0.2) ** 2)
    center = np.exp(-((R) / 0.2) ** 2)

    sci = np.zeros((nwav, ny, nx))
    for i, w in enumerate(wavelengths):
        sci[i] = (
            center * (1 + 0.5 * np.sin(2 * np.pi * (w - wav_min) / (wav_max - wav_min)))
            + 0.5 * ring * (1 + 0.3 * np.cos(2 * np.pi * (w - wav_min) / 3.0))
            + 0.08 * rng.normal(size=(ny, nx))
        )
    return sci, wavelengths

@st.cache_data
def load_fits_cube(file_like) -> Tuple[np.ndarray, dict]:
    """Load a FITS cube from path or uploaded file-like object.
    Returns sci_data (numpy array) and header (dict)."""
    # file_like can be an UploadedFile or filepath
    if hasattr(file_like, "read"):
        header = {}
        with fits.open(file_like) as hdul:
            if "SCI" in hdul:
                hdu = hdul["SCI"]
            else:
                hdu = hdul[0]
            data = np.array(hdu.data)
            header = dict(hdu.header)
    else:
        with fits.open(str(file_like)) as hdul:
            if "SCI" in hdul:
                hdu = hdul["SCI"]
            else:
                hdu = hdul[0]
            data = np.array(hdu.data)
            header = dict(hdu.header)
    # Normalize dims: want (nwav, ny, nx)
    data = np.squeeze(data)
    if data.ndim == 3:
        # heuristics: if first axis > last axis and length small, assume (lambda,y,x)
        # if shape (y,x,l) transpose:
        if data.shape[0] < 10 and data.shape[-1] > 10:
            data = np.transpose(data, (2, 0, 1))
    elif data.ndim == 2:
        data = data[None, ...]
    return data, header

def header_pixel_scale(header: dict):
    """Try to extract pixel scale (deg/pix) from header keys (CDELT1, CD1_1, PIXSCALE)."""
    if not header:
        return None
    for key in ("CDELT1", "CD1_1", "PIXSCALE", "PC1_1"):
        if key in header and header[key] is not None:
            try:
                val = float(header[key])
                # If header PIXSCALE typically is arcsec/pix sometimes; handle both
                # Heuristic: if value > 1e-2 assume arcsec (e.g. 0.1), if < 0.01 assume deg (e.g. 1.5e-4)
                if val > 1.0:  # arcsec/pix typical > 0.01; if >1 assume arcsec
                    return (val * u.arcsec).to(u.deg)
                # otherwise treat as deg directly
                return (val * u.deg)
            except Exception:
                continue
    return None

def compute_physical_scale(pixel_scale_deg: u.Quantity, redshift_z: float, H0=70.0, Om0=0.3):
    """Return pixel scale as (arcsec/pix, pc/pix) given pixel_scale_deg (Quantity)."""
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    ang_diam_dist = cosmo.angular_diameter_distance(redshift_z)
    pixel_scale_arcsec = pixel_scale_deg.to(u.arcsec)
    theta_rad = pixel_scale_arcsec.to(u.rad)
    # Use equivalencies to convert distance*angle -> length
    pixel_scale_pc = (ang_diam_dist * theta_rad).to(u.pc, equivalencies=u.dimensionless_angles())
    return pixel_scale_arcsec, pixel_scale_pc, ang_diam_dist

def make_center_ring_masks(ny: int, nx: int, center_radius: int, ring_inner: int, ring_outer: int):
    cy, cx = ny // 2, nx // 2
    Y, X = np.indices((ny, nx))
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    center_mask = R <= center_radius
    ring_mask = (R >= ring_inner) & (R <= ring_outer)
    return center_mask, ring_mask

def fig_to_bytes(fig, fmt="png"):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight", dpi=150)
    buf.seek(0)
    return buf

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="MIRI Explorer — NGC 7469", layout="wide", page_icon="🔭")
st.sidebar.title("MIRI — NGC 7469")
st.sidebar.markdown("Sankalp Sharma | v2.0  •  Streamlit demo")

# Top-level metadata / cosmology controls
with st.sidebar.expander("Cosmology & Pixel scale", expanded=True):
    H0 = st.number_input("H0 (km/s/Mpc)", value=70.0, step=1.0)
    Om0 = st.number_input("Om0 (Ωm)", value=0.3, step=0.01, format="%.2f")
    redshift_z = st.number_input("Redshift (z)", value=0.016335, format="%.6f")
    st.markdown("---")

# Data source: upload or simulate
st.sidebar.header("Data Source")
use_sim = st.sidebar.checkbox("Use built-in simulated cube", value=True)
uploaded = st.sidebar.file_uploader("Upload MIRI FITS cube (HLSP / s3d)", type=["fits", "fz"])

# Visualization options
st.sidebar.header("Visualization Options")
colormap = st.sidebar.selectbox("Colormap", ["magma", "inferno", "viridis", "plasma", "cividis"])
use_log = st.sidebar.checkbox("Log scale for images", value=False)
show_colorbar = st.sidebar.checkbox("Show colorbar", value=True)
st.sidebar.markdown("---")

# ROI & extraction defaults
st.sidebar.header("ROI (center & ring)")
default_center_radius = 8
center_radius = st.sidebar.slider("Center radius (pix)", 1, 60, default_center_radius)
ring_inner = st.sidebar.slider("Ring inner radius (pix)", center_radius + 1, 120, default_center_radius + 6)
ring_outer = st.sidebar.slider("Ring outer radius (pix)", ring_inner + 1, 140, ring_inner + 8)

# Create or load data
sci_data = None
header = {}
wavelengths = None
using_real = False

if uploaded is not None and not use_sim:
    st.sidebar.info("Using uploaded FITS cube.")
    try:
        sci_data, header = load_fits_cube(uploaded)
        using_real = True
    except Exception as e:
        st.sidebar.error(f"Failed to load FITS: {e}. Falling back to simulated cube.")
        sci_data, wavelengths = generate_synthetic_cube()
        header = {}
        using_real = False
else:
    sci_data, wavelengths = generate_synthetic_cube()
    header = {}
    using_real = False

# Normalize shape: ensure (nwav, ny, nx)
sci_data = np.squeeze(sci_data)
if sci_data.ndim == 2:
    sci_data = sci_data[None, ...]
nwav, ny, nx = sci_data.shape

# Wavelength vector if missing
if wavelengths is None:
    # guess or make a default vector
    wavelengths = np.linspace(5.0, 12.0, nwav)

# Pixel scale calculation
pix_scale_deg = header_pixel_scale(header) if header else None
if pix_scale_deg is None:
    # fallback placeholder
    pix_scale_deg = 0.00015 * u.deg

pix_scale_arcsec, pix_scale_pc, ang_diam_dist = compute_physical_scale(pix_scale_deg, redshift_z, H0=H0, Om0=Om0)

# Sidebar summary
with st.sidebar.expander("Dataset summary", expanded=True):
    st.write(f"Using: {'Uploaded FITS' if using_real else 'Simulated cube'}")
    st.write(f"Cube shape (nwav, ny, nx): {sci_data.shape}")
    st.write(f"Wavelengths (μm): {wavelengths[0]:.2f} — {wavelengths[-1]:.2f}")
    st.write(f"Pixel scale: {pix_scale_arcsec:.4f} /pix  ≈  {pix_scale_pc:.3f} pc/pix")
    st.write(f"Angular diameter distance: {an
