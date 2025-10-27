# =======================================================================
# JWST MIRI Data Analysis — NGC 7469 (Real + Simulated Compatible)
# Author: Sankalp Sharma
# Version: 2.0 (Enhanced & Stable)
# =======================================================================
# Description:
# This script performs a full mid-infrared (MIR) imaging and spectral
# analysis workflow for the Seyfert 1 galaxy NGC 7469, using either:
#   (a) a real JWST FITS cube (if available), or
#   (b) a simulated synthetic data cube (fallback mode).
#
# Features:
# - Auto-switch between real and synthetic data
# - Cosmology-aware scale conversion (deg→arcsec→radians→pc)
# - Visualization of slices, histograms, spectra, and flux maps
# - Simulated spectral extraction (center vs. ring)
# - Clean data exports and high-quality plots
# =======================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# ---------------------------------------------------------------
# 1. Configuration & Metadata
# ---------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11
})

object_name = "NGC 7469"
object_subcategory = "Seyfert 1 Galaxy, Starburst Galaxy"
redshift_z = 0.016335
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

print("=" * 70)
print(f"Analyzing {object_name} — {object_subcategory}  (z = {redshift_z})")
print("=" * 70)

# Angular diameter distance
ang_diam_dist = cosmo.angular_diameter_distance(redshift_z).to(u.Mpc)
print(f"Angular Diameter Distance: {ang_diam_dist:.4g}")

# ---------------------------------------------------------------
# 2. FITS Data Input or Synthetic Simulation
# ---------------------------------------------------------------
fits_path = "Data/jw01328-c1006_t014_miri_ch1-short_s3d.fits"

if os.path.exists(fits_path):
    print("\n✅ Real FITS file found. Loading data...")
    with fits.open(fits_path) as hdul:
        if "SCI" in hdul:
            sci_hdu = hdul["SCI"]
        else:
            sci_hdu = hdul[0]
        sci_data = np.array(sci_hdu.data)
        header = sci_hdu.header

    # Extract pixel scale (deg → arcsec)
    cdelt1 = header.get("CDELT1") or header.get("CD1_1")
    if cdelt1 is None:
        raise KeyError("CDELT1 or CD1_1 not found in header.")
    pixel_scale_deg = float(cdelt1) * u.deg
    pixel_scale_arcsec = pixel_scale_deg.to(u.arcsec)
    theta_rad = pixel_scale_arcsec.to(u.rad)

    # Convert angular → physical scale
    pixel_scale_pc = (ang_diam_dist * theta_rad).to(u.pc, equivalencies=u.dimensionless_angles())
    print(f"Pixel Scale: {pixel_scale_arcsec:.4g} per pixel ≈ {pixel_scale_pc:.4g} per pixel")

else:
    print("\n⚠️  No FITS file found. Generating synthetic cube for demonstration.")
    nx, ny, nwav = 150, 150, 30
    wavelengths = np.linspace(5.0, 12.0, nwav)  # microns

    x, y = np.linspace(-1, 1, nx), np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    # Simulate ring + center + wavelength-dependent patterns
    ring = np.exp(-((R - 0.6) / 0.2) ** 2)
    center = np.exp(-((R) / 0.25) ** 2)
    sci_data = np.zeros((nwav, ny, nx))
    for i, w in enumerate(wavelengths):
        sci_data[i] = (
            center * (1 + 0.4 * np.sin(2 * np.pi * (w - 5) / 7))
            + 0.6 * ring * (1 + 0.3 * np.cos(2 * np.pi * (w - 5) / 3))
            + 0.05 * np.random.randn(ny, nx)
        )

    # Synthetic scale
    pixel_scale_deg = 0.00015 * u.deg
    pixel_scale_arcsec = pixel_scale_deg.to(u.arcsec)
    theta_rad = pixel_scale_arcsec.to(u.rad)
    pixel_scale_pc = (ang_diam_dist * theta_rad).to(u.pc, equivalencies=u.dimensionless_angles())

    print(f"(Simulated) Pixel Scale: {pixel_scale_arcsec:.4g} per pixel ≈ {pixel_scale_pc:.4g} per pixel")

# ---------------------------------------------------------------
# 3. Visualization: Image Slice + Histogram
# ---------------------------------------------------------------
mid_idx = sci_data.shape[0] // 2
image_slice = sci_data[mid_idx]

plt.figure(figsize=(7, 6))
plt.imshow(image_slice, origin="lower", cmap="inferno")
plt.colorbar(label="Flux (arb. units)")
plt.title(f"{object_name} — MIR Slice #{mid_idx}")
plt.xlabel("X pixel")
plt.ylabel("Y pixel")
plt.tight_layout()
os.makedirs("Outputs", exist_ok=True)
plt.savefig("Outputs/MIRI_slice.png", dpi=200)
plt.show()

plt.figure(figsize=(7, 5))
sns.histplot(image_slice.flatten(), bins=40, kde=True, color="orange")
plt.xlabel("Flux (arb. units)")
plt.ylabel("Pixel count")
plt.title("Flux Distribution — Middle Wavelength Slice")
plt.tight_layout()
plt.savefig("Outputs/Flux_histogram.png", dpi=200)
plt.show()

# ---------------------------------------------------------------
# 4. Simulated Spectral Extraction (Center vs Ring)
# ---------------------------------------------------------------
if "wavelengths" not in locals():
    wavelengths = np.linspace(5.0, 12.0, 500)

center_flux = np.exp(-0.5 * ((wavelengths - 7.0) / 0.4) ** 2) + 0.2 * np.random.rand(len(wavelengths))
ring_flux = 0.8 * np.exp(-0.5 * ((wavelengths - 7.3) / 0.6) ** 2) + 0.15 * np.random.rand(len(wavelengths))

df_center = pd.DataFrame({"Wavelength (µm)": wavelengths, "Flux": center_flux})
df_ring = pd.DataFrame({"Wavelength (µm)": wavelengths, "Flux": ring_flux})

# ---------------------------------------------------------------
# 5. Spectrum Comparison Plot
# ---------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(df_center["Wavelength (µm)"], df_center["Flux"], label="Center Region", color="gold", lw=2)
plt.plot(df_ring["Wavelength (µm)"], df_ring["Flux"], label="Ring Region", color="deepskyblue", lw=2)
plt.xlabel("Wavelength (µm)")
plt.ylabel("Flux (arb. units)")
plt.title(f"Spectrum Comparison — {object_name}")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Outputs/Spectrum_Comparison.png", dpi=200)
plt.show()

# ---------------------------------------------------------------
# 6. Flux Ratio Analysis
# ---------------------------------------------------------------
df_fluxratio = df_center.copy()
df_fluxratio["Flux_Ratio_Center_to_Ring"] = df_center["Flux"] / (df_ring["Flux"] + 1e-8)

plt.figure(figsize=(10, 5))
plt.plot(df_fluxratio["Wavelength (µm)"], df_fluxratio["Flux_Ratio_Center_to_Ring"], color="crimson", lw=2)
plt.axhline(1.0, linestyle="--", color="gray", label="Ratio = 1")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Flux Ratio (Center / Ring)")
plt.title("Center-to-Ring Flux Ratio vs Wavelength")
plt.legend()
plt.tight_layout()
plt.savefig("Outputs/Flux_Ratio.png", dpi=200)
plt.show()

# ---------------------------------------------------------------
# 7. Average Flux Map (Over Wavelength)
# ---------------------------------------------------------------
avg_flux_map = np.mean(sci_data, axis=0)
plt.figure(figsize=(8, 7))
sns.heatmap(avg_flux_map, cmap="magma", cbar_kws={'label': 'Average Flux (arb. units)'})
plt.title(f"Average Flux Map — {object_name}")
plt.xlabel("X pixel")
plt.ylabel("Y pixel")
plt.tight_layout()
plt.savefig("Outputs/Average_Flux_Map.png", dpi=200)
plt.show()

# ---------------------------------------------------------------
# 8. Save Outputs
# ---------------------------------------------------------------
df_center.to_csv("Outputs/center_spectrum.csv", index=False)
df_ring.to_csv("Outputs/ring_spectrum.csv", index=False)
df_fluxratio.to_csv("Outputs/flux_ratio.csv", index=False)
print("\n✅ Analysis complete. All CSV and PNG outputs saved in 'Outputs/'.")

# ---------------------------------------------------------------
# 9. Real-Data Instructions
# ---------------------------------------------------------------
print("""
📘 Notes for Real JWST FITS Cubes:
----------------------------------
If you wish to use real MIRI data from JWST (e.g., NGC 7469):
1. Visit the MAST archive: https://archive.stsci.edu/hlsp/goals
2. Download a cube, e.g.:
   HLSP_GOALS_JWST_MIRI-MRS_NGC7469_ch1-short_s3d.fits
3. Place it in the `Data/` directory and rename as:
   jw01328-c1006_t014_miri_ch1-short_s3d.fits
4. The script will automatically detect and analyze it.

📦 Outputs generated:
- Outputs/center_spectrum.csv
- Outputs/ring_spectrum.csv
- Outputs/flux_ratio.csv
- PNG plots of spectra and flux maps
""")

print("=" * 70)
print("🪐 JWST MIR Analysis Script Completed — Ready for GitHub Publication.")
print("=" * 70)
