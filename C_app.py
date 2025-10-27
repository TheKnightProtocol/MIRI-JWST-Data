# ---------------------------------------------------------------
# MIRI JWST Data — Analysis of NGC 7469 (Simulated / Real-Data Friendly)
# Fixed: correct angular -> physical size conversion using astropy equivalencies
# Author: Sankalp Sharma (updated)
# ---------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

sns.set_style("whitegrid")

# 1. Basic object metadata
object_name = "NGC 7469"
ra = "23h 03m 15.61s"
dec = "+08d 52m 26.0s"
distance_mpc = 62.1
redshift_z = 0.016335
object_category = "Galaxy"
object_subcategory = "Seyfert 1 Galaxy, Starburst Galaxy"

print(f"Analyzing {object_name} ({object_subcategory}) — z={redshift_z}")

# 2. Role of Mid-Infrared Imaging
print("""
Mid-Infrared (MIR) observations (e.g., from MIRI on JWST) allow dust-enshrouded
star-forming regions and AGN-heated dust to be probed. They can penetrate regions
that optical/NIR cannot.
""")

# 3. Cosmological scaling
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
ang_diam_dist = cosmo.angular_diameter_distance(redshift_z).to(u.Mpc)
print(f"Angular Diameter Distance: {ang_diam_dist:.4g}")

# 4. Provide paths and simulation fallback
fits_path = "Data/jw01328-c1006_t014_miri_ch1-short_s3d.fits"

if os.path.exists(fits_path):
    # Real-data branch
    with fits.open(fits_path) as hdul:
        # try common SCI extension names; fallback to 0th
        if "SCI" in hdul:
            sci_hdu = hdul["SCI"]
        else:
            sci_hdu = hdul[0]
        sci_data = np.array(sci_hdu.data)
        header = sci_hdu.header

    # Extract pixel scale: assume CDELT1 is in degrees/pixel (common)
    # Use astropy units explicitly
    cdelt1 = header.get("CDELT1", None)
    if cdelt1 is None:
        # try alternative header keys (CDELT1 might be absent for some cubes)
        cdelt1 = header.get("CD1_1", None)
    if cdelt1 is None:
        raise KeyError("No CDELT1 or CD1_1 found in FITS header. Cannot deduce pixel scale.")

    pixel_scale_deg = float(cdelt1) * u.deg
    pixel_scale_arcsec = pixel_scale_deg.to(u.arcsec)
    theta_rad = pixel_scale_arcsec.to(u.rad)

    # Correct conversion: linear size = angular diameter distance * angle (radians)
    # Use equivalencies because we're converting an angle*distance quantity to a length unit
    pixel_scale_pc = (ang_diam_dist * theta_rad).to(u.pc, equivalencies=u.dimensionless_angles())

    print(f"Pixel scale: {pixel_scale_arcsec:.4g} per pixel ≈ {pixel_scale_pc:.4g} per pixel")

else:
    # Simulation branch (no FITS present)
    print("** Note: FITS file not found. Using synthetic data for demonstration. **")
    # Simulate a 3-D cube: shape [wavelength_slice, y, x]
    nx, ny = 150, 150
    nwav = 30
    wavelengths = np.linspace(5.0, 12.0, nwav)  # microns

    # create synthetic spatial cube: gaussian ring + central peak; wavelength-dependent modulations
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    ring = np.exp(-((R - 0.6) / 0.2) ** 2)
    center = np.exp(-((R) / 0.2) ** 2)

    sci_data = np.zeros((nwav, ny, nx))
    for i, w in enumerate(wavelengths):
        sci_data[i] = (
            center * (1 + 0.5 * np.sin(2 * np.pi * (w - 5) / 7))
            + 0.5 * ring * (1 + 0.3 * np.cos(2 * np.pi * (w - 5) / 3))
            + 0.08 * np.random.randn(ny, nx)
        )

    # Provide a realistic-ish placeholder pixel scale (deg/pix)
    header = {"CDELT1": 0.00015}  # degrees per pixel (placeholder)
    pixel_scale_deg = float(header["CDELT1"]) * u.deg
    pixel_scale_arcsec = pixel_scale_deg.to(u.arcsec)
    theta_rad = pixel_scale_arcsec.to(u.rad)
    pixel_scale_pc = (ang_diam_dist * theta_rad).to(u.pc, equivalencies=u.dimensionless_angles())

    print(f"(Simulated) Pixel scale: {pixel_scale_arcsec:.4g} per pixel ≈ {pixel_scale_pc:.4g} per pixel")

# 5. Image inspection (choose a mid-wavelength slice)
if isinstance(sci_data, np.ndarray) and sci_data.ndim == 3:
    mid_idx = sci_data.shape[0] // 2
    image_slice = sci_data[mid_idx]
    plt.figure(figsize=(7, 6))
    plt.imshow(image_slice, origin="lower", cmap="inferno", interpolation="nearest")
    plt.colorbar(label="Flux (arb units)")
    plt.title(f"{object_name} – MIRI MIR Slice (index {mid_idx})")
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")
    plt.tight_layout()
    plt.show()

    # Flux histogram
    plt.figure(figsize=(7, 5))
    sns.histplot(image_slice.flatten(), bins=40, kde=True)
    plt.xlabel("Flux (arb units)")
    plt.ylabel("Pixel count")
    plt.title("Flux distribution (middle wavelength slice)")
    plt.tight_layout()
    plt.show()
else:
    print("sci_data does not look like a 3D cube. Skipping image inspection.")

# 6. Spectral extraction simulation (or placeholder for real extraction)
# If you have real region-extraction code, replace the simulated extraction below.
if "wavelengths" not in locals():
    wavelengths = np.linspace(5.0, 12.0, 500)

center_flux = np.exp(-0.5 * ((wavelengths - 7.0) / 0.4) ** 2) + 0.2 * np.random.rand(len(wavelengths))
ring_flux = 0.8 * np.exp(-0.5 * ((wavelengths - 7.3) / 0.6) ** 2) + 0.15 * np.random.rand(len(wavelengths))

df_center = pd.DataFrame({"Wavelength (µm)": wavelengths, "Flux": center_flux})
df_ring = pd.DataFrame({"Wavelength (µm)": wavelengths, "Flux": ring_flux})

# 7. Spectra comparison
plt.figure(figsize=(10, 6))
plt.plot(df_center["Wavelength (µm)"], df_center["Flux"], label="Center Region", linewidth=1.3)
plt.plot(df_ring["Wavelength (µm)"], df_ring["Flux"], label="Ring Region", alpha=0.8)
plt.xlabel("Wavelength (µm)")
plt.ylabel("Flux (arb units)")
plt.title(f"Spectrum Comparison — {object_name}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 8. Flux Ratio curve
df_fluxratio = df_center.copy()
df_fluxratio["Flux_Ratio_Center_to_Ring"] = df_center["Flux"] / (df_ring["Flux"] + 1e-9)

plt.figure(figsize=(10, 5))
plt.plot(df_fluxratio["Wavelength (µm)"], df_fluxratio["Flux_Ratio_Center_to_Ring"], color="crimson")
plt.axhline(1.0, linestyle="--", color="gray", label="Ratio = 1")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Flux Ratio (Center/Ring)")
plt.title("Center-to-Ring Flux Ratio vs Wavelength")
plt.legend()
plt.tight_layout()
plt.show()

# 9. Heatmap of average flux map (spatial average over wavelengths)
if isinstance(sci_data, np.ndarray) and sci_data.ndim == 3:
    avg_flux_map = np.mean(sci_data, axis=0)
    plt.figure(figsize=(8, 7))
    sns.heatmap(avg_flux_map, cmap="magma", cbar_kws={"label": "Average Flux (arb units)"})
    plt.title(f"Average Flux Map — {object_name}")
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")
    plt.tight_layout()
    plt.show()

# 10. Save outputs
os.makedirs("Outputs", exist_ok=True)
df_center.to_csv("Outputs/center_spectrum.csv", index=False)
df_ring.to_csv("Outputs/ring_spectrum.csv", index=False)
print("✅ Analysis complete. CSV outputs saved in 'Outputs/'")

# 11. Notes for real-data usage
print("""
Notes:
- The correct formula for linear size per pixel is:
    linear_size = angular_diameter_distance * angle_in_radians
  and we convert using `equivalencies=u.dimensionless_angles()` so astropy
  knows an angle*distance is a length.
- If you run with real FITS data, inspect header keys: CDELT1, CD1_1, CRPIX*, WCS info.
- For real spectral extraction replace the simulated extraction with aperture/region code.
""")
