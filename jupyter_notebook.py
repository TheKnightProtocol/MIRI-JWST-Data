
# 1. Basic Exploration I - Object Identification
Here are the basic properties of NGC 7469 obtained from NED and Simbad:
 **Sky Coordinates (RA and Dec):** 23h 03m 15.61s, +08d 52m 26.0s
**Redshift (z) value:** 0.016335
**Category of the object:** Galaxy
**Sub-category (if available):** Seyfert 1 Galaxy, Starburst Galaxy
**Distance to the object:** 62.1 Mpc
# Store basic object info as variables
object_name = "NGC 7469"
ra = "23h 03m 15.61s"
dec = "+08d 52m 26.0s"
distance_mpc = 62.1
redshift_z = 0.016335
object_category = "Galaxy"
object_subcategory = "Seyfert 1 Galaxy, Starburst Galaxy"

print(f"Object: {object_name}, Redshift: {redshift_z}")
# 2. Read and Summarise
summary_category = """
[cite_start]NGC 7469 is identified as a Seyfert 1 Galaxy and a Starburst Galaxy[cite: 1].
[cite_start]A Seyfert galaxy is a type of active galaxy that has a very bright, point-like nucleus which is a compact source of light[cite: 1].
[cite_start]The Unified Model of AGNs suggests that different types of active galactic nuclei, like Seyfert galaxies, are fundamentally the same but appear different due to varying viewing angles relative to a central supermassive black hole and its surrounding obscuring material (like a torus of dust)[cite: 1].
[cite_start]A Starburst galaxy is a galaxy undergoing an exceptionally high rate of star formation, consuming its gas reserves very quickly[cite: 1].
"""

# 3. Understand the Role of MIR
role_of_mir = """
[cite_start]Mid-Infrared (MIR) imaging is critical for studying objects like NGC 7469 because it helps reveal hidden structures that Optical/NIR (Near-Infrared) cannot[cite: 1].
[cite_start]MIR light can penetrate significant amounts of dust and gas that would obscure visible and near-infrared light, allowing astronomers to see through dusty regions to the active galactic nucleus or intense star formation activity within the galaxy[cite: 1]. [cite_start]This makes it invaluable for studying processes hidden behind dense material, such as those in Seyfert and Starburst galaxies[cite: 1].
"""

print(f"Summary of Category: {summary_category}")
print(f"Role of MIR: {role_of_mir}")
from astropy.io import fits
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import numpy as np
# Replace 'path/to/your/data/' with the actual path to your Data folder
# and 'your_ch1_short_file.fits' with the actual filename
fits_file_path = 'Data/jw01328-c1006_t014_miri_ch1-short_s3d.fits' # Example filename
with fits.open(fits_file_path) as hdul:
    header = hdul['SCI'].header # Access the header of the 'SCI' extension
    print(header) # You can print the whole header to inspect it
pixel_scale_deg = header['CDELT1'] # Or CDELT2, depending on your header
pixel_scale_arcsec = pixel_scale_deg * 3600 * u.arcsec # Convert degrees to arcseconds
print(f"Pixel scale: {pixel_scale_arcsec}")
# Use the redshift you previously found
  redshift_z = 0.016335 # From your Basic Exploration I [cite: 1, 9]

  # Define a cosmology model (standard values, you can research if you need different ones)
  cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

  # Calculate the angular diameter distance
  angular_diameter_distance_mpc = cosmo.angular_diameter_distance(redshift_z).to(u.Mpc)
  print(f"Angular Diameter Distance: {angular_diameter_distance_mpc}")

  # Convert arcseconds to parsecs using the angular diameter distance
  # 1 arcsec corresponds to (angular_diameter_distance_mpc * (1/206265)) in Mpc
  # 1 Mpc = 10^6 parsecs
  pixel_scale_parsec_per_pixel = (pixel_scale_arcsec.to(u.rad) * angular_diameter_distance_mpc).to(u.pc)
  print(f"Pixel scale in parsecs per pixel: {pixel_scale_parsec_per_pixel}")
# Example (conceptual, exact code depends on Session 5 functions)
  # Assuming your Session 5 code has a function like 'extract_spectrum_from_region'
  # from your_session5_script import extract_spectrum_from_region # You might need this

  # List of your FITS files (e.g., from os.listdir('Data/'))
  fits_files = ['Data/jw01328-c1006_t014_miri_ch1-short_s3d.fits', ...]
  region_file = 'Regions/ngc7469_regions.reg'

  all_spectra_data = {} # To store all extracted spectra

  for f_path in fits_files:
      # Assuming the function takes fits file path and region file path
      center_spectrum = extract_spectrum_from_region(f_path, region_file, 'Center Region')
      ring_spectrum = extract_spectrum_from_region(f_path, region_file, 'Ring region')

      # Store or process the spectra (e.g., in a dictionary keyed by filename/channel)
      all_spectra_data[f_path + '_center'] = center_spectrum
      all_spectra_data[f_path + '_ring'] = ring_spectrum
import pandas as pd

  # Assuming center_spectrum and ring_spectrum are arrays/lists of [wavelengths, fluxes]
  # For a single file example:
  df_center = pd.DataFrame({'Wavelength': center_spectrum[0], 'Flux': center_spectrum[1]})
  df_ring = pd.DataFrame({'Wavelength': ring_spectrum[0], 'Flux': ring_spectrum[1]})

  # Save to CSV (optional, but good practice)
  df_center.to_csv('Outputs/center_spectrum_ch1_short.csv', index=False)
  df_ring.to_csv('Outputs/ring_spectrum_ch1_short.csv', index=False)
import matplotlib.pyplot as plt

  plt.figure(figsize=(10, 6))
  plt.plot(df_center['Wavelength'], df_center['Flux'], label='Center Region')
  plt.plot(df_ring['Wavelength'], df_ring['Flux'], label='Ring Region', alpha=0.7)
  plt.xlabel('Wavelength (microns)')
  plt.ylabel('Flux')
  plt.title('Spectra Comparison: Center vs. Ring (Channel 1 Short)')
  plt.legend()
  plt.grid(True)
  plt.show()
