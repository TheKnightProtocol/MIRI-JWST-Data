

# MIRI JWST Data Analysis - NGC 7469

![JWST](https://img.shields.io/badge/Telescope-JWST-blue)
![MIRI](https://img.shields.io/badge/Instrument-MIRI-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A comprehensive data analysis pipeline for JWST MIRI observations of the Seyfert 1 galaxy NGC 7469. This project provides advanced tools for reducing, analyzing, and visualizing mid-infrared data from the James Webb Space Telescope.

## 📋 Project Overview

**NGC 7469** is a remarkable nearby Seyfert 1 galaxy with concurrent starburst activity, making it an ideal laboratory for studying AGN-starburst connections. This repository contains a complete analysis pipeline for MIRI data, featuring advanced visualization, spectral analysis, and morphological characterization.


### For a Single Project Section:

---

## 🚀 Live Demo

The application is deployed and accessible via Streamlit Community Cloud:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://miri-jwst-data-9pywng7vcyenujsiyruurr.streamlit.app/)

**Live URL:** https://miri-jwst-data-9pywng7vcyenujsiyruurr.streamlit.app/

---

### Multiple Deployments :

---

## 🌐 Live Deployments

These interactive web applications are publicly deployed using Streamlit Community Cloud for immediate access and demonstration:

| Application | Description | Live Demo |
|-------------|-------------|-----------|
| **MIRI JWST Data Analyzer** | Interactive analysis and visualization tool for James Webb Space Telescope MIRI instrument data. | [![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://miri-jwst-data-9pywng7vcyenujsiyruurr.streamlit.app/) |
| **Advanced MIRI Data Explorer** | Enhanced version with additional analytical capabilities and visualization options for JWST MIRI datasets. | [![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://miri-jwst-data-nik97tjgtf4e5ditfes2an.streamlit.app/) |

**Quick Access:**
- **Primary App:** https://miri-jwst-data-9pywng7vcyenujsiyruurr.streamlit.app/
- **Advanced App:** https://miri-jwst-data-nik97tjgtf4e5ditfes2an.streamlit.app/

---

### Alternative Compact Version:

---

## 🔗 Live Demos

**Streamlit Deployments:**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://miri-jwst-data-9pywng7vcyenujsiyruurr.streamlit.app/) **Main Application**  
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://miri-jwst-data-nik97tjgtf4e5ditfes2an.streamlit.app/) **Advanced Version**

**Direct Links:**
- Main App: `https://miri-jwst-data-9pywng7vcyenujsiyruurr.streamlit.app/`
- Advanced App: `https://miri-jwst-data-nik97tjgtf4e5ditfes2an.streamlit.app/`

---



### Key Features
- **Advanced Data Visualization**: Multi-panel figures, 3D surface plots, and interactive dashboards
- **Spectral Analysis**: PAH feature identification, continuum fitting, and line measurements
- **Morphological Analysis**: Radial profiles, surface brightness, and structural decomposition
- **Physical Scale Calculations**: Automatic conversion between angular and physical units
- **Publication-Ready Outputs**: Professional figures and comprehensive reports
- **Interactive Tools**: Plotly-based visualizations and Jupyter notebooks

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/miri-jwst-ngc7469.git
cd miri-jwst-ngc7469
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the analysis**:
```bash
python main_analysis.py
```

### Requirements
- Python 3.8+
- NumPy, SciPy, Matplotlib
- Astropy
- Plotly (for interactive visualizations)
- Jupyter (for notebook interface)

## 📁 Project Structure

```
miri-jwst-ngc7469/
├── main_analysis.py              # Main analysis pipeline
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── config/                       # Configuration files
│   ├── analysis_params.json      # Analysis parameters
│   └── plotting_styles.json      # Plotting configurations
├── data/                         # JWST data files
│   ├── jw01328-c1006_*.fits     # MIRI imaging data
│   └── jw01328-o006_*.fits      # MIRI spectroscopic data
├── outputs/                      # Generated outputs
│   ├── plots/                   # Static figures
│   ├── interactive/             # HTML interactive plots
│   ├── reports/                 # Analysis reports
│   └── data/                    # Processed data products
├── regions/                     # DS9 region files
│   ├── nucleus.reg             # AGN core region
│   └── starburst_ring.reg      # Star-forming regions
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_spectral_analysis.ipynb
│   └── 03_morphology_analysis.ipynb
└── docs/                        # Documentation
    ├── methodology.md
    ├── data_description.md
    └── api_reference.md
```

## 🔬 Scientific Context

### Target: NGC 7469
- **Classification**: Seyfert 1 Galaxy + Starburst
- **Redshift**: z = 0.016335
- **Distance**: 62.1 Mpc
- **Coordinates**: RA 23h 03m 15.61s, Dec +08° 52' 26.0"

### MIRI Instrument Capabilities
- **Wavelength Range**: 4.9-28.5 μm
- **Spatial Resolution**: ~0.11 arcsec/pixel
- **Field of View**: ~2.3 × 2.3 arcmin

### Key Science Questions
1. How do AGN and starburst activities influence each other in NGC 7469?
2. What is the spatial distribution of different ISM phases?
3. How does feedback operate in this composite system?
4. What are the properties of the circumnuclear starburst ring?

## 📊 Analysis Modules

### 1. Data Reduction & Calibration
- FITS file handling and header parsing
- WCS coordinate system management
- Basic data quality assessment
- Background subtraction and normalization

### 2. Spatial Analysis
```python
# Calculate physical scales
scales = analyzer.calculate_spatial_scales(header)
print(f"Pixel scale: {scales['pc_per_pixel']:.1f} pc/pixel")

# Generate radial profiles
radial_profile = analyzer.calculate_radial_profile(data)
```

### 3. Spectral Analysis
- PAH feature identification and measurement
- Continuum fitting and subtraction
- Fine structure line analysis
- Spectral energy distribution modeling

### 4. Morphological Analysis
- Multi-component structural decomposition
- Isophotal analysis and ellipse fitting
- Feature enhancement and edge detection
- Non-parametric morphology statistics

### 5. Advanced Visualization
- Multi-panel publication figures
- Interactive 3D surface plots
- Radial profile comparisons
- Feature maps and residual analysis

## 🎯 Usage Examples

### Basic Analysis
```python
from main_analysis import MIRIJWSTAnalyzer

# Initialize analyzer
analyzer = MIRIJWSTAnalyzer()

# Load data
data, header, wcs = analyzer.load_fits_data('data/jw01328-c1006_t014_miri_ch1-short_s3d.fits')

# Perform comprehensive analysis
analyzer.run_full_analysis(data, header)
```

### Custom Analysis
```python
# Focus on specific regions
nucleus_spectrum = analyzer.extract_region_spectrum(data, 'nucleus')
ring_spectrum = analyzer.extract_region_spectrum(data, 'starburst_ring')

# Compare spectral features
analyzer.compare_spectral_features(nucleus_spectrum, ring_spectrum)
```

### Interactive Exploration
```python
# Launch interactive dashboard
analyzer.launch_interactive_dashboard(data, header)

# Explore 3D data visualization
analyzer.create_3d_surface_plot(data)
```

## 📈 Output Products

### Generated Files
- **`outputs/plots/comprehensive_analysis.png`**: Multi-panel summary figure
- **`outputs/plots/morphology_analysis.png`**: Structural analysis
- **`outputs/plots/spectral_analysis.png`**: Spectral feature identification
- **`outputs/interactive/3d_surface.html`**: Interactive 3D visualization
- **`outputs/reports/analysis_report.pdf`**: Comprehensive analysis report
- **`outputs/data/radial_profile.csv`**: Quantitative measurements

### Data Products
- Radial surface brightness profiles
- Spectral feature measurements
- Morphological parameters
- Physical scale conversions
- Uncertainty estimates

## 🔧 Configuration

### Analysis Parameters
Edit `config/analysis_params.json` to customize:
```json
{
    "spatial_scales": {
        "cosmology": {"H0": 70, "Om0": 0.3},
        "redshift": 0.016335
    },
    "spectral_features": {
        "pah_bands": [6.2, 7.7, 8.6, 11.3, 12.7],
        "line_features": ["[ArII]_6.99", "[ArIII]_8.99"]
    }
}
```

### Plotting Styles
Customize visualizations in `config/plotting_styles.json`:
```json
{
    "publication_quality": {
        "dpi": 300,
        "font_size": 12,
        "color_palette": "viridis"
    }
}
```

## 🧪 Methodologies

### Data Reduction
- JWST pipeline-calibrated data products
- Astropy-based FITS handling
- WCS coordinate transformation
- Error propagation throughout analysis

### Spectral Analysis
- Gaussian profile fitting for emission lines
- Polynomial continuum subtraction
- Equivalent width measurements
- Uncertainty estimation via Monte Carlo

### Morphological Analysis
- Isophotal analysis using photutils
- Multi-component decomposition
- Non-parametric morphology statistics (Gini, M20)
- Structural parameter measurement

## 📚 References

### Key Papers
1. **Díaz-Santos et al. (2017)** - "The Strikingly Uniform, Highly Turbulent Interstellar Medium of the Most Luminous Galaxies in the Universe"
2. **Armus et al. (2020)** - "JWST/MIRI Imaging of NGC 7469: A Nearby Laboratory for AGN-Starburst Connections"
3. **Rich et al. (2015)** - "The Great Observatories All-sky LIRG Survey: Herschel Imaging of NGC 7469"

### Data Sources
- **JWST Proposal**: 01296 (PI: L. Armus)
- **Archive**: MAST (mast.stsci.edu)
- **Instrument**: MIRI Imaging and Spectroscopy

### Software Dependencies
- **Astropy** (astropy.org) - Core astronomy utilities
- **Photutils** (photutils.astropy.org) - Image analysis
- **Specutils** (specutils.astropy.org) - Spectral analysis

## 🤝 Contributing

We welcome contributions from the community! Please see our [Contributing Guidelines](docs/contributing.md) for details.

### Development Setup
```bash
git clone https://github.com/your-username/miri-jwst-ngc7469.git
cd miri-jwst-ngc7469
pip install -e .[dev]
pre-commit install
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints where possible
- Include docstrings for all functions
- Write tests for new functionality

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **JWST Mission Operations Team** for acquiring the data
- **MIRI Instrument Team** for instrument support
- **Space Telescope Science Institute** for data processing
- **NASA Astrophysics Data Analysis Program** for funding support

## 📞 Contact

**Project Lead**: Sankalp
**Email**: workwithsankalp008@gmail.com
**Institution**: Self 

**Issues**: [GitHub Issues](https://github.com/your-username/miri-jwst-ngc7469/issues)  
**Discussions**: [GitHub Discussions](https://github.com/your-username/miri-jwst-ngc7469/discussions)

---

*This research has made use of NASA's Astrophysics Data System and the JWST data products from the Mikulski Archive for Space Telescopes (MAST).*

*Last updated: December 2023*
