# MIRI JWST Data Analysis - NGC 7469
# Enhanced Version with Advanced Visualizations and Error Handling

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import signal
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import astropy, but provide fallbacks if not available
try:
    from astropy.io import fits
    from astropy.wcs import WCS
    import astropy.units as u
    from astropy.cosmology import FlatLambdaCDM
    ASTROPY_AVAILABLE = True
except ImportError:
    print("Astropy not available. Using simulated data and basic calculations.")
    ASTROPY_AVAILABLE = False

# Try to import plotly for interactive plots
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Plotly not available. Using matplotlib only.")
    PLOTLY_AVAILABLE = False

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
        
        # Create output directory
        os.makedirs('Outputs', exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
    def create_simulated_data(self, shape=(200, 200)):
        """Create realistic simulated MIRI data for demonstration"""
        print("Creating simulated MIRI data...")
        y, x = np.indices(shape)
        center_y, center_x = shape[0]//2, shape[1]//2
        
        # Create a realistic galaxy profile
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Bright AGN core
        agn_core = 10 * np.exp(-(r**2) / (2 * 5**2))
        
        # Spiral arm structure
        theta = np.arctan2(y - center_y, x - center_x)
        spiral_arms = 2 * np.exp(-r/50) * (1 + 0.5 * np.sin(2 * theta + 0.5 * np.log(1 + r/10)))
        
        # Star-forming regions (random clumps)
        rng = np.random.RandomState(42)
        clumps = np.zeros(shape)
        for _ in range(20):
            cy = rng.randint(20, shape[0]-20)
            cx = rng.randint(20, shape[1]-20)
            clump_size = rng.uniform(3, 8)
            clump_amp = rng.uniform(0.5, 2)
            clumps += clump_amp * np.exp(-((x-cx)**2 + (y-cy)**2) / (2 * clump_size**2))
        
        # Background noise
        noise = 0.1 * rng.normal(size=shape)
        
        # Combine all components
        simulated_data = agn_core + spiral_arms + clumps + noise
        
        # Create a mock header
        mock_header = {
            'CDELT1': 0.0001,
            'CDELT2': 0.0001,
            'CRPIX1': center_x,
            'CRPIX2': center_y,
            'CRVAL1': 345.8150417,  # RA in degrees
            'CRVAL2': 8.8738889,    # Dec in degrees
        }
        
        return simulated_data, mock_header
    
    def load_fits_data(self, file_path):
        """Load and process FITS file data with fallback to simulated data"""
        if ASTROPY_AVAILABLE and os.path.exists(file_path):
            try:
                with fits.open(file_path) as hdul:
                    data = hdul['SCI'].data
                    header = hdul['SCI'].header
                    wcs = WCS(header)
                print(f"✓ Successfully loaded FITS data from {file_path}")
                return data, header, wcs
            except Exception as e:
                print(f"⚠ Error loading FITS file: {e}. Using simulated data.")
                return self.create_simulated_data()
        else:
            print("⚠ Using simulated data (FITS file not available or astropy not installed)")
            data, header = self.create_simulated_data()
            return data, header, None
    
    def calculate_spatial_scales(self, header):
        """Calculate spatial scales in various units"""
        try:
            pixel_scale_deg = header.get('CDELT1', 0.0001)
            pixel_scale_arcsec = pixel_scale_deg * 3600
            
            if ASTROPY_AVAILABLE:
                # Cosmology calculations
                cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
                angular_diameter_distance = cosmo.angular_diameter_distance(self.object_info['redshift'])
                
                # Convert to physical scales
                scale_pc_per_pixel = (pixel_scale_arcsec * u.arcsec).to(u.rad).value * angular_diameter_distance.to(u.pc).value
            else:
                # Manual calculation approximation
                scale_pc_per_pixel = pixel_scale_arcsec * self.object_info['distance_mpc'] * 1000 / 206265
            
            return {
                'arcsec_per_pixel': pixel_scale_arcsec,
                'pc_per_pixel': scale_pc_per_pixel,
                'kpc_per_pixel': scale_pc_per_pixel / 1000
            }
        except Exception as e:
            print(f"⚠ Error calculating spatial scales: {e}. Using default values.")
            return {
                'arcsec_per_pixel': 0.1,
                'pc_per_pixel': 30.0,
                'kpc_per_pixel': 0.03
            }
    
    def create_enhanced_visualizations(self, data, scales):
        """Create comprehensive visualizations"""
        print("Creating enhanced visualizations...")
        self.create_main_figure(data, scales)
        self.create_radial_profile(data, scales)
        self.create_spectral_analysis_plots()
        
        if PLOTLY_AVAILABLE:
            self.create_3d_visualization(data)
        else:
            self.create_3d_matplotlib(data)
    
    def create_main_figure(self, data, scales):
        """Create the main analysis figure with multiple subplots"""
        print("Creating main analysis figure...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'MIRI JWST Analysis: {self.object_info["name"]}', 
                    fontsize=16, fontweight='bold', color='navy')
        
        # Original data - Viridis
        im1 = axes[0,0].imshow(data, cmap='viridis', origin='lower', aspect='auto')
        axes[0,0].set_title('Original Data - Viridis', fontweight='bold', fontsize=12)
        axes[0,0].set_xlabel('Pixel X')
        axes[0,0].set_ylabel('Pixel Y')
        plt.colorbar(im1, ax=axes[0,0], label='Flux [MJy/sr]')
        
        # Log scale - Plasma
        log_data = np.log10(data + 1e-10)
        im2 = axes[0,1].imshow(log_data, cmap='plasma', origin='lower', aspect='auto')
        axes[0,1].set_title('Log Scale - Plasma', fontweight='bold', fontsize=12)
        axes[0,1].set_xlabel('Pixel X')
        axes[0,1].set_ylabel('Pixel Y')
        plt.colorbar(im2, ax=axes[0,1], label='log(Flux)')
        
        # Heatmap with contours - Inferno
        im3 = axes[0,2].imshow(data, cmap='inferno', origin='lower', aspect='auto')
        contours = axes[0,2].contour(data, levels=8, colors='white', alpha=0.7, linewidths=0.5)
        axes[0,2].clabel(contours, inline=True, fontsize=8)
        axes[0,2].set_title('Inferno + Contours', fontweight='bold', fontsize=12)
        axes[0,2].set_xlabel('Pixel X')
        axes[0,2].set_ylabel('Pixel Y')
        plt.colorbar(im3, ax=axes[0,2], label='Flux [MJy/sr]')
        
        # Histogram of flux values
        axes[1,0].hist(data.flatten(), bins=50, color=self.colors[0], alpha=0.7, 
                      edgecolor='black', linewidth=0.5)
        axes[1,0].set_xlabel('Flux Value [MJy/sr]')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Flux Distribution', fontweight='bold', fontsize=12)
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_yscale('log')
        
        # Spatial scales visualization
        scales_text = f"""Spatial Scales:
        • {scales['arcsec_per_pixel']:.4f} arcsec/pixel
        • {scales['pc_per_pixel']:.2f} pc/pixel
        • {scales['kpc_per_pixel']:.4f} kpc/pixel
        
        Distance: {self.object_info['distance_mpc']} Mpc
        Redshift: z = {self.object_info['redshift']}"""
        
        axes[1,1].text(0.1, 0.5, scales_text, transform=axes[1,1].transAxes, 
                      fontsize=10, verticalalignment='center', fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1,1].set_title('Spatial Information', fontweight='bold', fontsize=12)
        axes[1,1].set_xticks([])
        axes[1,1].set_yticks([])
        
        # Object information
        info_text = f"""Object: {self.object_info['name']}
        Category: {self.object_info['category']}
        Type: {self.object_info['subcategory']}
        
        Coordinates:
        RA: {self.object_info['ra']}
        Dec: {self.object_info['dec']}"""
        
        axes[1,2].text(0.1, 0.5, info_text, transform=axes[1,2].transAxes,
                      fontsize=10, verticalalignment='center', fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        axes[1,2].set_title('Object Properties', fontweight='bold', fontsize=12)
        axes[1,2].set_xticks([])
        axes[1,2].set_yticks([])
        
        plt.tight_layout()
        plt.savefig('Outputs/ngc7469_comprehensive_analysis.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
    
    def create_3d_visualization(self, data):
        """Create interactive 3D visualization using Plotly"""
        if not PLOTLY_AVAILABLE:
            return
            
        print("Creating 3D visualization...")
        x, y = np.mgrid[0:data.shape[1], 0:data.shape[0]]
        
        fig = go.Figure(data=[go.Surface(z=data, x=x, y=y, colorscale='Viridis')])
        
        fig.update_layout(
            title=f'3D Surface Plot: {self.object_info["name"]} MIRI Data',
            scene=dict(
                xaxis_title='Pixel X',
                yaxis_title='Pixel Y',
                zaxis_title='Flux [MJy/sr]',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=800,
            height=600
        )
        
        fig.write_html('Outputs/3d_surface_plot.html')
        fig.show()
    
    def create_3d_matplotlib(self, data):
        """Create 3D visualization using matplotlib (fallback)"""
        print("Creating 3D matplotlib visualization...")
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Downsample for performance
        step = max(1, data.shape[0] // 50)
        x, y = np.mgrid[0:data.shape[1]:step, 0:data.shape[0]:step]
        z = data[::step, ::step]
        
        surf = ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)
        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')
        ax.set_zlabel('Flux [MJy/sr]')
        ax.set_title(f'3D View: {self.object_info["name"]}')
        
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        plt.savefig('Outputs/3d_surface_matplotlib.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_radial_profile(self, data, scales):
        """Calculate and plot radial profile"""
        print("Creating radial profile analysis...")
        
        center_y, center_x = np.array(data.shape) // 2
        y, x = np.indices(data.shape)
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        r = r.astype(int)
        
        radial_profile = np.bincount(r.ravel(), data.ravel()) / np.bincount(r.ravel())
        
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        
        # Radial profile in pixels
        ax[0].plot(radial_profile, color=self.colors[1], linewidth=2)
        ax[0].set_xlabel('Radius (pixels)')
        ax[0].set_ylabel('Average Flux [MJy/sr]')
        ax[0].set_title('Radial Profile (Pixels)')
        ax[0].grid(True, alpha=0.3)
        ax[0].fill_between(range(len(radial_profile)), radial_profile, alpha=0.3, color=self.colors[1])
        
        # Radial profile in parsecs
        radius_pc = np.arange(len(radial_profile)) * scales['pc_per_pixel']
        ax[1].plot(radius_pc, radial_profile, color=self.colors[2], linewidth=2)
        ax[1].set_xlabel('Radius (parsecs)')
        ax[1].set_ylabel('Average Flux [MJy/sr]')
        ax[1].set_title('Radial Profile (Physical Scale)')
        ax[1].grid(True, alpha=0.3)
        ax[1].fill_between(radius_pc, radial_profile, alpha=0.3, color=self.colors[2])
        
        plt.tight_layout()
        plt.savefig('Outputs/radial_profile.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_spectral_analysis_plots(self):
        """Create simulated spectral analysis plots"""
        print("Creating spectral analysis plots...")
        
        # Simulate spectral data for demonstration
        wavelength = np.linspace(5, 28, 1000)
        
        # Simulate different spectral features
        continuum = 1e-15 * wavelength**-2
        pah_features = (0.5e-15 * np.exp(-(wavelength - 6.2)**2 / 0.1) +
                       0.8e-15 * np.exp(-(wavelength - 7.7)**2 / 0.15) +
                       0.6e-15 * np.exp(-(wavelength - 11.3)**2 / 0.2))
        
        silicate_feature = 2e-15 * np.exp(-(wavelength - 10)**2 / 1)
        
        total_flux = continuum + pah_features - silicate_feature
        
        # Create matplotlib version
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Total spectrum
        ax1.plot(wavelength, total_flux, color='navy', linewidth=2, label='Total Spectrum')
        ax1.set_xlabel('Wavelength (μm)')
        ax1.set_ylabel('Flux [erg/s/cm²/Å]')
        ax1.set_title('Simulated MIR Spectrum of NGC 7469', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Spectral components
        ax2.plot(wavelength, continuum, color='orange', linewidth=2, label='Continuum')
        ax2.plot(wavelength, pah_features, color='green', linewidth=2, label='PAH Features')
        ax2.plot(wavelength, silicate_feature, color='red', linewidth=2, label='Silicate Absorption')
        ax2.set_xlabel('Wavelength (μm)')
        ax2.set_ylabel('Component Flux [erg/s/cm²/Å]')
        ax2.set_title('Spectral Components', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('Outputs/spectral_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create interactive version if plotly is available
        if PLOTLY_AVAILABLE:
            fig = make_subplots(rows=2, cols=1, 
                              subplot_titles=('Simulated MIR Spectrum of NGC 7469', 'Spectral Components'))
            
            fig.add_trace(go.Scatter(x=wavelength, y=total_flux, mode='lines', 
                                   name='Total Spectrum', line=dict(color='navy', width=3)), 
                         row=1, col=1)
            
            fig.add_trace(go.Scatter(x=wavelength, y=continuum, mode='lines', 
                                   name='Continuum', line=dict(color='orange', width=2)), 
                         row=2, col=1)
            fig.add_trace(go.Scatter(x=wavelength, y=pah_features, mode='lines', 
                                   name='PAH Features', line=dict(color='green', width=2)), 
                         row=2, col=1)
            fig.add_trace(go.Scatter(x=wavelength, y=silicate_feature, mode='lines', 
                                   name='Silicate Absorption', line=dict(color='red', width=2)), 
                         row=2, col=1)
            
            fig.update_xaxes(title_text='Wavelength (μm)', row=2, col=1)
            fig.update_yaxes(title_text='Flux [erg/s/cm²/Å]', row=1, col=1)
            fig.update_yaxes(title_text='Component Flux', row=2, col=1)
            
            fig.update_layout(height=800, showlegend=True)
            fig.write_html('Outputs/spectral_analysis_interactive.html')
    
    def generate_analysis_report(self):
        """Generate a comprehensive analysis report"""
        print("Generating analysis report...")
        
        report = f"""
# MIRI JWST ANALYSIS REPORT
## Target: {self.object_info['name']}

### Basic Properties:
- **Coordinates**: RA {self.object_info['ra']}, Dec {self.object_info['dec']}
- **Distance**: {self.object_info['distance_mpc']} Mpc
- **Redshift**: z = {self.object_info['redshift']}
- **Classification**: {self.object_info['category']} - {self.object_info['subcategory']}

### Scientific Context:
NGC 7469 is a remarkable Seyfert 1 galaxy with concurrent starburst activity, 
making it an ideal laboratory for studying AGN-starburst connections.

### MIRI Advantages:
- Penetrates dusty regions obscuring optical/NIR views
- Reveals PAH features and silicate absorption
- Maps star formation and AGN activity simultaneously
- Provides kinematic information through spectral lines

### Key Science Questions:
1. How do AGN and starburst activities influence each other?
2. What is the spatial distribution of different ISM phases?
3. How does feedback operate in this composite system?

### Analysis Features:
- Multi-wavelength data comparison
- Spatial distribution mapping
- Spectral feature identification
- Physical scale conversions

### Generated Outputs:
- Comprehensive analysis figures
- Radial profile analysis
- Spectral feature identification
- 3D data visualization
- Interactive plots

---
*Generated by MIRI JWST Data Analysis Pipeline*
"""
        
        with open('Outputs/analysis_report.md', 'w') as f:
            f.write(report)
        
        # Also create a simple HTML report
        html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>MIRI JWST Analysis - {self.object_info['name']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  color: white; padding: 30px; border-radius: 10px; }}
        .info-box {{ background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>MIRI JWST Analysis Report</h1>
        <h2>Target: {self.object_info['name']}</h2>
    </div>
    
    <div class="info-box">
        <h3>Object Information</h3>
        <p><strong>Coordinates:</strong> {self.object_info['ra']}, {self.object_info['dec']}</p>
        <p><strong>Distance:</strong> {self.object_info['distance_mpc']} Mpc</p>
        <p><strong>Redshift:</strong> z = {self.object_info['redshift']}</p>
        <p><strong>Type:</strong> {self.object_info['subcategory']}</p>
    </div>
    
    <h3>Generated Plots</h3>
    <p>Check the Outputs directory for:</p>
    <ul>
        <li>Comprehensive analysis figure (ngc7469_comprehensive_analysis.png)</li>
        <li>Radial profile analysis (radial_profile.png)</li>
        <li>Spectral analysis (spectral_analysis.png)</li>
        <li>3D visualization (3d_surface_*.png/html)</li>
    </ul>
</body>
</html>
"""
        
        with open('Outputs/analysis_report.html', 'w') as f:
            f.write(html_report)
        
        print("✓ Analysis reports generated: Outputs/analysis_report.md and Outputs/analysis_report.html")

def main():
    """Main execution function"""
    print("🚀 Starting MIRI JWST Data Analysis for NGC 7469")
    print("=" * 50)
    
    analyzer = NGC7469Analyzer()
    
    # Load data (replace with actual file path)
    data, header, wcs = analyzer.load_fits_data('Data/jw01328-c1006_t014_miri_ch1-short_s3d.fits')
    
    # Calculate spatial scales
    scales = analyzer.calculate_spatial_scales(header)
    print("✓ Spatial scales calculated")
    print(f"  - Pixel scale: {scales['arcsec_per_pixel']:.4f} arcsec/pixel")
    print(f"  - Physical scale: {scales['pc_per_pixel']:.2f} pc/pixel")
    
    # Create visualizations
    analyzer.create_enhanced_visualizations(data, scales)
    print("✓ All visualizations created")
    
    # Generate report
    analyzer.generate_analysis_report()
    print("✓ Analysis reports generated")
    
    print("\n" + "=" * 50)
    print("🎉 Analysis complete! Check the 'Outputs' directory for results.")
    print("\nGenerated files:")
    for file in os.listdir('Outputs'):
        if file.endswith(('.png', '.html', '.md')):
            print(f"  - Outputs/{file}")

if __name__ == "__main__":
    main()
