import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_heatmaps2D(prediction_plot, true_plot, error_plot):

    fig = plt.figure(figsize=(18, 6), dpi=100, constrained_layout=True)  # Enable constrained layout
            
    # ================================================================================
    # Prediction Plot
    # ================================================================================
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(prediction_plot, clim=[0, 1], cmap='coolwarm')
    plt.title("Model Prediction", fontsize=14, pad=15)
    plt.axis('off')
    
    # Add colorbar to the first plot
    cbar1 = fig.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05, aspect=40)
    
    # ================================================================================
    # Ground Truth Plot
    # ================================================================================
    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(true_plot, clim=[0, 1], cmap='coolwarm')
    plt.title("Ground Truth", fontsize=14, pad=15)
    plt.axis('off')
    
    # Shared colorbar for second plot
    cbar2 = fig.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05, aspect=40)
    
    # ================================================================================
    # Error Analysis Plot
    # ================================================================================
    ax3 = plt.subplot(1, 3, 3)
    im3 = ax3.imshow(error_plot, clim=[0, 0.05], cmap='RdYlGn_r')
    plt.title("Error Map", fontsize=14, pad=15)
    plt.axis('off')
    
    # Error colorbar
    cbar3 = fig.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.05, aspect=40)
    cbar3.set_label('Absolute Error', fontsize=10)
    cbar3.ax.axvline(0.03, color='black', linestyle='--', linewidth=1)
    
    # ================================================================================
    # Final Adjustments
    # ================================================================================
    plt.savefig('model_comparison.png', bbox_inches='tight', dpi=300)
    plt.show()

def plot_errors(x_variable, errors, plot_name = "Errors", xlabel = "x variable"):

    plt.figure(figsize=(10, 6), dpi=100)
    plt.style.use('classic')  # Modern style (requires seaborn installed)

    # Create scatter plot with color mapping
    sc = plt.scatter(
        x=np.array(x_variable),
        y=errors,
        c=errors,
        cmap='coolwarm',  # Yellow-Orange-Red sequential colormap
        s=80,
        edgecolor='none',
        linewidth=0.8,
        alpha=0.9,
        zorder=3
    )
    
    # Add horizontal line for max error
    max_error = np.max(errors)
    plt.axhline(max_error, color='#2f4b7c', linestyle=':', linewidth=1.5, 
                label=f'Max Error: {max_error:.4f}')
    
    # Formatting
    plt.title(plot_name, fontsize=14, pad=15, 
                fontweight='bold', color='#222222')
    plt.xlabel(xlabel, fontsize=12, labelpad=8, color='#222222')
    plt.ylabel("$\overline{E}$", fontsize=12, labelpad=8, color='#222222')
    
    # Colorbar setup
    cbar = plt.colorbar(sc)
    cbar.set_label('Error Intensity', fontsize=10, labelpad=10)
    cbar.outline.set_visible(False)
    
    # Axis styling
    plt.xticks(fontsize=10, color='#222222')
    plt.yticks(fontsize=10, color='#222222')
    plt.grid(True, alpha=0.3, zorder=0)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Legend for max error line
    legend = plt.legend(loc='lower right', frameon=True, 
                        facecolor='white', edgecolor='k')
    legend.get_frame().set_alpha(0.9)
    
    # Set sensible y-axis limits
    plt.ylim(bottom=0, top=max_error*1.1)  # 10% padding on top
    plt.xlim(left = (np.min(x_variable)*0.99), right =(np.max(x_variable)*1.01) )
    
    plt.tight_layout()
    plt.show()

def plot_isosurfaces_3D(prediction, true, error):
    """
    Plots 3D isosurface visualizations for prediction, true values, and error side by side.
    
    Parameters:
        prediction (numpy.ndarray): 3D array of predicted velocity values (shape: num_X × num_Y × num_Z)
        true (numpy.ndarray): 3D array of true velocity values (same shape as prediction)
        error (numpy.ndarray): 3D array of absolute error values (same shape as prediction)
    
    Returns:
        plotly.graph_objs._figure.Figure: Figure containing the 3D subplots
    """
    # Generate grid coordinates
    num_X, num_Y, num_Z = prediction.shape
    x, y, z = np.mgrid[0:num_X, 0:num_Y, 0:num_Z]
    
    # Create subplots with 3D scenes
    fig = make_subplots(rows=1, cols=3,
                        specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
                        subplot_titles=('Prediction', 'True', 'Error'))
    
    # Add prediction isosurface
    fig.add_trace(go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=prediction.flatten(),
        isomin=prediction.min(),
        isomax=prediction.max(),
        surface_count=3,
        colorscale='Viridis',
        opacity=0.6,
        name='Prediction'
    ), row=1, col=1)
    
    # Add true isosurface
    fig.add_trace(go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=true.flatten(),
        isomin=true.min(),
        isomax=true.max(),
        surface_count=3,
        colorscale='Viridis',
        opacity=0.6,
        name='True'
    ), row=1, col=2)
    
    # Add error isosurface
    fig.add_trace(go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=error.flatten(),
        isomin=error.min(),
        isomax=error.max(),
        surface_count=3,
        colorscale='Reds',
        opacity=0.6,
        name='Error'
    ), row=1, col=3)
    
    # Update layout for consistent appearance
    fig.update_layout(
        title_text='3D Velocity Map Comparison',
        width=1500,
        height=500,
        scene=dict(
            xaxis_title='X Index',
            yaxis_title='Y Index',
            zaxis_title='Z Index',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))  # Consistent 3D perspective
        ),
        scene2=dict(
            xaxis_title='X Index',
            yaxis_title='Y Index',
            zaxis_title='Z Index',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        scene3=dict(
            xaxis_title='X Index',
            yaxis_title='Y Index',
            zaxis_title='Z Index',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        )
    )
    
    return fig