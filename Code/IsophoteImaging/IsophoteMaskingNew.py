import argparse,pickle,sys,warnings
import numpy as np
import matplotlib.pylab as plt
from math import pi,degrees
import PySimpleGUI as sg
from numpy import sin,cos
from numpy.linalg import eig, inv
from matplotlib.patches import Circle,Ellipse
from skimage.measure import ransac,EllipseModel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
plt.rcParams.update({'text.usetex':False})
warnings.filterwarnings("ignore")
def myprint(string,clear=False):
    if clear:
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K") 
    print(string)
def pix2kpc(pix,width):
    return(pix/1000.*width)

parser = argparse.ArgumentParser(description='Collect images of all resolved halos from a given simulation. Images will be generated across all orientations.')
parser.add_argument('-f','--feedback',choices=['BW','SB','RDZ'],default='RDZ',help='Feedback Model')
parser.add_argument('-s','--simulation',required=True,help='Simulation to analyze')
#parser.add_argument('-o','--overwrite',action='store_true',help='Overwrite existing images')
args = parser.parse_args()

# Assuming ellipse_params is defined globally
ellipse_params = None
image_size = (10, 10)
iso_size=1

#Load sim-level data
Images = pickle.load(open(f'../../Data/{args.simulation}.{args.feedback}.Images.pickle','rb'))
ShapeData = pickle.load(open(f'../../Data/{args.simulation}.{args.feedback}.ShapeData.pickle','rb'))
Masking = pickle.load(open(f'../../Data/{args.simulation}.{args.feedback}.Masking.pickle','rb'))
Profiles = pickle.load(open(f'../../Data/{args.simulation}.{args.feedback}.Profiles.pickle','rb'))



halos = list(Images.keys())  # Assuming 'Images' is a dictionary with halo keys
current_halo_index = 0
current_rotation_index = 0


def get_current_halo_rotation():
    halo = halos[current_halo_index]
    rotations = list(Images[halo].keys())
    if current_rotation_index < len(rotations):
        rotation = rotations[current_rotation_index]
    else:
        rotation = None
    return halo, rotation

def drawInteractiveEllipse(mode, LogImage, iso):
    global ellipse_params  # Use global here to modify ellipse_params
    plt.close('all')  # Close any existing figures
    fig, ax = plt.subplots(figsize=image_size)
    ax.imshow(LogImage, origin='lower')  # Display the image
    # Display any additional points as needed
    ax.scatter(500, 500, marker='+', s=2*10**2, c='w')
    ax.scatter(iso[1], iso[0], c='r', s=iso_size)

    points = []

    def onclick(event):
        global ellipse_params
        points.append((event.xdata, event.ydata))
        
        # Plot the point as it's clicked
        ax.plot(event.xdata, event.ydata, marker='o', markersize=5, color='blue')

        # Update the figure to show the new point
        fig.canvas.draw()
        
        if len(points) == 3:
            # Calculate the ellipse parameters here based on the three points
            center, major, minor = points[0], points[1], points[2]
            ell_width = np.linalg.norm(np.array(major) - np.array(center)) * 2
            ell_height = np.linalg.norm(np.array(minor) - np.array(center)) * 2
            angle = np.degrees(np.arctan2(major[1] - center[1], major[0] - center[0]))

            # Update the global variable
            ellipse_params = {'center': center, 'width': ell_width, 'height': ell_height, 'angle': angle}

            # Optionally, plot the defined ellipse here before closing if you want to see it
            # (You would need to calculate the ellipse plot here)

            # Close the figure after three points have been selected
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()  # Show the plot for interactive selection
    return ellipse_params






# Masking Functions
def MaskCircle(rad, cen_x, cen_y, isophote, mode):
    assert mode in ['Inclusive', 'Exclusive'], 'Masking Mode Error'
    masked_iso = [[], []]
    for i in np.arange(len(isophote[0])):
        r = np.sqrt((isophote[0][i] - 500 - cen_y) ** 2 + (isophote[1][i] - 500 - cen_x) ** 2)
        if r < rad and mode == 'Inclusive':
            masked_iso[0].append(isophote[0][i])
            masked_iso[1].append(isophote[1][i])
        if r > rad and mode == 'Exclusive':
            masked_iso[0].append(isophote[0][i])
            masked_iso[1].append(isophote[1][i])
    masked_iso = (np.array(masked_iso[0]), np.array(masked_iso[1]))
    return masked_iso


def MaskEllipse(ellipse_params, isophote, mode):

    assert mode in ['Inclusive', 'Exclusive'], 'Masking Mode Error'
    masked_iso = [[], []]

    # Correctly extract ellipse parameters
    center = ellipse_params['center']
    a, b = ellipse_params['width'] / 2, ellipse_params['height'] / 2
    angle = np.radians(ellipse_params['angle'])

    for x, y in zip(isophote[1], isophote[0]):
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)
        x_, y_ = cos_angle * (x - center[0]) + sin_angle * (y - center[1]), sin_angle * (x - center[0]) - cos_angle * (y - center[1])

        ellipse_eq = (x_ / a) ** 2 + (y_ / b) ** 2
        if mode == 'Inclusive':
            if ellipse_eq <= 1:
                masked_iso[0].append(y)
                masked_iso[1].append(x)
        elif mode == 'Exclusive':
            if ellipse_eq > 1:
                masked_iso[0].append(y)
                masked_iso[1].append(x)



    return (np.array(masked_iso[0]), np.array(masked_iso[1]))

def calculate_ellipse_distances(xy, xc, yc, a, b, theta):
    cos_angle = np.cos(np.radians(theta))
    sin_angle = np.sin(np.radians(theta))
    
    x, y = xy[:, 0] - xc, xy[:, 1] - yc
    x_ = cos_angle * x + sin_angle * y
    y_ = -sin_angle * x + cos_angle * y
    
    distance_normalized = (x_**2 / a**2) + (y_**2 / b**2)
    distances = np.abs(distance_normalized - 1)
    
    return distances

def percentile_based_ellipse_fitting(iso, percentile=97, max_iterations=1, param_change_threshold=0.01):
    xy = np.column_stack((iso[1], iso[0]))  # Reformat iso data for fitting
    prev_params = None
    best_fit_xy = None  # To keep track of the best fitting points

    for iteration in range(max_iterations):
        # Fit Ellipse to current data
        ellipse = EllipseModel()
        ellipse.estimate(xy)
        params = ellipse.params  # xc, yc, a, b, theta

        if prev_params is not None:
            # Calculate relative change in parameters
            param_changes = np.abs((np.array(params) - np.array(prev_params)) / np.array(prev_params))
            # Check for convergence
            if np.all(param_changes < param_change_threshold):
                xy = best_fit_xy  # Revert to the best fitting points before the last iteration
                break  # Ellipse shape has stabilized
        best_fit_xy = xy.copy()  # Update the best fitting points
        prev_params = params  # Update previous parameters for next iteration comparison
        
        # Calculate distances (residuals) from each point to the fitted ellipse
        distances = calculate_ellipse_distances(xy, *params)
        
        # Determine the distance threshold based on the desired percentile
        threshold = np.percentile(distances, percentile)
        
        # Filter points: keep points with distances below the threshold
        xy = xy[distances < threshold]

    if best_fit_xy is not None:
        # Use the best fitting points for the final mask
        masked_iso = [best_fit_xy[:, 1], best_fit_xy[:, 0]]  # Reformat to match the original 'iso' structure
    else:
        # If best_fit_xy is None (no iterations were done), return the original data
        masked_iso = iso

    return masked_iso

# from sklearn.linear_model import RANSACRegressor

# def robust_ellipse_fitting(iso, percentile=97, max_iterations=10, param_change_threshold=0.01):
#     xy = np.column_stack((iso[1], iso[0]))  # Reformat iso data for fitting
#     prev_params = None
#     best_fit_xy = None

#     for iteration in range(max_iterations):
#         # Fit Ellipse to current data using RANSAC
#         ellipse = EllipseModel()
#         ransac = RANSACRegressor(ellipse, min_samples=3, residual_threshold=1, max_trials=100)
#         ransac.fit(xy[:, 0].reshape(-1, 1), xy[:, 1])
        
#         # Assuming the EllipseModel could be fitted in such a way, otherwise, adapt the RANSAC output to ellipse parameters
#         params = ransac.estimator_.params  # xc, yc, a, b, theta

#         if prev_params is not None:
#             param_changes = np.abs((np.array(params) - np.array(prev_params)) / np.array(prev_params))
#             if np.all(param_changes < param_change_threshold):
#                 xy = best_fit_xy
#                 break
#         best_fit_xy = xy.copy()
#         prev_params = params
        
#         distances = calculate_ellipse_distances(xy, *params)
#         threshold = np.percentile(distances, percentile)
#         xy = xy[distances < threshold]

#     if best_fit_xy is not None:
#         masked_iso = [best_fit_xy[:, 1], best_fit_xy[:, 0]]
#     else:
#         masked_iso = iso

#     return masked_iso




# GUI Functions
def InitializeGUI(PlotName):
    # GUI Properties
    _VARS = {'window': False, 'fig_agg': False, 'pltFig': False}
    GuiFont = 'Any 16'
    GuiColor = '#E8E8E8'
    sg.theme('black')
    layout = [
        [sg.Text(text=PlotName,
                 font=GuiFont,
                 background_color=GuiColor,
                 text_color='Black', key='-PLOTNAME-')],
        [sg.Canvas(key='figCanvas',
                   background_color=GuiColor)],
        [sg.Text(text='Circular Mask:',
                 font=GuiFont,
                 background_color=GuiColor,
                 text_color='Black'),
         sg.Listbox(['Inclusive', 'Exclusive'],
                    key='CMode', size=(10, 1)),
         sg.Text(text='Radius:',
                 font=GuiFont,
                 background_color=GuiColor,
                 text_color='Black'),
         sg.Slider(range=(0, 400),
                   orientation='h', size=(20, 20),
                   default_value=0,
                   background_color=GuiColor,
                   resolution=1,
                   text_color='Black',
                   key='RadiusAdjust',
                   enable_events=True),
         sg.Text(text='Cen-X:',
                 font=GuiFont,
                 background_color=GuiColor,
                 text_color='Black'),
         sg.Slider(range=(-500, 500),
                   orientation='h', size=(20, 20),
                   default_value=0,
                   background_color=GuiColor,
                   resolution=1,
                   text_color='Black',
                   key='CenXAdjust',
                   enable_events=True),
         sg.Text(text='Cen-Y:',
                 font=GuiFont,
                 background_color=GuiColor,
                 text_color='Black'),
         sg.Slider(range=(-500, 500),
                   orientation='h', size=(20, 20),
                   default_value=0,
                   background_color=GuiColor,
                   resolution=1,
                   text_color='Black',
                   key='CenYAdjust',
                   enable_events=True)],
        [
        sg.Button('Update Ellipse', key='UpdateEllipse', font=GuiFont),
        sg.Button('Draw Ellipse', font=GuiFont),  # New button for drawing ellipse in its own row
        sg.Listbox(['Inclusive', 'Exclusive'], key='EllipseMode', size=(10, 1), default_values=['Inclusive']),  # Default to 'Inclusive'
        ],
        [
        sg.Text('Ellipse Adjustments:', font=GuiFont, background_color=GuiColor, text_color='Black'),
        sg.Text('Center X:', font=GuiFont, background_color=GuiColor, text_color='Black'),
        sg.Slider(range=(0, 1000), orientation='h', size=(20, 20), resolution=1, default_value=500, key='EllipseCenterX', background_color=GuiColor, text_color='Black'),
        sg.Text('Center Y:', font=GuiFont, background_color=GuiColor, text_color='Black'),
        sg.Slider(range=(0, 1000), orientation='h', size=(20, 20), resolution=1, default_value=500, key='EllipseCenterY', background_color=GuiColor, text_color='Black'),
        sg.Text('Width:', font=GuiFont, background_color=GuiColor, text_color='Black'),
        sg.Slider(range=(0, 1000), orientation='h', size=(20, 20), resolution=1, default_value=100, key='EllipseWidth', background_color=GuiColor, text_color='Black'),
        sg.Text('Height:', font=GuiFont, background_color=GuiColor, text_color='Black'),
        sg.Slider(range=(0, 1000), orientation='h', size=(20, 20), resolution=1, default_value=50, key='EllipseHeight', background_color=GuiColor, text_color='Black'),
        sg.Text('Angle:', font=GuiFont, background_color=GuiColor, text_color='Black'),
        sg.Slider(range=(-180, 180), orientation='h', size=(20, 20), resolution=1, default_value=0, key='EllipseAngle', background_color=GuiColor, text_color='Black')
        ],

        [sg.Button('Exit', font=GuiFont),
         sg.Button('Ignore Halo', font=GuiFont),
         sg.Button('Ignore', font=GuiFont, pad=((0, 130), (0, 0))),
         sg.Text(text='Isophote %:',
                 font=GuiFont,
                 background_color=GuiColor,
                 text_color='Black'),
         sg.InputText('1',
                      background_color=GuiColor,
                      font=GuiFont,
                      size=(5, 1),
                      text_color='Black',
                      key='Iso%'),
         sg.Button('Reset', font=GuiFont),
         sg.Button('Mask', font=GuiFont),
         #sg.Button('Auto Mask', font=GuiFont),
         sg.Button('Fit Ellipse', font=GuiFont)         ,
         sg.Button('Save', font=GuiFont)],
        [sg.Text('Auto Mask Percentile:',font=GuiFont, background_color=GuiColor, text_color='Black'),
         sg.Slider(range=(90, 99), default_value=97, orientation='horizontal', size=(20, 15), key='percentile_slider', background_color=GuiColor, text_color='Black'),
         sg.Text('Parameter Change Threshold:',font=GuiFont, background_color=GuiColor, text_color='Black'),
         sg.Slider(range=(0.001, 0.05), resolution=0.001, default_value=0.01, orientation='horizontal', size=(20, 15), key='param_change_threshold_slider', background_color=GuiColor, text_color='Black'),
         sg.Button('Auto Mask')],
        [sg.Button('Previous Halo', font=GuiFont), sg.Button('Next Halo', font=GuiFont),
        sg.Button('Previous Rotation', font=GuiFont), sg.Button('Next Rotation', font=GuiFont)],
    ]
    _VARS['window'] = sg.Window('Isophote Fitting',
                                layout,
                                finalize=True,
                                resizable=True,
                                location=(100, 100),
                                element_justification="center",
                                background_color=GuiColor)
    return _VARS


def draw_figure(canvas, figure):
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        return figure_canvas_agg
def drawChart():
    _VARS['pltFig'] = plt.figure(figsize=image_size)
    plt.imshow(LogImage)
    #plt.grid(b=None)
    _VARS['pltFig'].axes[0].set_xlim([0,1000])
    _VARS['pltFig'].axes[0].set_ylim([0,1000])
    #plt.grid(b=None)
    plt.scatter(500,500,marker='+',s=2*10**2,c='w')
    plt.scatter(iso[1],iso[0],c='r',s=iso_size)
    _VARS['fig_agg'] = draw_figure(_VARS['window']['figCanvas'].TKCanvas,_VARS['pltFig'])
def updateChart(iso, DrawCircle=False, DrawAngle=False, DrawEllipse=False, DrawUserEllipse=False):
    global ellipse_patches  # Access the global variable

    # Remove existing ellipse patches
    for patch in ellipse_patches:
        patch.remove()
    ellipse_patches.clear()  # Clear the list after removing the patches

    _VARS['fig_agg'].get_tk_widget().forget()
    _VARS['pltFig'].clf()  # Clear the figure
    new_plot_name = f"Halo {halo} - {rotation}"  # Construct the new plot name based on current data
    _VARS['window']['-PLOTNAME-'].update(new_plot_name)
    ax = _VARS['pltFig'].add_subplot(111)  # Recreate the axes
    ax.imshow(LogImage, origin='lower')
    ax.set_xlim([0, 1000])
    ax.set_ylim([0, 1000])
    ax.scatter(iso[1], iso[0], c='r', s=.5**2)  # Redraw with updated iso points

    if DrawCircle:
        _VARS['pltFig'].axes[0].add_patch(Circle((500+values['CenXAdjust'], 500+values['CenYAdjust']),
                                                 values['RadiusAdjust'], color='w', fill=False))
    
    if DrawAngle:
        plt.plot([500, 500+710*np.cos(np.radians(values['MinAngle']))],
                 [500, 500+710*np.sin(np.radians(values['MinAngle']))], c='w', linewidth=1)
        plt.plot([500, 500+710*np.cos(np.radians(values['MaxAngle']))],
                 [500, 500+710*np.sin(np.radians(values['MaxAngle']))], c='w', linewidth=1)
    
    if DrawEllipse:
        # Fit Ellipse to Isophote
        xy = np.zeros((len(iso[0]), 2))
        for idx in range(len(iso[0])):
            xy[idx] = [iso[1][idx], iso[0][idx]]
        E = EllipseModel()
        E.estimate(np.array(xy))
        params = E.params
        cen = np.array([params[0], params[1]])
        phi = params[4]
        a, b = params[2], params[3]
        _VARS['pltFig'].axes[0].add_patch(Ellipse(cen, 2*a, 2*b, angle=degrees(phi), facecolor='None', edgecolor='orange'))

    if DrawUserEllipse and ellipse_params is not None:
        # Assuming ellipse_params contains the user-drawn ellipse parameters
        user_center, user_width, user_height, user_angle = ellipse_params['center'], ellipse_params['width'], ellipse_params['height'], ellipse_params['angle']
        # Draw user ellipse with distinct properties, e.g., blue dashed line
        _VARS['pltFig'].axes[0].add_patch(Ellipse(user_center, user_width, user_height, angle=user_angle, facecolor='None', edgecolor='blue', linestyle='--', linewidth=2))
    if DrawUserEllipse and ellipse_params:
           # Draw the user-defined ellipse with the parameters
           user_center, user_width, user_height, user_angle = ellipse_params['center'], ellipse_params['width'], ellipse_params['height'], ellipse_params['angle']
           user_ellipse = Ellipse(user_center, user_width, user_height, angle=user_angle, facecolor='None', edgecolor='blue', linestyle='--', linewidth=2)
           ax.add_patch(user_ellipse)
           ellipse_patches.append(user_ellipse)  # Keep track of the patch

    _VARS['fig_agg'] = draw_figure(_VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])

#Create new Isophote Plot
def RMS(res):
    if not isinstance(res,np.ndarray): res = np.array(res)
    return( np.sqrt(sum(res**2)/len(res)) )
def SavePlot(logimage,iso,Rhalf,fname):
    f,ax = plt.subplots(1,1)
    ax.imshow(logimage)
    ax.set_xlim([0,1e3])
    ax.set_ylim([1e3,0])
    ax.scatter(500,500,c='k',marker='+')
    if len(iso[0])>0:
        ax.scatter(iso[1],iso[0],c='r',marker='.',s=1)
        xy=np.zeros((len(iso[0]),2))
        for idx in range(len(iso[0])):
            xy[idx]=[iso[1][idx],iso[0][idx]]
        #Fit ellipse
        E = EllipseModel()
        E.estimate(np.array(xy))
        params = E.params
        cen = np.array([params[0],params[1]])
        phi = params[4]
        a,b = params[2],params[3]
        #a = max([params[2],params[3]])
        #b = min([params[2],params[3]])
        residual = E.residuals(np.array(xy))
        rms = RMS(residual)
        #Plot Ellipse and Fit Results
        ax.add_patch(Ellipse(cen,2*a,2*b,angle=degrees(phi),facecolor='None',edgecolor='orange'))
        plt.plot([-a*cos(phi)+cen[0],a*cos(phi)+cen[0]],[-a*sin(phi)+cen[1],a*sin(phi)+cen[1]],color='orange')
        plt.plot([-b*cos(phi+pi/2)+cen[0],b*cos(phi+pi/2)+cen[0]],[-b*sin(phi+pi/2)+cen[1],b*sin(phi+pi/2)+cen[1]],color='orange')
        atrue,btrue = max([a,b]),min([a,b])
        ax.set_title(f'b/a: {round(btrue/atrue,3)}  RMS: {round(rms,3)}  Manual: True',fontsize=15)
        f.savefig(fname,bbox_inches='tight',pad_inches=.1)
        out = {}
        out['b/a'] = btrue/atrue
        out['a'] = pix2kpc(atrue,6*Rhalf)
        out['b'] = pix2kpc(btrue,6*Rhalf)
        return(out)
    else:
        ax.set_title(f'b/a: NaN  RMS: NaN  Manual: True',fontsize=15)
        f.savefig(fname,bbox_inches='tight',pad_inches=.1)
        return({'b/a':np.NaN,'a':np.NaN,'b':np.NaN})



#test = pickle.load(open(f'../../Data/{args.simulation}.{args.feedback}.ShapeData.pickle','rb'))
def load_and_display_current_image():
    global LogImage, iso  # Make sure these are accessible and updatable globally

    halo, rotation = get_current_halo_rotation()  # Retrieve current halo and rotation
    if rotation:  # Check if rotation is valid (not None)
        # Load image and isophote data for the current halo and rotation
        Image = Images[halo][rotation]
        Image = np.flip(Image, 0)
        LogImage = plt.imread(f'../../Figures/Images/{args.simulation}.{args.feedback}/{halo}/{halo}.{".y".join(rotation.split("y"))}.png')
        Image = Images[halo][rotation]
        Image = np.flip(Image, 0)
        LogImage = plt.imread(f'../../Figures/Images/{args.simulation}.{args.feedback}/{halo}/{halo}.{".y".join(rotation.split("y"))}.png')
        rbins = Profiles[halo][rotation]['rbins']
        Rhalf = Profiles[halo][rotation]['Reff']
        if np.isnan(Rhalf): Rhalf = Profiles[halo][rotation]['Rhalf']
        ind_eff = np.argmin(abs(rbins - Rhalf))
        v = Profiles[halo][rotation]['v_lum_den'][ind_eff]
        tol = .01
        iso = np.where((Image > v * (1 - tol)) & (Image < v * (1 + tol)))
        
        # Example of retrieving isophote points for the current image
        # This will need to be adapted based on how your isophote points are stored or calculated
        # For demonstration, let's assume iso is recalculated or retrieved here
        
        # After loading the new LogImage and iso, update the chart
        updateChart(iso, DrawCircle=False, DrawEllipse=False, DrawUserEllipse=False)  # Adapt parameters as needed


# what i want 
# plot origianlly data, and masked data on top of each other, and plot ellipse if it has been found 
# navigate forward and backwards though buttons

ellipse_patches = []  # Holds the matplotlib patches for ellipses


halo, rotation = get_current_halo_rotation()
_VARS = InitializeGUI(f'Halo {halo} - {rotation}')
Image = Images[halo][rotation]
Image = np.flip(Image, 0)
LogImage = plt.imread(f'../../Figures/Images/{args.simulation}.{args.feedback}/{halo}/{halo}.{".y".join(rotation.split("y"))}.png')
rbins = Profiles[halo][rotation]['rbins']
Rhalf = Profiles[halo][rotation]['Reff']
if np.isnan(Rhalf): Rhalf = Profiles[halo][rotation]['Rhalf']
ind_eff = np.argmin(abs(rbins - Rhalf))
v = Profiles[halo][rotation]['v_lum_den'][ind_eff]
tol = .01
iso = np.where((Image > v * (1 - tol)) & (Image < v * (1 + tol)))
drawChart()
load_and_display_current_image()
new_view=False

while True:  # Main event loop
    halo, rotation = get_current_halo_rotation()
    print(f'Processing {args.simulation} {halo}-{rotation}...')
    
    # Load halo-level data
    Image = Images[halo][rotation]
    Image = np.flip(Image, 0)
    LogImage = plt.imread(f'../../Figures/Images/{args.simulation}.{args.feedback}/{halo}/{halo}.{".y".join(rotation.split("y"))}.png')

    # Find default isophote
    rbins = Profiles[halo][rotation]['rbins']
    Rhalf = Profiles[halo][rotation]['Reff']
    if np.isnan(Rhalf): Rhalf = Profiles[halo][rotation]['Rhalf']
    ind_eff = np.argmin(abs(rbins - Rhalf))
    v = Profiles[halo][rotation]['v_lum_den'][ind_eff]
    tol = .01
    #iso = np.where((Image > v * (1 - tol)) & (Image < v * (1 + tol)))
    #load_and_display_current_image() 
    if new_view == True:
        iso = np.where((Image > v * (1 - tol)) & (Image < v * (1 + tol)))
        load_and_display_current_image() 
    new_view=False
    

    DC, DE = False, False  # Updated: Removed DA

    event, values = _VARS['window'].read()
    if event in [sg.WIN_CLOSED, 'Exit']:
        break
    if event in ['Next Halo', 'Previous Halo', 'Next Rotation', 'Previous Rotation']:
        # Update current_halo_index or current_rotation_index based on the event
        
        
        if event == 'Next Halo':
            current_halo_index = min(current_halo_index + 1, len(halos) - 1)
            current_rotation_index = 0  # Reset rotation index when changing halos
        elif event == 'Previous Halo':
            current_halo_index = max(current_halo_index - 1, 0)
            current_rotation_index = 0  # Reset rotation index when changing halos
        elif event == 'Next Rotation':
            rotations = list(Images[halos[current_halo_index]].keys())
            current_rotation_index = min(current_rotation_index + 1, len(rotations) - 1)
        elif event == 'Previous Rotation':
            current_rotation_index = max(current_rotation_index - 1, 0)
        new_view = True
            
        load_and_display_current_image()  # Load and display the image for the new selection
        
    if event in ['Save', 'Ignore']:
        rotations = list(Images[halos[current_halo_index]].keys())
        if current_rotation_index == len(rotations) - 1:
            sg.popup('This was the last rotation for the current halo.', title='Notification')
        if current_halo_index == len(halos) - 1 and current_rotation_index == len(rotations) - 1:
            sg.popup('This was the last rotation of the last halo.', title='Notification')
        #_VARS['window'].close()
        myprint(f'Saving {args.simulation} {halo}-{rotation}...', clear=True)
        if event == 'Ignore': 
            iso = [[], []]
        ba = SavePlot(LogImage, iso, Rhalf, f'../../Figures/Images/{args.simulation}.{args.feedback}/{halo}/{halo}.{".y".join(rotation.split("y"))}.Isophote.png')
        ShapeData[halo][rotation] = ba
        pickle.dump(ShapeData, open(f'../../Data/{args.simulation}.{args.feedback}.ShapeData.pickle', 'wb'))
        Masking[halo][rotation] = False
        pickle.dump(Masking, open(f'../../Data/{args.simulation}.{args.feedback}.Masking.pickle', 'wb'))
        myprint(f'{args.simulation} {halo}-{rotation} saved.', clear=True)
        #advance to next rotation
        rotations = list(Images[halos[current_halo_index]].keys())
        current_rotation_index = min(current_rotation_index + 1, len(rotations) - 1)
        load_and_display_current_image() 
        #break
    if event == 'Reset':
        tol = float(values['Iso%']) / 100
        iso = np.where((Image > v * (1 - tol)) & (Image < v * (1 + tol)))
        iso_mask = np.where((Image > v * (1 - tol)) & (Image < v * (1 + tol)))
        DC, DE = False, False  # Updated: Removed DA
        updateChart(iso,DrawCircle=DC, DrawEllipse=DE)
    if event == 'Mask': #keep track of iso, and masked iso seperately
        DE = False
        if DC:
            iso = MaskCircle(values['RadiusAdjust'], cen_x=values['CenXAdjust'],
                             cen_y=values['CenYAdjust'], isophote=iso, mode=values['CMode'][0])
            updateChart(iso,DrawEllipse=DE)
        mode = values['EllipseMode'][0] if values['EllipseMode'] else 'Inclusive'  # Default to 'Inclusive' if not set
        if ellipse_params:  # Check if ellipse_params is populated
            
            iso = MaskEllipse(ellipse_params, iso, mode)
            updateChart(iso,DrawEllipse=DE)
    if event == 'Auto Mask':
        percentile = values['percentile_slider']
        param_change_threshold = values['param_change_threshold_slider']
        iso = percentile_based_ellipse_fitting(iso,percentile=percentile,max_iterations=10,param_change_threshold=param_change_threshold)
        updateChart(iso,DrawCircle=DC, DrawEllipse=True)
        
               
        #updateChart(DrawCircle=DC, DrawEllipse=DE)
    if event in ['RadiusAdjust', 'CenXAdjust', 'CenYAdjust'] and values['CMode'] != []:
        DC, DE = True, False
        updateChart(iso,DrawCircle=DC, DrawEllipse=DE)
    if event == 'Fit Ellipse':  # This might be adjusted or integrated with new ellipse drawing functionality
        DE = True
        updateChart(iso,DrawCircle=DC, DrawEllipse=True)
    if event == 'Draw Ellipse':
        if values['EllipseMode']:  # Check if the list is not empty
            mode = values['EllipseMode'][0]  # Now safe to access the first element
            ellipse_params = drawInteractiveEllipse(mode, LogImage, iso)
            _VARS['window']['EllipseCenterX'].update(value=ellipse_params['center'][0])
            _VARS['window']['EllipseCenterY'].update(value=ellipse_params['center'][1])
            _VARS['window']['EllipseWidth'].update(value=ellipse_params['width'])
            _VARS['window']['EllipseHeight'].update(value=ellipse_params['height'])
            _VARS['window']['EllipseAngle'].update(value=ellipse_params['angle'])
            updateChart(iso, DrawUserEllipse=True)
        else:
            sg.popup("Please select an ellipse mode.")
    elif event == 'UpdateEllipse':
        ellipse_params = {
            'center': (values['EllipseCenterX'], values['EllipseCenterY']),
            'width': values['EllipseWidth'],
            'height': values['EllipseHeight'],
            'angle': values['EllipseAngle']
        }
        updateChart(iso, DrawUserEllipse=True)
    if event == 'Ignore Halo':
        confirmation = sg.popup_ok_cancel('Are you sure you want to ignore this halo? This action cannot be undone.', title='Confirm Ignore')
        if confirmation == 'OK':
            # Loop through all rotations for the current halo
            
            for rotation in Masking[halo]:
                # Mark the halo rotation as processed/ignored
                Masking[halo][rotation] = False
                # Set ShapeData for the halo rotation to np.nan
                ShapeData[halo][rotation] = {'b/a': np.nan, 'a': np.nan, 'b': np.nan}
        
            # Optionally, save the updated dictionaries back to the pickle files
            # Ensure you do this only if necessary to avoid redundant I/O operations
            pickle.dump(Masking, open(f'../../Data/{args.simulation}.{args.feedback}.Masking.pickle', 'wb'))
            pickle.dump(ShapeData, open(f'../../Data/{args.simulation}.{args.feedback}.ShapeData.pickle', 'wb'))
        
            # Provide feedback in the console or GUI about the ignored halo
            print(f'Halo {halo} ignored. Moving to the next halo...')
        
            # Implement logic to move to the next halo here
            # This might involve breaking out of the loop, fetching the next halo data, or other steps
            #break  # Skip the rest of the loop for this iteration if appropriate
        
    halo, rotation = get_current_halo_rotation()
    
plt.close()
_VARS['window'].close()    
    
    
''' 
for halo in Images:
    for rotation in Images[halo]:
        print(f'Processing {args.simulation} {halo}-{rotation}...')
        
        # Load halo-level data
        Image = Images[halo][rotation]
        Image = np.flip(Image, 0)
        LogImage = plt.imread(f'../../Figures/Images/{args.simulation}.{args.feedback}/{halo}/{halo}.{".y".join(rotation.split("y"))}.png')

        # Find default isophote
        rbins = Profiles[halo][rotation]['rbins']
        Rhalf = Profiles[halo][rotation]['Reff']
        if np.isnan(Rhalf): Rhalf = Profiles[halo][rotation]['Rhalf']
        ind_eff = np.argmin(abs(rbins - Rhalf))
        v = Profiles[halo][rotation]['v_lum_den'][ind_eff]
        tol = .01
        iso = np.where((Image > v * (1 - tol)) & (Image < v * (1 + tol)))

        DC, DE = False, False  # Updated: Removed DA
        _VARS = InitializeGUI(f'Halo {halo} - {rotation}')
        drawChart()
        while True:
            event, values = _VARS['window'].read()
            if event in [sg.WIN_CLOSED, 'Exit']:
                print('Aborting Code')
                sys.exit(0)
            if event in ['Save', 'Ignore']:
                _VARS['window'].close()
                myprint(f'Saving {args.simulation} {halo}-{rotation}...', clear=True)
                if event == 'Ignore': 
                    iso = [[], []]
                ba = SavePlot(LogImage, iso, Rhalf, f'../../Figures/Images/{args.simulation}.{args.feedback}/{halo}/{halo}.{".y".join(rotation.split("y"))}.Isophote.png')
                ShapeData[halo][rotation] = ba
                pickle.dump(ShapeData, open(f'../../Data/{args.simulation}.{args.feedback}.ShapeData.pickle', 'wb'))
                Masking[halo][rotation] = False
                pickle.dump(Masking, open(f'../../Data/{args.simulation}.{args.feedback}.Masking.pickle', 'wb'))
                myprint(f'{args.simulation} {halo}-{rotation} saved.', clear=True)
                break
            if event == 'Reset':
                tol = float(values['Iso%']) / 100
                iso = np.where((Image > v * (1 - tol)) & (Image < v * (1 + tol)))
                DC, DE = False, False  # Updated: Removed DA
                updateChart(iso,DrawCircle=DC, DrawEllipse=DE)
            if event == 'Mask':
                DE = False
                if DC:
                    iso = MaskCircle(values['RadiusAdjust'], cen_x=values['CenXAdjust'],
                                     cen_y=values['CenYAdjust'], isophote=iso, mode=values['CMode'][0])
                    updateChart(iso,DrawEllipse=DE)
                mode = values['EllipseMode'][0] if values['EllipseMode'] else 'Inclusive'  # Default to 'Inclusive' if not set
                if ellipse_params:  # Check if ellipse_params is populated
                    
                    iso = MaskEllipse(ellipse_params, iso, mode)
                    updateChart(iso,DrawEllipse=DE)
                
                       
                #updateChart(DrawCircle=DC, DrawEllipse=DE)
            if event in ['RadiusAdjust', 'CenXAdjust', 'CenYAdjust'] and values['CMode'] != []:
                DC, DE = True, False
                updateChart(iso,DrawCircle=DC, DrawEllipse=DE)
            if event == 'Fit Ellipse':  # This might be adjusted or integrated with new ellipse drawing functionality
                DE = True
                updateChart(iso,DrawCircle=DC, DrawEllipse=True)
            if event == 'Draw Ellipse':
                if values['EllipseMode']:  # Check if the list is not empty
                    mode = values['EllipseMode'][0]  # Now safe to access the first element
                    drawInteractiveEllipse(mode, LogImage, iso)
                    _VARS['window']['EllipseCenterX'].update(value=ellipse_params['center'][0])
                    _VARS['window']['EllipseCenterY'].update(value=ellipse_params['center'][1])
                    _VARS['window']['EllipseWidth'].update(value=ellipse_params['width'])
                    _VARS['window']['EllipseHeight'].update(value=ellipse_params['height'])
                    _VARS['window']['EllipseAngle'].update(value=ellipse_params['angle'])
                    updateChart(iso, DrawUserEllipse=True)
                else:
                    sg.popup("Please select an ellipse mode.")
            if event == 'UpdateEllipse':
                ellipse_params = {
                    'center': (values['EllipseCenterX'], values['EllipseCenterY']),
                    'width': values['EllipseWidth'],
                    'height': values['EllipseHeight'],
                    'angle': values['EllipseAngle']
                }
                updateChart(iso, DrawUserEllipse=True)
            if event == 'Ignore Halo':
                confirmation = sg.popup_ok_cancel('Are you sure you want to ignore this halo? This action cannot be undone.', title='Confirm Ignore')
                if confirmation == 'OK':
                    # Loop through all rotations for the current halo
                    
                    for rotation in Masking[halo]:
                        # Mark the halo rotation as processed/ignored
                        Masking[halo][rotation] = False
                        # Set ShapeData for the halo rotation to np.nan
                        ShapeData[halo][rotation] = {'b/a': np.nan, 'a': np.nan, 'b': np.nan}
                
                    # Optionally, save the updated dictionaries back to the pickle files
                    # Ensure you do this only if necessary to avoid redundant I/O operations
                    pickle.dump(Masking, open(f'../../Data/{args.simulation}.{args.feedback}.Masking.pickle', 'wb'))
                    pickle.dump(ShapeData, open(f'../../Data/{args.simulation}.{args.feedback}.ShapeData.pickle', 'wb'))
                
                    # Provide feedback in the console or GUI about the ignored halo
                    print(f'Halo {halo} ignored. Moving to the next halo...')
                
                    # Implement logic to move to the next halo here
                    # This might involve breaking out of the loop, fetching the next halo data, or other steps
                    break  # Skip the rest of the loop for this iteration if appropriate
        plt.close()
        _VARS['window'].close()
print(f'No halos in {args.simulation} need masking.')
'''