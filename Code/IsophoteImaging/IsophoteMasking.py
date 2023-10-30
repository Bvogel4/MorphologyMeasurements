import argparse,pickle,sys,warnings
import numpy as np
import matplotlib.pylab as plt
from math import pi,degrees
import PySimpleGUI as sg
from numpy import sin,cos
from numpy.linalg import eig, inv
from matplotlib.patches import Circle,Ellipse
from skimage.measure import EllipseModel
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
parser.add_argument('-f','--feedback',choices=['BW','SB'],default='BW',help='Feedback Model')
parser.add_argument('-s','--simulation',choices=['cptmarvel','elektra','storm','rogue','h148','h229','h242','h329'],required=True,help='Simulation to analyze')
#parser.add_argument('-o','--overwrite',action='store_true',help='Overwrite existing images')
args = parser.parse_args()

#Masking Functions
def MaskCircle(rad,cen_x,cen_y,isophote,mode):
    assert mode in ['Inclusive','Exclusive'], 'Masking Mode Error'
    masked_iso = [[],[]]
    for i in np.arange(len(isophote[0])):
        r = np.sqrt((isophote[0][i]-500-cen_y)**2+(isophote[1][i]-500-cen_x)**2)
        if r<rad and mode=='Inclusive':
            masked_iso[0].append(isophote[0][i])
            masked_iso[1].append(isophote[1][i])
        if r>rad and mode=='Exclusive':
            masked_iso[0].append(isophote[0][i])
            masked_iso[1].append(isophote[1][i])
    masked_iso = ( np.array(masked_iso[0]),np.array(masked_iso[1]) )
    return(masked_iso)
def MaskSlice(minang,maxang,isophote,mode):
    ###NOT READY, angle shit
    assert mode in ['Inclusive','Exclusive'], 'Masking Mode Error'
    masked_iso = [[],[]]
    for i in np.arange(len(isophote[0])):
        y,x = isophote[0][i]-500,isophote[1][i]-500
        if x<0:
            if y>0: ang = degrees(np.arctan(y/x))+180
            if y<0: ang = degrees(np.arctan(y/x))-180
        else: ang = degrees(np.arctan(y/x))
        if minang<ang<maxang and mode=='Inclusive':
            masked_iso[0].append(isophote[0][i])
            masked_iso[1].append(isophote[1][i])
        if (minang>ang or maxang<ang) and mode=='Exclusive':
            masked_iso[0].append(isophote[0][i])
            masked_iso[1].append(isophote[1][i])
    masked_iso = ( np.array(masked_iso[0]),np.array(masked_iso[1]) )
    return masked_iso

#GUI Functions
def InitializeGUI(PlotName):
    #GUI Properties
    _VARS = {'window':False,'fig_agg':False,'pltFig':False}#,'a':np.nan,'b':np.nan}
    #plt.style.use('Solarize_Light2')
    GuiFont = 'Any 16'
    GuiColor = '#E8E8E8'
    sg.theme('black')
    layout = [
        [sg.Text(text=PlotName,
            font=GuiFont,
            background_color=GuiColor,
            text_color='Black')],
        [sg.Canvas(key='figCanvas',
            background_color=GuiColor)],
        [sg.Text(text='Circular Mask:',
            font=GuiFont,
            background_color=GuiColor,
            text_color='Black'),
        sg.Listbox(['Inclusive','Exclusive'],
            key='CMode'),
        sg.Text(text='Radius:',
            font=GuiFont,
            background_color=GuiColor,
            text_color='Black'),
        sg.Slider(range=(0,400), 
            orientation='h',size=(10,10),
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
        sg.Slider(range=(-500,500), 
            orientation='h',size=(10,10),
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
        sg.Slider(range=(-500,500), 
            orientation='h',size=(10,10),
            default_value=0,
            background_color=GuiColor,
            resolution=1,
            text_color='Black',
            key='CenYAdjust',
            enable_events=True)],
        [sg.Text(text='Angular Mask:',
            font=GuiFont,
            background_color=GuiColor,
            text_color='Black'),
        sg.Listbox(['Exclusive','Inclusive'],
            key='AMode'),
        sg.Text(text='Min Angle:',
            font=GuiFont,
            background_color=GuiColor,
            text_color='Black'),
        sg.Slider(range=(-180,180), 
            orientation='h',size=(17,10),
            default_value=0,
            background_color=GuiColor,
            resolution=1,
            text_color='Black',
            key='MinAngle',
            enable_events=True),
        sg.Text(text='Max Angle:',
            font=GuiFont,
            background_color=GuiColor,
            text_color='Black'),
        sg.Slider(range=(-180,180), 
            orientation='h',size=(17,10),
            default_value=0,
            background_color=GuiColor,
            resolution=1,
            text_color='Black',
            key='MaxAngle',
            enable_events=True)],
        [sg.Button('Exit',font=GuiFont),
        sg.Button('Ignore',font=GuiFont,pad=((0,130),(0,0))),
        sg.Text(text='Isophote %:',
            font=GuiFont,
            background_color=GuiColor,
            text_color='Black'),
        sg.InputText('1',
            background_color=GuiColor,
            font=GuiFont,
            size=3,
            text_color='Black',
            key='Iso%'),
        sg.Button('Reset',font=GuiFont),
        sg.Button('Mask', font=GuiFont),
        sg.Button('Fit Ellipse', font=GuiFont),
        sg.Button('Save', font=GuiFont)]
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
    _VARS['pltFig'] = plt.figure()
    plt.imshow(LogImage)
    plt.grid(b=None)
    _VARS['pltFig'].axes[0].set_xlim([0,1000])
    _VARS['pltFig'].axes[0].set_ylim([0,1000])
    plt.grid(b=None)
    plt.scatter(500,500,marker='+',s=10**2,c='w')
    plt.scatter(iso[1],iso[0],c='r',s=.5**2)
    _VARS['fig_agg'] = draw_figure(_VARS['window']['figCanvas'].TKCanvas,_VARS['pltFig'])
def updateChart(DrawCircle=False,DrawAngle=False,DrawEllipse=False):
    _VARS['fig_agg'].get_tk_widget().forget()
    plt.clf()
    plt.imshow(LogImage)
    plt.grid(b=None)
    _VARS['pltFig'].axes[0].set_xlim([0,1000])
    _VARS['pltFig'].axes[0].set_ylim([0,1000])
    plt.grid(b=None)
    plt.scatter(500,500,marker='+',s=10**2,c='w')
    plt.scatter(iso[1],iso[0],c='r',s=.5**2)
    if DrawCircle:
        _VARS['pltFig'].axes[0].add_patch(Circle((500+values['CenXAdjust'],500+values['CenYAdjust']),
                                        values['RadiusAdjust'],color='w',fill=False))
    if DrawAngle:
        plt.plot([500,500+710*np.cos(np.radians(values['MinAngle']))],
                 [500,500+710*np.sin(np.radians(values['MinAngle']))],c='w',linewidth=1)
        plt.plot([500,500+710*np.cos(np.radians(values['MaxAngle']))],
                 [500,500+710*np.sin(np.radians(values['MaxAngle']))],c='w',linewidth=1)
    if DrawEllipse:
        #Fit Ellipse to Isophote
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
        #Plot the ellipse fit on the image and set image title to axis ratio
        _VARS['pltFig'].axes[0].add_patch(Ellipse(cen,2*a,2*b,angle=degrees(phi),facecolor='None',edgecolor='orange'))
        plt.plot([-a*cos(phi)+cen[0],a*cos(phi)+cen[0]],[-a*sin(phi)+cen[1],a*sin(phi)+cen[1]],color='orange')
        plt.plot([-b*cos(phi+pi/2)+cen[0],b*cos(phi+pi/2)+cen[0]],[-b*sin(phi+pi/2)+cen[1],
                   b*sin(phi+pi/2)+cen[1]],color='orange')   
    _VARS['fig_agg'] = draw_figure(_VARS['window']['figCanvas'].TKCanvas,_VARS['pltFig'])
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


#Load sim-level data
Images = pickle.load(open(f'../../Data/{args.simulation}.{args.feedback}.Images.pickle','rb'))
ShapeData = pickle.load(open(f'../../Data/{args.simulation}.{args.feedback}.ShapeData.pickle','rb'))
Masking = pickle.load(open(f'../../Data/{args.simulation}.{args.feedback}.Masking.pickle','rb'))
Profiles = pickle.load(open(f'../../Data/{args.simulation}.{args.feedback}.Profiles.pickle','rb'))
#test = pickle.load(open(f'../../Data/{args.simulation}.{args.feedback}.ShapeData.pickle','rb'))

for halo in Masking:
    for rotation in Masking[halo]:
        if Masking[halo][rotation]:
            print(f'Masking {args.simulation} {halo}-{rotation}...')
            #Load halo-level data
            Image = Images[halo][rotation]
            Image = np.flip(Image,0)
            LogImage = plt.imread(f'../../Figures/Images/{args.simulation}.{args.feedback}/{halo}/{halo}.{".y".join(rotation.split("y"))}.png')

            #Find default isophote
            rbins = Profiles[halo][rotation]['rbins']
            Rhalf = Profiles[halo][rotation]['Reff']
            if np.isnan(Rhalf): Rhalf = Profiles[halo][rotation]['Rhalf']
            ind_eff = np.argmin(abs(rbins-Rhalf))
            v = Profiles[halo][rotation]['v_lum_den'][ind_eff]
            tol = .01
            iso = np.where((Image>v*(1-tol)) & (Image<v*(1+tol)))

            DC,DA,DE = False,False,False
            _VARS = InitializeGUI(f'Halo {halo} - {rotation}')
            drawChart()
            while True:
                event, values = _VARS['window'].read()
                if event in [sg.WIN_CLOSED,'Exit']:
                    print('Aborting Code')
                    sys.exit(0)
                if event in ['Save','Ignore']:
                    _VARS['window'].close()
                    myprint(f'Saving {args.simulation} {halo}-{rotation}...',clear=True)
                    if event=='Ignore':iso=[[],[]]
                    ba = SavePlot(LogImage,iso,Rhalf,f'../../Figures/Images/{args.simulation}.{args.feedback}/{halo}/{halo}.{".y".join(rotation.split("y"))}.Isophote.png')
                    ShapeData[halo][rotation] = ba
                    pickle.dump(ShapeData,open(f'../../Data/{args.simulation}.{args.feedback}.ShapeData.pickle','wb'))
                    Masking[halo][rotation] = False
                    pickle.dump(Masking,open(f'../../Data/{args.simulation}.{args.feedback}.Masking.pickle','wb'))
                    myprint(f'{args.simulation} {halo}-{rotation} saved.',clear=True)
                    break
                if event=='Reset':
                    tol = float(values['Iso%'])/100
                    iso = np.where((Image>v*(1-tol)) & (Image<v*(1+tol)))
                    DC,DA,DE = False,False,False
                    updateChart(DrawCircle=DC,DrawAngle=DA,DrawEllipse=DE)
                if event=='Mask':
                    DE = False
                    if DC:
                        iso = MaskCircle(values['RadiusAdjust'],cen_x=values['CenXAdjust'],
                                cen_y=values['CenYAdjust'],isophote=iso,mode=values['CMode'][0])
                    if DA:
                        iso = MaskSlice(values['MinAngle'],values['MaxAngle'],
                                isophote=iso,mode=values['AMode'][0])
                    updateChart(DrawCircle=DC,DrawAngle=DA,DrawEllipse=DE)
                if event in ['RadiusAdjust','CenXAdjust','CenYAdjust'] and values['CMode']!=[]:
                    DC,DE = True,False
                    updateChart(DrawCircle=DC,DrawAngle=DA,DrawEllipse=DE)
                if event in ['MinAngle','MaxAngle'] and values['AMode']!=[]:
                    DA,DE = True,False
                    updateChart(DrawCircle=DC,DrawAngle=DA,DrawEllipse=DE)
                if event=='Fit Ellipse':
                    DE = True
                    updateChart(DrawCircle=DC,DrawAngle=DA,DrawEllipse=True)
            plt.close()
            _VARS['window'].close()

print(f'No halos in {args.simulation} need masking.')