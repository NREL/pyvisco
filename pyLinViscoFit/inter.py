"""Collection of classes and functions to create a graphical user interface
based on ipywidgets and Jupyter notebook.
"""

import pandas as pd
import ipywidgets as widgets
import io
import zipfile

from IPython.display import display, clear_output, HTML
from base64 import b64encode

from . import load
from . import shift
from . import master
from . import prony
from . import opt
from . import styles
from . import out

"""
--------------------------------------------------------------------------------
Convenience classes and methods
--------------------------------------------------------------------------------
"""

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
bcolors = bcolors()


def generate_zip(files):
    mem_zip = io.BytesIO()

    with zipfile.ZipFile(mem_zip, mode="w",compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            pre = name.split(sep='_')[0]
            if pre == 'df':
                fname = name + '.csv'
            elif pre == 'fig':
                fname = name + '.png'
                data = fig_bytes(data)
            else:
                fname = name
            zf.writestr(fname, data)

    return mem_zip.getvalue()


def fig_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, dpi = 600)
    return buf.getvalue()


"""
--------------------------------------------------------------------------------
Widget class for Jupyter dashboard
--------------------------------------------------------------------------------
"""

class Widgets():
    """
    Create widgets for GUI
    """
    def __init__(self):
        self.ini_variables()
        self.widgets()
        styles.format_fig()
        styles.format_HTML(self.out_html)
        self.modul = 'E'


    def ini_variables(self):
        self.RefT = 0
        self.nprony = 0


    def widgets(self):
        """
        Define GUI widgets.
        """      
        _height = 'auto'
        _width = 'auto'
        _width_b = '200px'
        _layout = {'width' : '100%', 'justify_content' : 'space-between'}

        """
        ------------------------------------------------------------------------
        Section - Overview
        ------------------------------------------------------------------------
        """
        #Theory button
        self.b_theory = widgets.ToggleButton(
            value=False,
            description='Click here for more details!',
            layout = widgets.Layout(width = '200px'))
        self.b_theory.observe(self.show_theory, 'value')

        #Theory out
        self.out_theory = widgets.HTMLMath(value='')
        

        """
        ------------------------------------------------------------------------
        Subsection - Specify input and upload data
        ------------------------------------------------------------------------
        """
        #Modulus data 
        #-----------------------------------------------------------------------
        #Domain
        self.rb_domain = widgets.RadioButtons(
            options=['freq', 'time'],
            value='freq', 
            description='Domain:',
            disabled=False,
            layout = widgets.Layout(height = _height, width = _width))
        self.rb_domain.observe(self.set_domain, 'value')

        #Loading direction
        self.rb_loading = widgets.RadioButtons(
            options=['tensile', 'shear'],
            value='tensile', 
            description='Loading:',
            disabled=False,
            layout = widgets.Layout(height = _height, width = _width))
        self.rb_loading.observe(self.set_loading, 'value')

        #Instrument
        self.rb_instrument = widgets.RadioButtons(
            options=['Eplexor', 'user'],
            value='Eplexor', 
            description='Instrument:',
            disabled=False,
            layout = widgets.Layout(height = _height, width = _width))
        self.rb_instrument.observe(self.set_instrument, 'value')

        #Type
        self.rb_type = widgets.RadioButtons(
            options=['master', 'raw'],
            value='master',
            description='Type:',
            disabled=False,
            layout = widgets.Layout(height = _height, width = _width))
        self.rb_type.observe(self.set_type, 'value')

        #Upload modulus data
        self.up_inp = widgets.FileUpload(
            accept= '.xls, .xlsx',
            multiple=False,
            button_style='success',
            layout = widgets.Layout(height = _height, width = _width_b))
        self.up_inp.observe(self.inter_load_modul, names='_counter')

        #Out modulus data
        self.out_load_modul = widgets.Output()

        #Layout
        _inp_gen = widgets.HBox([self.rb_domain, self.rb_loading, self.rb_instrument,
            self.rb_type, self.up_inp,], layout = widgets.Layout(**_layout))
        self.w_inp_gen = widgets.VBox([_inp_gen, self.out_load_modul],
            layout = widgets.Layout(**_layout))


        #Shift factor data 
        #-----------------------------------------------------------------------
        #User shift factors
        self.cb_shift = widgets.Checkbox(
            value=False, 
            description='user shift factors',
            disabled=False,
            indent = True,
            layout = widgets.Layout(height = _height, width = _width),
            style = {'description_width' : 'initial'})
        self.cb_shift.observe(self.set_shift, 'value')

        #Upload shift factor
        self.up_shift = widgets.FileUpload(
            accept='.csv, .xls', 
            multiple=False, 
            disabled=True,
            button_style='success',
            layout = widgets.Layout(height = _height, width = _width_b))
        self.up_shift.observe(self.inter_load_shift, names='_counter')

        #Out shift factor
        self.out_load_shift = widgets.Output()

        #Layout
        _inp_shift = widgets.HBox([self.cb_shift, self.up_shift],
            layout = widgets.Layout(**_layout))
        self.w_inp_shift = widgets.VBox([_inp_shift, self.out_load_shift],
            layout = widgets.Layout(**_layout))


        #Reference temperature
        #-----------------------------------------------------------------------
        #Reference Temperature - FloatText
        self.ft_RefT = widgets.FloatText(
            value=self.RefT,
            description='Reference temperature (\N{DEGREE SIGN}C):',
            disabled=True,
            layout = widgets.Layout(height = _height, width = '250px'),
            style = {'description_width' : 'initial'})
        self.ft_RefT.observe(self.set_RefT, 'value')


        #Reference Temperature - Dropdown
        self.dd_RefT = widgets.Dropdown(
            description='Reference temperature (\N{DEGREE SIGN}C):',
            disabled=False,
            layout = widgets.Layout(height = _height, width = '250px'),
            style = {'description_width' : 'initial'})
        self.ft_RefT.observe(self.set_RefT, 'value')

        #Layout
        self.w_RefT = widgets.HBox([self.ft_RefT])


        #Control section
        #-----------------------------------------------------------------------
        self.v_modulus = widgets.Valid(
            value=False,
            description='Modul',
            continuous_update=True,
            readout = '', #string.whitespace
            layout = widgets.Layout(height = _height, width = _width_b))

        self.v_aT = widgets.Valid(
            value=False,
            description='Shift factors',
            continuous_update=True,
            readout = '', #string.whitespace
            layout = widgets.Layout(height = _height, width = _width_b))

        self.v_WLF = widgets.Valid(
            value=False,
            description='WLF shift function',
            continuous_update=True,
            readout = '', #string.whitespace
            style = {'description_width' : 'initial'},
            layout = widgets.Layout(height = _height, width = _width_b))

        #Layout
        _valid = widgets.HBox([self.v_modulus, self.v_aT, self.v_WLF], 
            layout = widgets.Layout(width = '100%'))
        self.w_check_inp = widgets.VBox([_valid],
            layout = widgets.Layout(width = '100%')) 


        """
        ------------------------------------------------------------------------
        Subsection - Time-temperature superposition
        ------------------------------------------------------------------------
        """
        #Shift factors
        #-----------------------------------------------------------------------
        #Fit and plot shift factors
        self.b_aT = widgets.Button(
            description='master raw data',
            button_style='info',
            disabled = True,
            layout = widgets.Layout(height = _height, width = _width_b))
        self.b_aT.on_click(self.inter_aT)

        #Overwrite shift factors
        self.cb_aT = widgets.Checkbox(
            value=False, 
            description='fit and overwrite provided shift factors',
            disabled=True,
            indent = True,
            layout = widgets.Layout(height = _height, width = _width),
            style = {'description_width' : 'initial'})

        #Allow manual shifting of log_aT
        self.cb_ManShift = widgets.Checkbox(
            value=False, 
            description='manually adjust shift factors',
            disabled=True,
            indent = True,
            layout = widgets.Layout(height = _height, width = _width),
            style = {'description_width' : 'initial'})
        self.cb_ManShift.observe(self.show_manual_shift, 'value')

        #Out shift factors
        self.out_aT = widgets.Output()
        self.out_aT_man = widgets.Output()

        #Layout
        _aT = widgets.HBox([self.b_aT, self.cb_aT, self.cb_ManShift])
        self.w_aT = widgets.VBox([_aT, self.out_aT, self.out_aT_man])


        #Shift functions
        #-----------------------------------------------------------------------
        #Overwrite WLF shift function coefficients
        self.cb_WLF = widgets.Checkbox(
            value=False, 
            description='fit and overwrite provided WLF',
            disabled=True,
            indent = True,
            layout = widgets.Layout(height = _height, width = _width),
            style = {'description_width' : 'initial'})

        #Fit and plot shift functions
        self.b_shift = widgets.Button(
            description='(fit) & plot shift functions',
            button_style='info',
            disabled = True,
            layout = widgets.Layout(height = _height, width = _width_b))
        self.b_shift.on_click(self.inter_shift)

        #Out shift function
        self.out_shift = widgets.Output()

        #Layout
        _shift = widgets.HBox([self.b_shift, self.cb_WLF])
        self.w_shift = widgets.VBox([_shift, self.out_shift])

        """
        ------------------------------------------------------------------------
        Subsection - Estimate Prony series parameters
        ------------------------------------------------------------------------
        """
        #Filter master curve
        #-----------------------------------------------------------------------
        #Smooth
        self.b_smooth = widgets.Button(
            description='smooth master curve',
            button_style='info',
            layout = widgets.Layout(height = _height, width = _width_b))
        self.b_smooth.on_click(self.inter_smooth_fig)

        #Out smooth
        self.out_smooth = widgets.Output()

        #Layout
        self.w_smooth = widgets.VBox([self.b_smooth, self.out_smooth])


        #Discretization
        #-----------------------------------------------------------------------
        #Discretization type
        self.rb_dis = widgets.RadioButtons(
            options=['default', 'manual'],
            value='default', 
            description='Discretization:',
            disabled=False,
            style = {'description_width' : 'initial'},
            layout = widgets.Layout(height = _height, width = _width))
        self.rb_dis.observe(self.set_dis, 'value')

        #Discretization window
        self.rb_dis_win = widgets.RadioButtons(
            options=['round', 'exact'],
            value='round', 
            description='Window:',
            disabled=True,
            layout = widgets.Layout(height = _height, width = _width))

        #Discretization number of Prony terms
        self.it_nprony = widgets.BoundedIntText(
            value=self.nprony,
            min = 0,
            max = 1000,
            step = 1,
            description='Number of Prony terms:',
            disabled=True,
            layout = widgets.Layout(height = _height, width = '220px'),
            style = {'description_width' : 'initial'})

        #Plot discretization
        self.b_dis = widgets.Button(
            description='plot discretization',
            button_style='info', 
            layout = widgets.Layout(height = _height, width = _width_b))
        self.b_dis.on_click(self.inter_dis)

        #Out discretization
        self.out_dis = widgets.Output()

        #Layout
        _dis = widgets.HBox([self.rb_dis, self.rb_dis_win, self.it_nprony],  
            layout = widgets.Layout(**_layout))
        self.w_dis = widgets.VBox([_dis, self.b_dis, self.out_dis])


        #Fit Prony parameter
        #-----------------------------------------------------------------------
        #Prony fit
        self.b_fit = widgets.Button(
            description='fit Prony series',
            button_style='info',
            layout = widgets.Layout(height = _height, width = _width_b))
        self.b_fit.on_click(self.inter_fit)

        #Out Prony
        self.out_fit = widgets.Output()
        self.out_prony = widgets.Output()

        #Layout
        self.w_out_fit_prony = widgets.HBox([self.out_fit, self.out_prony],  
            layout = widgets.Layout(width = '100%')) 


        #Generalized Maxwell model
        #-----------------------------------------------------------------------
        #Plot Generalized Maxwell
        self.b_GMaxw = widgets.Button(
            description='plot Generalized Maxwell',
            button_style='info',
            layout = widgets.Layout(height = _height, width = _width_b))
        self.b_GMaxw.on_click(self.inter_GMaxw)
        
        #Out Generalized Maxwell
        self.out_GMaxw_freq = widgets.Output()
        self.out_GMaxw_temp = widgets.Output()

        #Layout
        self.w_out_GMaxw = widgets.HBox([self.out_GMaxw_freq, self.out_GMaxw_temp],  
            layout = widgets.Layout(width = '100%')) 

        """
        ------------------------------------------------------------------------
        Subsection - Optional minimization routine
        ------------------------------------------------------------------------
        """
        #Minimize number of Prony terms
        self.b_opt = widgets.Button(
            description='minimize Prony terms',
            button_style='warning',
            layout = widgets.Layout(height = _height, width = _width_b))
        self.b_opt.on_click(self.inter_opt)
        
        #Out Minimization
        self.out_dro = widgets.Output()
        self.out_opt = widgets.Output()
        self.out_res = widgets.Output()
        self.out_par = widgets.Output()

        #Layout
        _minProny = widgets.HBox([self.out_opt, self.out_res],  
            layout = widgets.Layout(width = '100%')) 
        self.w_out_fit_min = widgets.VBox([self.out_dro, _minProny, self.out_par])


        """
        ------------------------------------------------------------------------
        Subsections - Download & Reload
        ------------------------------------------------------------------------
        """
        #Download zip
        self.db_zip = widgets.Button(
            description='Download zip',
            button_style='success',
            layout = widgets.Layout(height = _height, width = _width_b))
        self.db_zip.on_click(self.down_zip)
        
        #Reload notebook
        self.b_reload = widgets.Button(
            description='Clear notebook!',
            button_style='danger',
            layout = widgets.Layout(height = 'auto', width = _width_b))
        self.b_reload.on_click(self.reload)           
 
        #Output widgets for HTML content
        self.out_html = widgets.Output()


    """
    ----------------------------------------------------------------------------
    Optional section - Manual shifting
    ----------------------------------------------------------------------------
    """
    def show_manual_shift(self, change):
        _layout = {'width' : '100%', 'justify_content' : 'space-between'}

        #Step size
        self.inp_step = widgets.Dropdown(
            options=['coarse', 'medium', 'fine'], 
            value = 'coarse',
            description='Step size:',
            disabled=False,
            layout = widgets.Layout(width = '200px'),
            style = {'description_width' : 'initial'}) 
        self.inp_step.observe(self.set_inp_step, 'value')

        #Temperature
        self.inp_T = widgets.Dropdown(
            options=self.df_aT['T'], 
            value = self.df_aT['T'].iloc[0],
            description='Temperature (\N{DEGREE SIGN}C):',
            disabled=False,
            layout = widgets.Layout(width = '200px'),
            style = {'description_width' : 'initial'})
        self.inp_T.observe(self.set_inp_T, 'value')

        #Shift factor
        self.inp_aT = widgets.FloatText(
            step=0.5,
            value = self.get_aT(self.df_aT['T'].iloc[0]),
            description='log(a_T):',
            disabled=False,
            continuous_update=False,
            layout = widgets.Layout(width = '250px'),
            style = {'description_width' : 'initial'})
        self.inp_aT.observe(self.set_inp_aT, 'value')

        #Shift single set
        self.cb_single = widgets.Checkbox(
            value=False, 
            description='shift single set',
            disabled=False,
            indent = False,
            layout = widgets.Layout(width = '200px'),
            style = {'description_width' : 'initial'})

        #Layout
        self.w_inp_man = widgets.HBox([self.inp_step, self.inp_T, self.inp_aT,
            self.cb_single], layout = widgets.Layout(**_layout))

        #Display optional section
        with self.out_aT_man:
            if change['new'] == True:
                self.df_aT_ref = self.df_aT.copy()
                clear_output()
                display(self.w_inp_man)
            else:
                self.reset_df_aT()
                clear_output()


"""
--------------------------------------------------------------------------------
Control class for Jupyter dashboard
--------------------------------------------------------------------------------
"""

class Control(Widgets):
    """
    Collection of methods to provide interactive functionality for the Jupyter
    Notebook.
    """
    def __init__(self):
        super().__init__()
        self.collect_files()
        self.create_loading_bar()

        with open('./theory.html') as file:  
            self.html_theory = file.read() 


    def collect_files(self):
        """
        Create dictionary to collect output files.
        """
        self.files = {}


    def create_loading_bar(self):
        """
        Create loading bar object for methods with longer runtime.
        """
        gif_address = './figures/loading.gif'
        with open(gif_address, 'rb') as f:
            img = f.read()
        self.w_loading = widgets.Image(value=img)


    def reset_notebook(self):
        """
        Reset notebook configuration and clear data.
        """
        #Clear output widgets
        with self.out_load_modul:
            clear_output()
        with self.out_load_shift:
            clear_output()
        with self.out_aT:
            clear_output()
        with self.out_aT_man:
            clear_output()
        with self.out_shift:
            clear_output()
        with self.out_smooth:
            clear_output()
        with self.out_dis:
            clear_output()
        with self.out_fit:
            clear_output()
        with self.out_prony:
            clear_output()
        with self.out_GMaxw_freq:
            clear_output()
        with self.out_GMaxw_temp:
            clear_output()
        with self.out_opt:
            clear_output()
        with self.out_res:
            clear_output()
        with self.out_dro:
            clear_output()
        with self.out_par:
            clear_output()

        #Set default widget configuration
        self.cb_ManShift.value = False
        self.cb_ManShift.disabled = True
        self.v_modulus.value = False
        self.v_aT.value = False
        self.v_WLF.value = False
        self.cb_shift.value = False
        self.ft_RefT.disabled = False
        self.cb_aT.disabled = True
        self.cb_WLF.value = False
        self.cb_WLF.disabled = True
        self.b_shift.disabled = True

        #Clear data
        self.df_master = None
        self.df_raw = None
        self.df_dis = None
        self.df_aT = None
        self.df_WLF = None
        self.df_poly = None
        self.df_GMaxw = None
        self.df_GMaxw_temp = None
        self.df_GMaxw_opt = None
        self.units = None
        self.prony = None
        self.dict_prony = None
        self.arr_RefT = None
        self.collect_files()

    def exceptions(func):
        """
        Wrapper method providing exception handling for methods loading files.
        """
        def wrap(self, *args):
            try:
                if func.__name__ == 'inter_load_modul':
                    _out = self.out_load_modul
                elif func.__name__ == 'inter_load_shift':
                    _out = self.out_load_shift
                
                func(self, *args)

            except KeyError as e:
                with _out:
                    _msg = 'Input file header not as expected, check conventions!'
                    print(f'{bcolors.FAIL}' + _msg + f'{bcolors.ENDC}')
                    _msg = '<-- missing or unkown!'
                    print(str(e).replace('not found in axis', 
                        f'{bcolors.FAIL}' + _msg + f'{bcolors.ENDC}'))
            except ValueError as e:
                with _out:
                    if str(e).split(',')[0] == 'Temperatures':
                        _msg = 'Temperatures of user shift factors and modulus data need to be identical!'
                        print(f'{bcolors.FAIL}' + _msg + f'{bcolors.ENDC}')
                        print('T[modulus]: {}'.format(str(e).split(',')[1]))
                        print('T[log(aT)]: {}'.format(str(e).split(',')[2]))
                    else:
                        _msg = 'Wrong file format, check conventions!'
                        print(f'{bcolors.FAIL}' + _msg + f'{bcolors.ENDC}')
            except AttributeError as e:
                with _out:
                    _msg = 'Upload modul data before uploading shift factors!'
                    print(f'{bcolors.FAIL}' + _msg + f'{bcolors.ENDC}')
        return wrap
       
    """
    ----------------------------------------------------------------------------
    Section - Overview
    ----------------------------------------------------------------------------
    """
    def show_theory(self, change):
        """
        Show theory section from HTML file.
        """
        if change['new'] == True:
            self.out_theory.value = self.html_theory
        elif change ['new'] == False:
            self.out_theory.value = ''

    """
    ----------------------------------------------------------------------------
    Subsection - Specify input and upload data
    ----------------------------------------------------------------------------
    """
    #Modulus data 
    #---------------------------------------------------------------------------
    def set_domain(self, change):
        """
        Set measurement domain and update widgets.
        """
        with self.out_load_modul:
            clear_output()
        if change['new'] == 'freq':
            self.rb_instrument.disabled = False
        elif change ['new'] == 'time':
            self.rb_instrument.value = 'user'
            self.rb_instrument.disabled = True


    def set_loading(self, change):
        """
        Set modulus based on loading direction and update widgets.
        """
        with self.out_load_modul:
            clear_output()
        if change['new'] == 'tensile':
            self.modul = 'E'
        elif change['new'] == 'shear':
            self.modul = 'G'


    def set_instrument(self, change):
        """
        Set instrument type and update widgets.
        """
        with self.out_load_modul:
            clear_output()
        if change['new'] == 'Eplexor':
            self.up_inp.accept='.xls, .xlsx'
        elif change ['new'] == 'user':
            self.up_inp.accept='.csv'


    def set_type(self, change):
        """
        Set type of modulus data and update widgets.
        """
        with self.out_load_modul:
            clear_output()
        if change['new'] == 'raw':
            self.b_aT.disabled = False
        else:
            self.b_aT.disabled = True


    @exceptions
    def inter_load_modul(self, b):
        """
        Execute interactive routine to load modulus data from file.
        """
        self.reset_notebook()

        #Load modulus
        if self.rb_instrument.value == 'Eplexor':
            if self.rb_type.value == 'master':
                _epl  = load.Eplexor_master(self.up_inp.data[0], self.modul)
                self.df_master, self.df_aT, self.df_WLF, self.units = _epl
                self.set_RefT(self.df_master.RefT)
                self.ft_RefT.disabled = True
            elif self.rb_type.value == 'raw':
                self.df_raw, self.arr_RefT, self.units = load.Eplexor_raw(
                    self.up_inp.data[0], self.modul)
                self.set_RefT(self.ft_RefT.value)
        elif self.rb_instrument.value == 'user':
            if self.rb_type.value == 'master':
                _master = load.user_master(self.up_inp.data[0], 
                    self.rb_domain.value, self.RefT, self.modul)
                self.df_master, self.units = _master
                self.set_RefT(0)
                self.ft_RefT.disabled = False
            elif self.rb_type.value == 'raw':
                self.df_raw, self.arr_RefT, self.units = load.user_raw(
                    self.up_inp.data[0], self.rb_domain.value, self.modul)
                self.set_RefT(self.ft_RefT.value)

        #Add data to file package and update widgets
        if isinstance(self.df_master, pd.DataFrame):             
            self.files['df_master'] = out.to_csv(self.df_master, self.units)
            self.v_modulus.value = True
        if isinstance(self.df_raw, pd.DataFrame):             
            self.v_modulus.value = True
        if isinstance(self.df_aT, pd.DataFrame):
            self.files['df_aT'] = out.to_csv(self.df_aT, self.units)
            self.v_aT.value = True
            self.b_shift.disabled = False
            if isinstance(self.df_raw, pd.DataFrame):   
                self.cb_aT.disabled = False
        if isinstance(self.df_WLF, pd.DataFrame):
            self.files['df_shift_WLF_Eplexor'] = out.to_csv(self.df_WLF, 
                self.units) 
            self.v_WLF.value = True
            self.cb_WLF.disabled = False

        #Indicate succesful upload
        with self.out_load_modul:
            _msg = 'Upload successful!'
            print(f'{bcolors.OKGREEN}' + _msg + f'{bcolors.ENDC}')


    #Shift factor data 
    #---------------------------------------------------------------------------
    def set_shift(self, change):
        """
        Set wether user shift factors are provided and update widgets.
        """
        with self.out_load_shift:
            clear_output()
        if change['new']:
            self.up_shift.disabled = False
        else:
            self.up_shift.disabled = True


    @exceptions
    def inter_load_shift(self, b):
        """
        Execute interactive routine to load shift factors from file.
        """
        with self.out_load_shift:
            clear_output()

        #Load shift factors
        if self.cb_shift.value:
            self.df_aT = load.user_shift(self.up_shift.data[0])
            if isinstance(self.arr_RefT, pd.Series): 
                _T_shift = self.df_aT['T'].sort_values(
                    ignore_index=True).to_numpy(dtype=float)
                _T_modulus = self.arr_RefT.sort_values(
                    ignore_index=True).to_numpy(dtype=float)
                if not all(_T_shift == _T_modulus):
                    self.df_aT = None
                    raise ValueError('Temperatures,' + str(_T_modulus) 
                        + ','+ str(_T_shift))

        #Add data to file package and update widgets
        if isinstance(self.df_aT, pd.DataFrame):
            self.files['df_aT'] = out.to_csv(self.df_aT, self.units)
            self.v_aT.value = True
            self.b_shift.disabled = False
            if isinstance(self.df_raw, pd.DataFrame):   
                self.cb_aT.disabled = False

        #Indicate succesful upload
        with self.out_load_shift:
            _msg = 'Upload successful!'
            print(f'{bcolors.OKGREEN}' + _msg + f'{bcolors.ENDC}')


    #Reference temperature
    #-----------------------------------------------------------------------
    def set_RefT(self, change):
        """
        Set reference temperature and update widgets.
        """
        if isinstance(change, dict):
            _RefT = change['new']
        else:
            _RefT = change

        if isinstance(self.arr_RefT, pd.Series): 
            self.w_RefT.children = [self.dd_RefT]
            self.RefT = self.arr_RefT.iloc[(
                self.arr_RefT - _RefT).abs().argsort()[:1]].values[0]
            self.dd_RefT.options = self.arr_RefT.values
            self.dd_RefT.value = self.RefT
        else:
            self.w_RefT.children = [self.ft_RefT]
            self.RefT = _RefT
            self.ft_RefT.value = self.RefT
            
    """
    ----------------------------------------------------------------------------
    Subsection - Estimate Prony series parameters
    ----------------------------------------------------------------------------
    """
    #Shift factors
    #---------------------------------------------------------------------------
    def inter_aT(self,b):
        """
        Execute interactive routine to fit shift factors.
        """
        with self.out_aT:
            clear_output()
            display(self.w_loading)
            try:
                #Fit shift factors if not present or overwrite
                if not isinstance(self.df_aT, pd.DataFrame) or self.cb_aT.value:
                    self.df_aT = master.get_aT(self.df_raw, self.RefT)

                #Assembly master curve
                self.df_master = master.get_curve(self.df_raw, self.df_aT, self.RefT)

                #Plot figure
                clear_output()
                self.fig_master_shift, self.fig_master_shift_lax = master.plot_shift(
                    self.df_raw, self.df_master, self.units)

                #Add data to file package 
                self.files['fig_master_shift'] = self.fig_master_shift
                self.files['df_master'] = out.to_csv(self.df_master, self.units)
                self.files['df_aT'] = out.to_csv(self.df_aT, self.units)

                #Update widgets
                self.b_shift.disabled = False
                self.cb_ManShift.disabled = False
            except NameError:
                    print('Raw and/or master dataframes are missing!')


    #Shift functions
    #---------------------------------------------------------------------------
    def inter_shift(self, b):
        """
        Execute interactive routine to obtaine shift functions.
        """
        with self.out_shift:
            clear_output()
            #Fit WLF shift function if not present or overwrite
            if not isinstance(self.df_WLF, pd.DataFrame) or self.cb_WLF.value:
                self.df_WLF = shift.fit_WLF(self.df_master.RefT, self.df_aT)

            #Fit Polynomial shift functions
            self.df_poly_C, self.df_poly_K = shift.fit_poly(self.df_aT)

            #Plot figure
            self.fig_shift, self.df_shift = shift.plot(self.df_aT, self.df_WLF, 
                self.df_poly_C)
            
        #Add data to file package 
        self.files['fig_shift'] = self.fig_shift
        self.files['df_shift_poly_Celsius'] = self.df_poly_C.to_csv()
        self.files['df_shift_poly_Kelvin'] = self.df_poly_K.to_csv()
        self.files['df_shift_WLF'] = out.to_csv(self.df_WLF, self.units)
        #Optional figure data:
        #self.files['df_shift_plot'] = out.to_csv(self.df_shift, self.units)


    #Smooth master curve
    #---------------------------------------------------------------------------
    def inter_smooth_fig(self, b):
        """
        Create interactive figure for smoothing routine.
        """
        with self.out_smooth:
            clear_output()
            widgets.interact(self.inter_smooth, 
                win=widgets.IntSlider(min=1, max=20, step=1, value=1, 
                    description = 'Window size:',
                    style = {'description_width' : 'initial'},
                    layout = widgets.Layout(height = 'auto', width = '300px'),
                    continuous_update=False))


    def inter_smooth(self, win):
        """
        Execute the interactive smoothing routine of the master curve.
        """
        try:
            #Smooth master curve
            self.df_master = master.smooth(self.df_master, win)

            #Plot figure
            self.fig_smooth = master.plot_smooth(self.df_master, self.units)
        
            #Add data to file package 
            self.files['df_master'] = out.to_csv(self.df_master, self.units)
            self.files['fig_smooth'] = self.fig_smooth
        except AttributeError:
            print('Upload or create master curve!')


    #Discretization
    #---------------------------------------------------------------------------
    def set_dis(self, change):
        """
        Set discretization parameters and update widgets.
        """
        if change['new'] == 'default':
            self.rb_dis_win.disabled = True
            self.it_nprony.disabled = True
            self.rb_dis_win.value = 'round'
            self.it_nprony.value = 0
        elif change['new'] == 'manual':
            self.rb_dis_win.disabled = False
            self.it_nprony.disabled = False


    def inter_dis(self, b):
        """
        Execute the interactive discretization routine.
        """
        with self.out_dis:
            clear_output()
            try:
                #Discretize
                if self.rb_dis.value == 'default':
                    self.it_nprony.value = 0
                self.df_dis = prony.discretize(self.df_master, 
                    self.rb_dis_win.value, self.it_nprony.value)

                #Plot figure 
                self.fig_dis = prony.plot_dis(self.df_master, self.df_dis, self.units)

                #Update widget
                self.it_nprony.value = self.df_dis.nprony

                #Add data to file package 
                self.files['df_dis'] = out.to_csv(self.df_dis, self.units,
                    index_label='i')
                self.files['fig_dis'] = self.fig_dis
            except (AttributeError, KeyError):
                  print('Smooth master curve before discretzation (win=1 -> no filter).')
            

    #Fit Prony terms
    #---------------------------------------------------------------------------
    def inter_fit(self, b):
        """
        Execute the interactive Prony parameter fitting routine.
        """
        with self.out_prony:
            clear_output()
        with self.out_fit:
            try:
                clear_output()
                display(self.w_loading)

                #Perform curve fitting
                if self.rb_domain.value == 'freq':
                    self.prony = prony.fit_freq(self.df_dis)
                elif self.rb_domain.value == 'time':
                    self.prony = prony.fit_time(self.df_dis, self.df_master)

                #Calculate Generalized Maxwell model
                self.df_GMaxw = prony.calc_GMaxw(**self.prony)
            
                #Plot figure
                clear_output()
                self.fig_fit = prony.plot_fit(self.df_master, self.df_GMaxw, self.units)
                self.files['fig_fit'] = self.fig_fit
                self.files['df_GMaxw'] = out.to_csv(self.df_GMaxw, self.units)
                self.files['df_prony'] = out.to_csv(self.prony['df_terms'], 
                    self.units, index_label = 'i')
            except AttributeError:
                clear_output()
                print('Discretization of master curve is missing!')
                return

        with self.out_prony:
            #Plot Prony terms next to figure
            clear_output()
            print('{}_0 = {:.2f} {}'.format(self.modul, 
                self.prony['E_0'], self.units['{}_0'.format(self.modul)]))
            print(self.prony['df_terms'][['tau_i', 'alpha_i']])


    #Calculate Generalized Maxwell model
    #---------------------------------------------------------------------------
    def inter_GMaxw(self, b):
        """
        Execute interactive routine to calculate and plot the Generalized
        Maxwell model. 
        """
        with self.out_GMaxw_freq:
            clear_output()
            try:
                #Plot figure
                self.fig_GMaxw = prony.plot_GMaxw(self.df_GMaxw, self.units)

                #Add data to file package
                self.files['fig_GMaxw'] = self.fig_GMaxw
            except AttributeError:
                print('Prony series parameters are missing!')
                return

        with self.out_GMaxw_temp:
            clear_output()
            if isinstance(self.df_WLF, pd.DataFrame):
                #Calculate temperature dependence
                self.df_GMaxw_temp = prony.GMaxw_temp('WLF', self.df_GMaxw, 
                    self.df_WLF, self.df_aT)

                #Plot figure
                self.fig_GMaxw_temp = prony.plot_GMaxw_temp(self.df_GMaxw_temp, 
                    self.units)

                #Add data to file package
                self.files['df_GMaxw_temp'] = out.to_csv(self.df_GMaxw_temp, self.units)
                self.files['fig_GMaxw_temp'] = self.fig_GMaxw_temp


    #Minimize number of Prony terms
    #---------------------------------------------------------------------------
    def inter_opt_fig(self, N):
        """
        Create interactive figure for optimization routine.
        """
        with self.out_opt:
            clear_output()
            #Plot optimized model fit
            self.df_GMaxw_opt, self.fig_opt = opt.plot_fit(self.df_master, 
                self.dict_prony, N, self.units)
        with self.out_par:
            clear_output()
            #Plot comparison of Prony parameters
            self.fig_coeff = prony.plot_param([self.prony, self.dict_prony[N]], ['initial', 'minimized'])

        #Add data to file package
        self.files['df_prony_min'] = out.to_csv(self.dict_prony[N]['df_terms'], 
            self.units, index_label = 'i')
        self.files['df_GMaxw_min'] = out.to_csv(self.df_GMaxw_opt, self.units)
        self.files['fig_fit_min'] = self.fig_opt
        self.files['fig_coeff'] = self.fig_coeff
        

    def inter_opt(self, b):
        """
        Execute the interactive optimization routine.
        """
        with self.out_res:
            clear_output()
        with self.out_par:
            clear_output()
        with self.out_opt:
            clear_output()
            try:
                display(self.w_loading)
                #Optimize number of Prony terms
                self.dict_prony, self.N_opt, self.N_opt_err = opt.nprony(
                    self.df_master, self.prony, window='min')
                clear_output()
            except (AttributeError, TypeError):
                clear_output()
                print('Initial Prony series fit is missing!')
                return
        with self.out_dro:
            clear_output()
            #Create interactive Plot to change number of Prony terms
            widgets.interact(self.inter_opt_fig, 
                    N=widgets.Dropdown(
                        options=self.dict_prony.keys(), 
                        value=self.N_opt, 
                        description = 'Number of Prony terms:',
                        style = {'description_width' : 'initial'},
                        layout = widgets.Layout(height = 'auto', width = '200px'),
                        continuous_update=False))
        with self.out_res:
            #Plot least squares residual
            self.fig_res = opt.plot_residual(self.N_opt_err)

            #Add figure to file package
            self.files['fig_res'] = self.fig_res


    """
    ----------------------------------------------------------------------------
    Subsections - Download & Clear
    ----------------------------------------------------------------------------
    """
    
    #Download zip
    #---------------------------------------------------------------------------
    def trigger_download(self, data, filename, kind='text/json'):
        """
        Trigger download through HTML output widget. 
        
        Works in Jupyter notebook and voila.

        Reference
        ---------
        https://github.com/voila-dashboards/voila/issues/711
        """
        if isinstance(data, str):
            content_b64 = b64encode(data.encode()).decode()
        elif isinstance(data, bytes):
            content_b64 = b64encode(data).decode()
        data_url = f'data:{kind};charset=utf-8;base64,{content_b64}'
        js_code = f"""
            var a = document.createElement('a');
            a.setAttribute('download', '{filename}');
            a.setAttribute('href', '{data_url}');
            a.click()
        """
        with self.out_html:
            clear_output()
            display(HTML(f'<script>{js_code}</script>'))
            

    def down_zip(self, b):
        """
        Create zip archive of all dataframes and figures and trigger download.
        """
        if len(self.files) == 0:
            with self.out_html:
                clear_output()
                print('No files to download!')
        else:
            with self.out_html:
                clear_output()
                display(self.w_loading)
            zip_b64 = generate_zip(self.files)
            self.trigger_download(zip_b64, 'fit.zip', kind='text/plain')


    #Reload notebook
    #---------------------------------------------------------------------------
    def reload(self,b):
        """
        Reload the webpage to clear all data and recreate class objects.
        """
        with self.out_html:
            clear_output()
            display(HTML(
                '''
                    <script>
                        window.location.reload();
                    </script>            
                '''
            ))


    """
    ----------------------------------------------------------------------------
    Optional section - Manual shifting
    ----------------------------------------------------------------------------
    """
    def set_inp_step(self, change):
        """
        Set the step size for manually modifying the shift factors.
        """
        if change['new'] == 'coarse':
            self.inp_aT.step = 0.5
        elif change['new'] == 'medium':
            self.inp_aT.step = 0.1
        elif change['new'] == 'fine':
            self.inp_aT.step = 0.05

    def get_aT(self, T):
        """
        Get corresponding shift factor to specified temperature.
        """
        try:
            idx = self.df_aT['T'][self.df_aT['T'] == T].index
            return self.df_aT['log_aT'].iloc[idx].to_list()[0]
        except IndexError:
            print('Selected temperature not in DataFrame!')

    def set_inp_T(self, change):
        """
        Set temperature of shift factor to be modified and update widgets.
        """
        self.inp_aT.value = self.get_aT(change['new'])
        if change['new'] == self.RefT:
            self.inp_aT.disabled = True
        else:
            self.inp_aT.disabled = False
        
    def set_inp_aT(self, change):
        """
        Manually modify shift factors and update master curve.
        """
        single = self.cb_single.value
        idx0 = self.df_aT['T'][self.df_aT['T'] == self.RefT].index
        idx = self.df_aT['T'][self.df_aT['T'] == self.inp_T.value].index
        delta = change['new'] - self.df_aT['log_aT'].iloc[idx].values
        
        #Update shift factors based on user input
        if delta != 0.0:
            if single:    
                self.df_aT['log_aT'].iloc[idx] += delta
            else:
                if idx < idx0:
                    for i in range(0, int(idx.values)+1):
                        self.df_aT['log_aT'].iloc[i] += delta
                elif idx > idx0:
                    for i in range(int(idx.values), self.df_aT['T'].shape[0]):
                        self.df_aT['log_aT'].iloc[i] += delta

            #Update master curve
            self.df_master = master.get_curve(self.df_raw, self.df_aT, self.RefT)

            #Update figure
            self.fig_master_shift = master.plot_shift_update(
                self.df_master, self.fig_master_shift, self.fig_master_shift_lax)

            #Update data in file package
            self.files['df_master'] = out.to_csv(self.df_master, self.units)
            self.files['df_aT'] = out.to_csv(self.df_aT, self.units)
            self.files['fig_master_shift'] = self.fig_master_shift
            
    def reset_df_aT(self):
        """
        Reset shift factors to initial state after manually modifying them.
        """
        #Reset shift factors
        self.df_aT = self.df_aT_ref.copy()

        #Reset master curve
        self.df_master = master.get_curve(self.df_raw, self.df_aT, self.RefT)

        #Reset figure
        self.fig_master_shift = master.plot_shift_update(
            self.df_master, self.fig_master_shift, self.fig_master_shift_lax)

        #Reset data in file package
        self.files['fig_master_shift'] = self.fig_master_shift
