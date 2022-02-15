"""Necessary classes and functions to create a graphical user interface
within a Jupyter notebook to perform the Prony series paramater identification.
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


#Convenience functions
#-----------------------------------------------------------------------------

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


#Classes for Jupyter notebook
#-----------------------------------------------------------------------------
class Widgets():
    """Create widgets for GUI"""
    def __init__(self):
        #self.notebook_width()
        self.ini_variables()
        self.widgets()
        self.layout()
        #self.show()
        styles.format_fig()
        styles.format_HTML(self.out_html)
        self.modul = 'E'

    def notebook_width(self):
        """Use full screen width for notebook."""
        display(HTML(
            '<style>'
                '#notebook { padding-top:0px !important; } '
                '.container { width:100% !important; } '
                '.end_space { min-height:0px !important; } '
            '</style>'
        ))

    def ini_variables(self):
        self.RefT = 0
        self.nprony = 0

    def widgets(self):
        """Define GUI widgets."""      
        
        ###INPUTS###
        #--------------------------------------

        _height = 'auto'
        _width = 'auto'
        _width_b = '200px'
        
        #Radio buttons -----------------------------
        self.rb_eplexor = widgets.RadioButtons(
            options=['Eplexor', 'user'],
            value='Eplexor', 
            description='Instrument:',
            disabled=False,
            layout = widgets.Layout(height = _height, width = _width))
        self.rb_eplexor.observe(self.set_ftype, 'value')
        
        self.rb_type = widgets.RadioButtons(
            options=['master', 'raw'],
            value='master',
            description='Type:',
            disabled=False,
            layout = widgets.Layout(height = _height, width = _width))
        self.rb_type.observe(self.set_mastern, 'value')
            
        self.rb_domain = widgets.RadioButtons(
            options=['freq', 'time'],
            value='freq', 
            description='Domain:',
            disabled=False,
            layout = widgets.Layout(height = _height, width = _width))
        self.rb_domain.observe(self.set_instr, 'value')

        self.rb_load = widgets.RadioButtons(
            options=['tensile', 'shear'],
            value='tensile', 
            description='Loading:',
            disabled=False,
            layout = widgets.Layout(height = _height, width = _width))
        self.rb_load.observe(self.set_load, 'value')

        self.rb_dis = widgets.RadioButtons(
            options=['default', 'manual'],
            value='default', 
            description='Discretization:',
            disabled=False,
            style = {'description_width' : 'initial'},
            layout = widgets.Layout(height = _height, width = _width))
        self.rb_dis.observe(self.set_dis, 'value')

        self.rb_dis_win = widgets.RadioButtons(
            options=['round', 'exact'],
            value='round', 
            description='Window:',
            disabled=True,
            layout = widgets.Layout(height = _height, width = _width))

        
        #Check box -------------------------------
        self.cb_shift = widgets.Checkbox(
            value=False, 
            description='user shift factors',
            disabled=False,
            indent = True,
            layout = widgets.Layout(height = _height, width = _width),
            style = {'description_width' : 'initial'})
        self.cb_shift.observe(self.set_shift, 'value')

        self.cb_aT = widgets.Checkbox(
            value=False, 
            description='fit and overwrite provided shift factors',
            disabled=True,
            indent = True,
            layout = widgets.Layout(height = _height, width = _width),
            style = {'description_width' : 'initial'})
        #self.cb_WLF.observe(self.set_shift, 'value')

        self.cb_WLF = widgets.Checkbox(
            value=False, 
            description='fit and overwrite provided WLF',
            disabled=True,
            indent = True,
            layout = widgets.Layout(height = _height, width = _width),
            style = {'description_width' : 'initial'})
        #self.cb_WLF.observe(self.set_shift, 'value')

        self.cb_ManShift = widgets.Checkbox(
            value=False, 
            description='manually adjust shift factors',
            disabled=True,
            indent = True,
            layout = widgets.Layout(height = _height, width = _width),
            style = {'description_width' : 'initial'})
        self.cb_ManShift.observe(self.show_manual_shift, 'value')
        
        #Text field -------------------------------
        self.ft_RefT = widgets.FloatText(
            value=self.RefT,
            description='Reference temperature (\N{DEGREE SIGN}C):',
            disabled=True,
            layout = widgets.Layout(height = _height, width = '250px'),
            style = {'description_width' : 'initial'})
        self.ft_RefT.observe(self.set_RefT, 'value')

        self.dd_RefT = widgets.Dropdown(
            #options = self.arr_RefT.values,
            #value=self.RefT,
            description='Reference temperature (\N{DEGREE SIGN}C):',
            disabled=False,
            layout = widgets.Layout(height = _height, width = '250px'),
            style = {'description_width' : 'initial'})
        self.ft_RefT.observe(self.set_RefT, 'value')

        self.it_nprony = widgets.BoundedIntText(
            value=self.nprony,
            min = 0,
            max = 1000,
            step = 1,
            description='Number of Prony terms:',
            disabled=True,
            layout = widgets.Layout(height = _height, width = '220px'),
            style = {'description_width' : 'initial'})
        #self.it_nprony.observe(self.set_nprony, 'value')
        
        #Upload buttons ---------------------------
        self.up_inp = widgets.FileUpload(
            accept= '.xls, .xlsx',
            multiple=False,
            button_style='success',
            layout = widgets.Layout(height = _height, width = _width_b))
        self.up_inp.observe(self.inter_load_master, names='_counter')

        self.out_load = widgets.Output()
        
        self.up_shift = widgets.FileUpload(
            accept='.csv, .xls', 
            multiple=False, 
            disabled=True,
            button_style='success',
            layout = widgets.Layout(height = _height, width = _width_b))
        self.up_shift.observe(self.inter_load_shift, names='_counter')

        self.out_load_shift = widgets.Output()


        #Valid indicator ---------------------------------
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
    
        #Buttons and outputs ---------------------------------

        #Theory
        self.b_theory = widgets.ToggleButton(
            value=False,
            description='Click here for more details!',
            layout = widgets.Layout(width = '200px'))
        self.b_theory.observe(self.show_theory, 'value')

        self.out_theory = widgets.HTMLMath(value='')
        
        #Load
        # self.b_load = widgets.Button(
        #     description='Load data',
        #     button_style='success',
        #     layout = widgets.Layout(height = _height, width = _width_b))
        # self.b_load.on_click(self.inter_load_master)
        
               
        #fit shift factors
        self.b_aT = widgets.Button(
            description='master raw data',
            button_style='info',
            disabled = True,
            layout = widgets.Layout(height = _height, width = _width_b))
        self.b_aT.on_click(self.inter_aT)

        self.out_aT = widgets.Output()
        self.out_aT_man = widgets.Output()
        
        
        #fit shift functions
        self.b_shift = widgets.Button(
            description='(fit) & plot shift functions',
            button_style='info',
            disabled = True,
            layout = widgets.Layout(height = _height, width = _width_b))
        self.b_shift.on_click(self.inter_shift)

        self.out_shift = widgets.Output()
        
        #Smooth
        self.b_smooth = widgets.Button(
            description='smooth master curve',
            button_style='info',
            layout = widgets.Layout(height = _height, width = _width_b))
        self.b_smooth.on_click(self.inter_smooth_fig)

        self.out_smooth = widgets.Output()
        
        #Discretization
        self.b_dis = widgets.Button(
            description='plot discretization',
            button_style='info', 
            layout = widgets.Layout(height = _height, width = _width_b))
        self.b_dis.on_click(self.inter_dis)

        self.out_dis = widgets.Output()
        
        #Prony fit
        self.b_fit = widgets.Button(
            description='fit Prony series',
            button_style='info',
            layout = widgets.Layout(height = _height, width = _width_b))
        self.b_fit.on_click(self.inter_fit)

        self.out_fit = widgets.Output()
        self.out_prony = widgets.Output()
               
        #Generalized Maxwell
        self.b_GMaxw = widgets.Button(
            description='plot Generalized Maxwell',
            button_style='info',
            layout = widgets.Layout(height = _height, width = _width_b))
        self.b_GMaxw.on_click(self.inter_GMaxw)
        
        self.out_GMaxw_freq = widgets.Output()
        self.out_GMaxw_temp = widgets.Output()

        #Minimize nprony
        self.b_opt = widgets.Button(
            description='minimize Prony terms',
            button_style='warning',
            layout = widgets.Layout(height = _height, width = _width_b))
        self.b_opt.on_click(self.inter_opt)
        
        self.out_dro = widgets.Output()
        self.out_opt = widgets.Output()
        self.out_res = widgets.Output()
        self.out_par = widgets.Output()

        #Output widgets for HTML content
        self.out_html = widgets.Output()
        
        #Download/HTML buttons -----------------------
        # self.db_prony = widgets.Button(
        #     description='Download Prony series',
        #     button_style='warning',
        #     layout = widgets.Layout(height = _height, width = _width_b))
        # self.db_prony.on_click(self.down_prony)
        
        self.db_zip = widgets.Button(
            description='Download zip',
            button_style='success',
            layout = widgets.Layout(height = _height, width = _width_b))
        self.db_zip.on_click(self.down_zip)

        self.b_reload = widgets.Button(
            description='Clear notebook!',
            button_style='danger',
            layout = widgets.Layout(height = 'auto', width = _width_b))
        self.b_reload.on_click(self.reload)




    def layout(self):

        _inp_gen = widgets.HBox([
            self.rb_domain,
            self.rb_load,
            self.rb_eplexor,
            self.rb_type,
            self.up_inp,],
            layout = widgets.Layout(width = '100%', justify_content='space-between'))

        self.w_inp_gen = widgets.VBox([
            _inp_gen,
            self.out_load],
            layout = widgets.Layout(width = '100%', justify_content='space-between'))

        _inp_shift = widgets.HBox([
            self.cb_shift,
            self.up_shift],
            layout = widgets.Layout(width = '100%', justify_content='space-between'))

        self.w_inp_shift = widgets.VBox([
            _inp_shift,
            self.out_load_shift],
            layout = widgets.Layout(width = '100%', justify_content='space-between'))

        self.w_RefT = widgets.HBox([self.ft_RefT])

        _load = widgets.HBox([
            #self.b_load,
            self.v_modulus,
            self.v_aT,
            self.v_WLF], 
            layout = widgets.Layout(width = '100%'))
            
        self.w_inp_load = widgets.VBox([
            _load],
            #self.out_load],
            layout = widgets.Layout(width = '100%')) #, align_items='center'

        self.w_aT = widgets.VBox([
            widgets.HBox([self.b_aT, self.cb_aT, self.cb_ManShift]),
            self.out_aT,
            self.out_aT_man])


        self.w_shift = widgets.VBox([
            widgets.HBox([self.b_shift, self.cb_WLF]),
            self.out_shift])

        _dis = widgets.HBox([
            self.rb_dis, 
            self.rb_dis_win, 
            self.it_nprony],  
            layout = widgets.Layout(width = '100%', justify_content='space-between'))

        self.w_dis = widgets.VBox([
            _dis,
            self.b_dis,
            self.out_dis])

        self.w_out_fit_prony = widgets.HBox([
            self.out_fit, 
            self.out_prony],  
            layout = widgets.Layout(width = '100%')) #, justify_content='space-between'


        self.w_out_GMaxw = widgets.HBox([
            self.out_GMaxw_freq, 
            self.out_GMaxw_temp],  
            layout = widgets.Layout(width = '100%')) #, justify_content='space-between'

        _minProny = widgets.HBox([
            self.out_opt,
            self.out_res],  
            layout = widgets.Layout(width = '100%')) #, justify_content='space-between'

        self.w_out_fit_min = widgets.VBox([
            self.out_dro,
            _minProny,
            self.out_par])


class Control(Widgets):
    """GUI Controls"""
    def __init__(self):
        super().__init__()
        self.collect_files()
        self.create_loading_bar()

        with open('./theory.html') as file:  
            self.html_theory = file.read() 

    def collect_files(self):
        self.files = {}

    def create_loading_bar(self):
        gif_address = './figures/loading.gif'

        with open(gif_address, 'rb') as f:
            img = f.read()

        self.w_loading = widgets.Image(value=img)
        
    #Set widgets and variables--------------------------------------------------------------------------------  
    def set_shift(self, change):
        with self.out_load_shift:
            clear_output()
        if change['new']:
            self.up_shift.disabled = False
        else:
            self.up_shift.disabled = True
            
    def set_RefT(self, change):
        if isinstance(self.arr_RefT, pd.Series): 
            self.w_RefT.children = [self.dd_RefT]
            self.RefT = self.arr_RefT.iloc[(self.arr_RefT-change['new']).abs().argsort()[:1]].values[0]
            self.dd_RefT.options = self.arr_RefT.values
            self.dd_RefT.value = self.RefT
        else:
            self.w_RefT.children = [self.ft_RefT]
            self.RefT = change['new']
            self.ft_RefT.value = self.RefT
            

    #def set_nprony(self, change):
    #    self.nprony = change['new']

    def set_mastern(self, change):
        with self.out_load:
            clear_output()
        if change['new'] == 'raw':
            self.b_aT.disabled = False
        else:
            self.b_aT.disabled = True

    def set_ftype(self, change):
        with self.out_load:
            clear_output()
        if change['new'] == 'Eplexor':
            self.up_inp.accept='.xls, .xlsx'
        elif change ['new'] == 'user':
            self.up_inp.accept='.csv'

    def set_instr(self, change):
        with self.out_load:
            clear_output()
        if change['new'] == 'freq':
            self.rb_eplexor.disabled = False
        elif change ['new'] == 'time':
            self.rb_eplexor.value = 'user'
            self.rb_eplexor.disabled = True

    def set_load(self, change):
        with self.out_load:
            clear_output()
        if change['new'] == 'tensile':
            self.modul = 'E'
        elif change['new'] == 'shear':
            self.modul = 'G'

    def set_dis(self, change):
        if change['new'] == 'default':
            self.rb_dis_win.disabled = True
            self.it_nprony.disabled = True
            self.rb_dis_win.value = 'round'
            self.it_nprony.value = 0
        elif change['new'] == 'manual':
            self.rb_dis_win.disabled = False
            self.it_nprony.disabled = False


    #Interactive functionality---------------------------------------------------------------------------------  
    def show_theory(self, change):
        if change['new'] == True:
            self.out_theory.value = self.html_theory
        elif change ['new'] == False:
            self.out_theory.value = ''
    
    
    
    def inter_load_master(self, b):
        with self.out_load:
            clear_output()
        with self.out_load_shift:
            clear_output()
        try:
            self.df_master = None
            self.df_raw = None
            self.df_aT = None
            self.df_WLF = None
            self.df_poly = None
            self.df_GMaxw = None
            self.prony = None
            self.arr_RefT = None
            self.v_modulus.value = False
            self.v_aT.value = False
            self.v_WLF.value = False
            self.cb_shift.value = False
            self.ft_RefT.disabled = False
            self.cb_aT.disabled = True
            self.cb_WLF.value = False
            self.cb_WLF.disabled = True
            self.cb_ManShift.value = False
            self.cb_ManShift.disabled = True
            self.b_shift.disabled = True

            self.collect_files()

            #Load modulus
            if self.rb_eplexor.value == 'Eplexor':
                if self.rb_type.value == 'master':
                    self.df_master, self.df_aT, self.df_WLF, self.units  = load.Eplexor_master(
                        self.up_inp.data[0], self.modul)
                    _change = {}
                    _change['new'] = self.df_master.RefT
                    self.set_RefT(_change)
                    self.ft_RefT.disabled = True

                   
                elif self.rb_type.value == 'raw':
                    self.df_raw, self.arr_RefT, self.units = load.Eplexor_raw(self.up_inp.data[0], self.modul)
                    _change = {}
                    _change['new'] = self.ft_RefT.value
                    self.set_RefT(_change)
   
                
            elif self.rb_eplexor.value == 'user':
                if self.rb_type.value == 'master':
                    self.df_master, self.units = load.user_master(
                        self.up_inp.data[0], self.rb_domain.value, self.RefT, self.modul)
                    _change = {}
                    _change['new'] = 0
                    self.set_RefT(_change)
                    self.ft_RefT.disabled = False

                elif self.rb_type.value == 'raw':
                    self.df_raw, self.arr_RefT, self.units = load.user_raw(
                        self.up_inp.data[0], self.rb_domain.value, self.modul)
                    _change = {}
                    _change['new'] = self.ft_RefT.value
                    self.set_RefT(_change)

            #Load shift factors
            #if self.cb_shift.value:
            #    self.df_aT = load.user_shift(self.up_shift.data[0])

            #Add dataframes to zip package and adjust widgets
            if isinstance(self.df_master, pd.DataFrame):             
                #self.files['df_master'] = self.df_master.to_csv(index = False)
                self.files['df_master'] = out.csv(self.df_master, self.units)
                self.v_modulus.value = True
            if isinstance(self.df_raw, pd.DataFrame):             
                #self.files['df_raw'] = self.df_raw.to_csv(index = False)
                #self.files['df_raw'] = out.csv(self.df_raw, self.units)
                self.v_modulus.value = True
            if isinstance(self.df_aT, pd.DataFrame):
                #self.files['df_aT'] = self.df_aT.to_csv(index = False)
                self.files['df_aT'] = out.csv(self.df_aT, self.units)
                self.v_aT.value = True
                self.b_shift.disabled = False
                if isinstance(self.df_raw, pd.DataFrame):   
                    self.cb_aT.disabled = False
            if isinstance(self.df_WLF, pd.DataFrame):
                #self.files['df_WLF'] = self.df_WLF.to_csv(index = False)
                self.files['df_shift_WLF_Eplexor'] = out.csv(self.df_WLF, self.units) 
                self.v_WLF.value = True
                self.cb_WLF.disabled = False

            with self.out_load:
                print(f'{bcolors.OKGREEN}Upload successful!{bcolors.ENDC}')


        except IndexError:
            with self.out_load:
                print(f'{bcolors.FAIL}Files not yet uploaded!')
        except KeyError as e:
            with self.out_load:
                print(f'{bcolors.FAIL}Input file header not as expected. Check the variable names and units.')
                print(str(e).replace('not found in axis', 
                    f'{bcolors.FAIL}Variable(s) missing or unkown in input file, check conventions!{bcolors.ENDC}'))
        except ValueError:
            with self.out_load:
                print(f'{bcolors.FAIL}Files uploaded in wrong format! Check file header conventions and file format.{bcolors.ENDC}')


    def inter_load_shift(self, b):
        with self.out_load_shift:
            clear_output()
        try:
            #Load shift factors
            if self.cb_shift.value:
                self.df_aT = load.user_shift(self.up_shift.data[0])

                if isinstance(self.arr_RefT, pd.Series): 
                    _T_shift = self.df_aT['T'].sort_values(ignore_index=True).to_numpy(dtype=float)
                    _T_modulus = self.arr_RefT.sort_values(ignore_index=True).to_numpy(dtype=float)
                    if not all(_T_shift == _T_modulus):
                        self.df_aT = None
                        raise ValueError('Temperatures')


            if isinstance(self.df_aT, pd.DataFrame):
                #self.files['df_aT'] = self.df_aT.to_csv(index = False)
                self.files['df_aT'] = out.csv(self.df_aT, self.units)

                self.v_aT.value = True
                self.b_shift.disabled = False
                if isinstance(self.df_raw, pd.DataFrame):   
                    self.cb_aT.disabled = False

            with self.out_load_shift:
                print(f"{bcolors.OKGREEN}Upload successful!{bcolors.ENDC}")

        except IndexError:
            with self.out_load_shift:
                print(f'{bcolors.FAIL}Files not yet uploaded!')
        except KeyError as e:
            with self.out_load_shift:
                print(f'{bcolors.FAIL}Input file header not as expected. Check the variable names and units.{bcolors.ENDC}')
                print(str(e).replace('not found in axis', 
                    f'{bcolors.FAIL}Variable(s) missing or unkown in input file, check conventions!{bcolors.ENDC}'))
        except ValueError as e:
            with self.out_load_shift:
                if str(e) == 'Temperatures':
                    print(f'{bcolors.FAIL}User shift factors are ignored!{bcolors.ENDC}')
                    print('User shift factors need to be provided for same temperature levels as modulus data!')
                    print('T[modulus]: {}'.format(_T_modulus))
                    print('T[log(aT)]: {}'.format(_T_shift))
                else:
                    print(f'{bcolors.FAIL}Files uploaded in wrong format! Check file header conventions and file format.{bcolors.ENDC}')
        except AttributeError as e:
            if 'units' in str(e):
                print(f'{bcolors.FAIL}Upload measurement data before uploading shift factors!{bcolors.ENDC}')
            else:
                raise


    def inter_aT(self,b):
        with self.out_aT:
            clear_output()
            display(self.w_loading)
            if not isinstance(self.df_aT, pd.DataFrame) or self.cb_aT.value:
                self.df_aT = master.get_aT(self.df_raw, self.RefT)

            self.df_master = master.get_curve(self.df_raw, self.df_aT, self.RefT)

            #self.files['df_master'] = self.df_master.to_csv(index = False)
            self.files['df_master'] = out.csv(self.df_master, self.units)
            #self.files['df_aT'] = self.df_aT.to_csv(index = False)
            self.files['df_aT'] = out.csv(self.df_aT, self.units)
            self.b_shift.disabled = False
            clear_output()
            try:
                self.fig_master_shift, self.fig_master_shift_lax = master.plot_shift(
                    self.df_raw, self.df_master, self.units)
                self.files['fig_master_shift'] = self.fig_master_shift
                self.cb_ManShift.disabled = False
            except NameError:
                with self.out_aT:
                    print('Raw and/or master dataframes are missing!')
      
    def inter_shift(self, b):
        with self.out_shift:
            clear_output()
            if not isinstance(self.df_WLF, pd.DataFrame) or self.cb_WLF.value:
                self.df_WLF = shift.fit_WLF(self.df_master.RefT, self.df_aT)
            self.df_poly_C, self.df_poly_K = shift.fit_poly(self.df_aT)

            self.fig_shift, self.df_shift = shift.plot(self.df_aT, self.df_WLF, self.df_poly_C)
            
        self.files['fig_shift'] = self.fig_shift
        #self.files['df_shift_plot'] = self.df_shift.to_csv(index = False)  
        #self.files['df_shift_WLF'] = self.df_WLF.to_csv(index = False)
        self.files['df_shift_poly_Celsius'] = self.df_poly_C.to_csv()
        self.files['df_shift_poly_Kelvin'] = self.df_poly_K.to_csv()

        #self.files['df_shift_plot'] = out.csv(self.df_shift, self.units) #TODO: do we want plot data?
        self.files['df_shift_WLF'] = out.csv(self.df_WLF, self.units)
        #self.files['df_shift_poly_Celsius'] = out.csv(self.df_poly_C.T, self.units, index_label = 'Coeff.')
        #self.files['df_shift_poly_Kelvin'] = out.csv(self.df_poly_K.T, self.units, index_label = 'Coeff.')
            
    def inter_smooth(self, win):
        try:
            self.df_master = master.smooth(self.df_master, win)
            self.fig_smooth = master.plot_smooth(self.df_master, self.units)
        
            #self.files['df_master'] = self.df_master.to_csv(index = False)
            self.files['df_master'] = out.csv(self.df_master, self.units)
            self.files['fig_smooth'] = self.fig_smooth
        except AttributeError:
            print('Upload or create master curve!')


    def inter_smooth_fig(self, b):
        with self.out_smooth:
            clear_output()
            widgets.interact(self.inter_smooth, 
                win=widgets.IntSlider(min=1, max=20, step=1, value=1, 
                    description = 'Window size:',
                    style = {'description_width' : 'initial'},
                    layout = widgets.Layout(height = 'auto', width = '300px'),
                    continuous_update=False))

            
    def inter_dis(self, b):
        with self.out_dis:
            clear_output()
            try:
                if self.rb_dis.value == 'default':
                    self.it_nprony.value = 0
                self.df_dis = prony.discretize(self.df_master, self.rb_dis_win.value, self.it_nprony.value)
                self.fig_dis = prony.plot_dis(self.df_master, self.df_dis, self.units)
                self.it_nprony.value = self.df_dis.nprony
                #self.files['df_dis'] = self.df_dis.to_csv()
                self.files['df_dis'] = out.csv(self.df_dis, self.units, index_label='i')
                self.files['fig_dis'] = self.fig_dis
            except (AttributeError, KeyError):
                  print('Smooth master curve before discretzation (win=1 for no filter).')
            
    def inter_fit(self, b):
        with self.out_prony:
            clear_output()
        with self.out_fit:
            try:
                #Display loading bar
                clear_output()
                display(self.w_loading)

                #Perform curve fitting
                if self.rb_domain.value == 'freq':
                    self.prony = prony.fit_freq(self.df_dis)
                elif self.rb_domain.value == 'time':
                    self.prony = prony.fit_time(self.df_dis, self.df_master)
                    
                self.df_GMaxw = prony.calc_GMaxw(**self.prony)
            
                #Clear loading bar
                clear_output()

                #Display figure and store data
                self.fig_fit = prony.plot_fit(self.df_master, self.df_GMaxw, self.units)
                self.files['fig_fit'] = self.fig_fit
                #self.files['df_GMaxw'] = self.df_GMaxw.to_csv(index = False)
                self.files['df_GMaxw'] = out.csv(self.df_GMaxw, self.units)
                #self.files['df_prony'] = self.prony['df_terms'].to_csv(index_label = 'i')
                self.files['df_prony'] = out.csv(self.prony['df_terms'], self.units, index_label = 'i')
            except AttributeError:
                clear_output()
                print('Discretization of master curve is missing!')
                return

        with self.out_prony:
            #Plot prony terms next to figure
            clear_output()
            print('{}_0 = {:.2f} {}'.format(self.modul, 
                self.prony['E_0'], self.units['{}_0'.format(self.modul)]))
            print(self.prony['df_terms'][['tau_i', 'alpha_i']])

    def inter_GMaxw(self, b):
        try:
            with self.out_GMaxw_freq:
                clear_output()
                self.fig_GMaxw = prony.plot_GMaxw(self.df_GMaxw, self.units)
                self.files['fig_GMaxw'] = self.fig_GMaxw

            with self.out_GMaxw_temp:
                clear_output()
                if isinstance(self.df_WLF, pd.DataFrame):
                    self.df_GMaxw_temp = prony.GMaxw_temp('WLF', self.df_GMaxw, self.df_WLF, self.df_aT)
                    self.fig_GMaxw_temp = prony.plot_GMaxw_temp(self.df_GMaxw_temp, self.units)

                    #self.files['df_GMaxw_temp'] = self.df_GMaxw_temp.to_csv(index = False)
                    self.files['df_GMaxw_temp'] = out.csv(self.df_GMaxw_temp, self.units)
                    self.files['fig_GMaxw_temp'] = self.fig_GMaxw_temp

        except AttributeError:
            print('Prony series parameters are missing!')


    def inter_opt_fig(self, N):
        with self.out_opt:
            clear_output()
            self.df_GMaxw_opt, self.fig_opt = opt.plot_fit(self.df_master, self.dict_prony, N, self.units)
        with self.out_par:
            clear_output()
            self.fig_coeff = prony.plot_param([self.prony, self.dict_prony[N]], ['initial', 'minimized'])

        #self.files['df_prony_min'] = self.dict_prony[N]['df_terms'].to_csv(index_label = 'i')
        #self.files['df_GMaxw_min'] = self.df_GMaxw_opt.to_csv(index = False)
        self.files['df_prony_min'] = out.csv(self.dict_prony[N]['df_terms'], self.units, index_label = 'i')
        self.files['df_GMaxw_min'] = out.csv(self.df_GMaxw_opt, self.units)
        self.files['fig_fit_min'] = self.fig_opt
        self.files['fig_coeff'] = self.fig_coeff
        

    def inter_opt(self, b):
        with self.out_res:
            clear_output()
        with self.out_par:
            clear_output()
        with self.out_opt:
            clear_output()
            display(self.w_loading)
            try:
                self.dict_prony, self.N_opt, self.N_opt_err = opt.nprony(self.df_master, self.prony, window='min')
                clear_output()
            except (AttributeError, TypeError):
                clear_output()
                print('Initial Prony series fit is missing!')
                return
        with self.out_dro:
            clear_output()
            widgets.interact(self.inter_opt_fig, 
                    N=widgets.Dropdown(
                        options=self.dict_prony.keys(), 
                        value=self.N_opt, 
                        description = 'Number of Prony terms:',
                        style = {'description_width' : 'initial'},
                        layout = widgets.Layout(height = 'auto', width = '200px'),
                        continuous_update=False))
        with self.out_res:
            clear_output()
            self.fig_res = opt.plot_residual(self.N_opt_err)
            self.files['fig_res'] = self.fig_res


    #Manual shifting of master curve---------------------------------------------------------------------------------
    def show_manual_shift(self, change):
        #Widgets for manual shifting

        self.inp_step = widgets.Dropdown(
            options=['coarse', 'medium', 'fine'], 
            value = 'coarse',
            description='Step size:',
            disabled=False,
            layout = widgets.Layout(width = '200px'),
            style = {'description_width' : 'initial'}) 
        self.inp_step.observe(self.set_inp_step, 'value')

        self.inp_T = widgets.Dropdown(
            options=self.df_aT['T'], 
            value = self.df_aT['T'].iloc[0],
            description='Temperature (\N{DEGREE SIGN}C):',
            disabled=False,
            layout = widgets.Layout(width = '200px'),
            style = {'description_width' : 'initial'})
        self.inp_T.observe(self.set_inp_T, 'value')

        self.inp_aT = widgets.FloatText(
            step=0.5,
            value = self.get_aT(self.df_aT['T'].iloc[0]),
            description='log(a_T):',
            disabled=False,
            continuous_update=False,
            layout = widgets.Layout(width = '250px'),
            style = {'description_width' : 'initial'})
        self.inp_aT.observe(self.set_inp_aT, 'value')

        self.cb_single = widgets.Checkbox(
            value=False, 
            description='shift single set',
            disabled=False,
            indent = False,
            layout = widgets.Layout(width = '200px'),
            style = {'description_width' : 'initial'})
        #self.cb_single.observe(self.set_shift, 'value')


        self.w_inp_man = widgets.HBox([
            self.inp_step,
            self.inp_T,
            self.inp_aT,
            self.cb_single],
            layout = widgets.Layout(width = '100%', justify_content='space-between'))


        # aT_ref = self.get_aT(self.df_aT['T'].iloc[0])
        # self.inp_aT = widgets.FloatSlider(
        #     value=aT_ref,
        #     min=aT_ref - 10,
        #     max=aT_ref + 10,
        #     step=0.1,
        #     description='Test:',
        #     disabled=False,
        #     continuous_update=False,
        #     orientation='horizontal',
        #     readout=True,
        #     readout_format='.1f')

        with self.out_aT_man:
            if change['new'] == True:
                self.df_aT_ref = self.df_aT.copy()
                clear_output()
                display(self.w_inp_man)
            else:
                self.reset_df_aT()
                clear_output()

    def set_inp_step(self, change):
        if change['new'] == 'coarse':
            self.inp_aT.step = 0.5
        elif change['new'] == 'medium':
            self.inp_aT.step = 0.1
        elif change['new'] == 'fine':
            self.inp_aT.step = 0.05

            
    def get_aT(self, T):
        try:
            idx = self.df_aT['T'][self.df_aT['T'] == T].index
            return self.df_aT['log_aT'].iloc[idx].to_list()[0]
        except IndexError:
            print('Selected temperature not in DataFrame!')

    # def get_T0(self):
    #     idx0 = self.df_aT['log_aT'][self.df_aT['log_aT'] == 0].index
    #     T0 = self.df_aT['T'].iloc[idx0].to_list()[0]
    #     return T0, idx0


    def set_inp_T(self, change):
        self.inp_aT.value = self.get_aT(change['new'])
        if change['new'] == self.RefT:
            self.inp_aT.disabled = True
        else:
            self.inp_aT.disabled = False
        

    def set_inp_aT(self, change):
        idx0 = self.df_aT['T'][self.df_aT['T'] == self.RefT].index
        idx = self.df_aT['T'][self.df_aT['T'] == self.inp_T.value].index
        delta = change['new'] - self.df_aT['log_aT'].iloc[idx].values
       
        single = self.cb_single.value
        
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

            self.df_master = master.get_curve(self.df_raw, self.df_aT, self.RefT)
            self.fig_master_shift = master.plot_shift_update(
                self.df_master, self.fig_master_shift, self.fig_master_shift_lax)

            self.files['df_master'] = out.csv(self.df_master, self.units)
            self.files['df_aT'] = out.csv(self.df_aT, self.units)
            self.files['fig_master_shift'] = self.fig_master_shift
            
    def reset_df_aT(self):
        self.df_aT = self.df_aT_ref.copy()

        self.df_master = master.get_curve(self.df_raw, self.df_aT, self.RefT)
        self.fig_master_shift = master.plot_shift_update(
            self.df_master, self.fig_master_shift, self.fig_master_shift_lax)
        self.files['fig_master_shift'] = self.fig_master_shift










    #Download functionality---------------------------------------------------------------------------------
    def trigger_download(self, data, filename, kind='text/json'):
        """
        see https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/Data_URIs for details
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
            
    #def down_prony(self, b):
    #    self.trigger_download(self.files['df_prony'], 'df_prony.csv', kind='text/plain')
        
    def down_zip(self, b):
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

    #Clear/refresh notebook---------------------------------------------------------------------------------
    def reload(self,b):
        with self.out_html:
            clear_output()
            display(HTML(
                '''
                    <script>
                        window.location.reload();
                    </script>            
                '''
            ))


