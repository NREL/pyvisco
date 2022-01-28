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
            else:
                fname = name
            zf.writestr(fname, data)

    return mem_zip.getvalue()


def fig_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, dpi = 600)
    return buf.getvalue()


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
            description='fit and overwrite provided shift factors?',
            disabled=True,
            indent = True,
            layout = widgets.Layout(height = _height, width = _width),
            style = {'description_width' : 'initial'})
        #self.cb_WLF.observe(self.set_shift, 'value')

        self.cb_WLF = widgets.Checkbox(
            value=False, 
            description='fit and overwrite provided WLF?',
            disabled=True,
            indent = True,
            layout = widgets.Layout(height = _height, width = _width),
            style = {'description_width' : 'initial'})
        #self.cb_WLF.observe(self.set_shift, 'value')
        
        #Text field -------------------------------
        self.ft_RefT = widgets.FloatText(
            value=self.RefT,
            description='Reference temperature (\N{DEGREE SIGN}C):',
            disabled=False,
            layout = widgets.Layout(height = _height, width = '220px'),
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
            layout = widgets.Layout(height = _height, width = _width_b))
        
        self.up_shift = widgets.FileUpload(
            accept='.csv, .xls', 
            multiple=False, 
            disabled=True,
            layout = widgets.Layout(height = _height, width = _width_b))


        #Valid indicator ---------------------------------
        self.v_modulus = widgets.Valid(
            value=False,
            description='modulus data',
            continuous_update=True,
            readout = '', #string.whitespace
            layout = widgets.Layout(height = _height, width = _width_b))

        self.v_aT = widgets.Valid(
            value=False,
            description='shift factors',
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
        self.b_load = widgets.Button(
            description='Load data',
            button_style='success',
            layout = widgets.Layout(height = _height, width = _width_b))
        self.b_load.on_click(self.inter_load_master)
        
        self.out_load = widgets.Output()

                
        #fit shift factors
        self.b_aT = widgets.Button(
            description='master raw data',
            button_style='info',
            disabled = True,
            layout = widgets.Layout(height = _height, width = _width_b))
        self.b_aT.on_click(self.inter_aT)

        self.out_aT = widgets.Output()
        
        
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
        
        self.out_GMaxw = widgets.Output()

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
        self.db_prony = widgets.Button(
            description='Download Prony series',
            button_style='warning',
            layout = widgets.Layout(height = _height, width = _width_b))
        self.db_prony.on_click(self.down_prony)
        
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

        self.w_inp_gen = widgets.HBox([
            self.rb_domain,
            self.rb_eplexor,
            self.rb_type,
            self.up_inp,],
            layout = widgets.Layout(width = '100%', justify_content='space-between'))

        self.w_inp_shift = widgets.HBox([
            self.ft_RefT,
            self.cb_shift,
            self.up_shift],
            layout = widgets.Layout(width = '100%', justify_content='space-between'))

        _load = widgets.HBox([
            self.b_load,
            self.v_modulus,
            self.v_aT,
            self.v_WLF], 
            layout = widgets.Layout(width = '100%'))
            
        self.w_inp_load = widgets.VBox([
            _load,
            self.out_load],
            layout = widgets.Layout(width = '100%')) #, align_items='center'

        self.w_aT = widgets.VBox([
            widgets.HBox([self.b_aT, self.cb_aT]),
            self.out_aT])


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
        if change['new']:
            self.up_shift.disabled = False
        else:
            self.up_shift.disabled = True
            
    def set_RefT(self, change):
        try:
            self.arr_RefT
        except AttributeError:
            self.RefT = change['new']
        else:
            self.RefT = self.arr_RefT.iloc[(self.arr_RefT-change['new']).abs().argsort()[:1]].values[0]
            self.ft_RefT.value = self.RefT

    #def set_nprony(self, change):
    #    self.nprony = change['new']

    def set_mastern(self, change):
        if change['new'] == 'raw':
            self.b_aT.disabled = False
        else:
            self.b_aT.disabled = True

    def set_ftype(self, change):
        if change['new'] == 'Eplexor':
            self.up_inp.accept='.xls, .xlsx'
        elif change ['new'] == 'user':
            self.up_inp.accept='.csv'

    def set_instr(self, change):
        if change['new'] == 'freq':
            self.rb_eplexor.disabled = False
        elif change ['new'] == 'time':
            self.rb_eplexor.value = 'user'
            self.rb_eplexor.disabled = True

    def set_dis(self, change):
        if change['new'] == 'default':
            self.rb_dis_win.disabled = True
            self.it_nprony.disabled = True
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
        try:
            self.df_master = None
            self.df_raw = None
            self.df_aT = None
            self.df_WLF = None
            self.df_GMaxw = None
            self.prony = None
            self.v_modulus.value = False
            self.v_aT.value = False
            self.v_WLF.value = False
            self.ft_RefT.disabled = False
            self.cb_aT.disabled = True
            self.cb_WLF.disabled = True
            self.b_shift.disabled = True
            self.collect_files()

            #Load modulus
            if self.rb_eplexor.value == 'Eplexor':
                if self.rb_type.value == 'master':
                    self.df_master, self.df_aT, self.df_WLF  = load.Eplexor_master(self.up_inp.data[0])
                    self.ft_RefT.value = self.df_master.RefT
                    self.ft_RefT.disabled = True

                   
                elif self.rb_type.value == 'raw':
                    self.df_raw, self.arr_RefT = load.Eplexor_raw(self.up_inp.data[0])
                    _change = {}
                    _change['new'] = self.ft_RefT.value
                    self.set_RefT(_change)
   
                
            elif self.rb_eplexor.value == 'user':
                if self.rb_type.value == 'master':
                    self.df_master = load.user_master(self.up_inp.data[0], self.rb_domain.value, self.RefT)

                elif self.rb_type.value == 'raw':
                    self.df_raw, self.arr_RefT = load.user_raw(self.up_inp.data[0], self.rb_domain.value)
                    _change = {}
                    _change['new'] = self.ft_RefT.value
                    self.set_RefT(_change)

            #Load shift factors
            if self.cb_shift.value:
                self.df_aT = load.user_shift(self.up_shift.data[0])

            #Add dataframes to zip package and adjust widgets
            if isinstance(self.df_master, pd.DataFrame):             
                self.files['df_master'] = self.df_master.to_csv(index = False)
                self.v_modulus.value = True
            if isinstance(self.df_raw, pd.DataFrame):             
                self.files['df_raw'] = self.df_raw.to_csv(index = False)
                self.v_modulus.value = True
            if isinstance(self.df_aT, pd.DataFrame):
                self.files['df_aT'] = self.df_aT.to_csv(index = False)
                self.v_aT.value = True
                self.b_shift.disabled = False
                if isinstance(self.df_raw, pd.DataFrame):   
                    self.cb_aT.disabled = False
            if isinstance(self.df_WLF, pd.DataFrame):
                self.files['df_WLF'] = self.df_WLF.to_csv(index = False)
                self.v_WLF.value = True
                self.cb_WLF.disabled = False

            with self.out_load:
                print('Upload successful!')
            
        except IndexError:
            with self.out_load:
                print('Upload files first!')
        except KeyError:
            with self.out_load:
                print('Column names not as expected. Check the headers in your input files!')
        except ValueError:
            with self.out_load:
                print('Uploaded files not in required format!')


    def inter_aT(self,b):
        with self.out_aT:
            clear_output()
            if not isinstance(self.df_aT, pd.DataFrame) or self.cb_aT.value:
                self.df_aT = master.get_aT(self.df_raw, self.RefT)

            self.df_master = master.get_curve(self.df_raw, self.df_aT, self.RefT)

            self.files['df_master'] = self.df_master.to_csv(index = False)
            self.files['df_aT'] = self.df_aT.to_csv(index = False)
            self.b_shift.disabled = False

            try:
                self.fig_master_shift = master.plot_shift(self.df_raw, self.df_master)
                self.files['fig_master_shift'] = fig_bytes(self.fig_master_shift)
            except NameError:
                with self.out_aT:
                    print('Raw and/or master dataframes are missing!')
      
    def inter_shift(self, b):
        with self.out_shift:
            clear_output()
            if not isinstance(self.df_WLF, pd.DataFrame) or self.cb_WLF.value:
                self.df_WLF = shift.fit_WLF(self.df_master.RefT, self.df_aT)
            self.df_poly = shift.fit_poly(self.df_aT)

            self.fig_shift, self.df_shift = shift.plot(self.df_aT, self.df_WLF, self.df_poly)
            
        self.files['fig_shift'] = fig_bytes(self.fig_shift)
        self.files['df_shift_plot'] = self.df_shift.to_csv(index = False)
        self.files['df_shift_WLF'] = self.df_WLF.to_csv(index = False)
        self.files['df_shift_poly'] = self.df_poly.to_csv(index_label = 'Coeff.')
            
    def inter_smooth(self, win):
        try:
            self.df_master = master.smooth(self.df_master, win)
            self.fig_smooth = master.plot_smooth(self.df_master)
        
            self.files['df_master'] = self.df_master.to_csv(index = False)
            self.files['fig_smooth'] = fig_bytes(self.fig_smooth)
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
                self.df_dis = prony.discretize(self.df_master, self.rb_dis_win.value, self.it_nprony.value)
                self.fig_dis = prony.plot_dis(self.df_master, self.df_dis)
                self.it_nprony.value = self.df_dis.nprony
                self.files['df_dis'] = self.df_dis.to_csv()
                self.files['fig_dis'] = fig_bytes(self.fig_dis)
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
                self.fig_fit = prony.plot_fit(self.df_master, self.df_GMaxw)
                self.files['fig_fit'] = fig_bytes(self.fig_fit)
                self.files['df_GMaxw'] = self.df_GMaxw.to_csv(index = False)
                self.files['df_prony'] = self.prony['df_terms'].to_csv(index_label = 'i')
            except AttributeError:
                clear_output()
                print('Discretization of master curve is missing!')
                return

        with self.out_prony:
            #Plot prony terms next to figure
            clear_output()
            print('E_0 = {:.2f} MPa'.format(self.prony['E_0']))
            print(self.prony['df_terms'][['tau', 'alpha']])

    def inter_GMaxw(self, b):
        with self.out_GMaxw:
            try:
                clear_output()
                self.fig_GMaxw = prony.plot_GMaxw(self.df_GMaxw)
                self.files['fig_GMaxw'] = fig_bytes(self.fig_GMaxw)
            except AttributeError:
                print('Prony series parameters are missing!')


    def inter_opt_fig(self, N):
        with self.out_opt:
            clear_output()
            self.df_GMaxw_opt, self.fig_opt = opt.plot_fit(self.df_master, self.dict_prony, N)
        with self.out_par:
            clear_output()
            self.fig_coeff = prony.plot_param([self.prony, self.dict_prony[N]], ['initial', 'minimized'])

        self.files['df_prony_min'] = self.dict_prony[N]['df_terms'].to_csv(index_label = 'i')
        self.files['df_GMaxw_min'] = self.df_GMaxw_opt.to_csv(index = False)
        self.files['fig_fit_min'] = fig_bytes(self.fig_opt)
        self.files['fig_coeff'] = fig_bytes(self.fig_coeff)
        

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
            self.files['fig_res'] = fig_bytes(self.fig_res)

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
            
    def down_prony(self, b):
        self.trigger_download(self.files['df_prony'], 'df_prony.csv', kind='text/plain')
        
    def down_zip(self, b):
        if len(self.files) == 0:
            with self.out_html:
                clear_output()
                print('No files to download!')
        else:
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


