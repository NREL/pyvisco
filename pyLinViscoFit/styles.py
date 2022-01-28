from ipympl.backend_nbagg import Canvas
from IPython.display import display, HTML
import matplotlib.pyplot as plt

#Define figure style
#-----------------------------------------------------------------------------
def format_fig():
    #Set default colors
    #plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sns.color_palette("colorblind")) #colorblind

    #Activate Grid
    plt.rcParams['axes.grid'] = True

    #Set default figure size to one column
    plt.rcParams['figure.figsize'] = (4,0.75*4)

    #Increase default resolution
    #plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600

    #Default use of Latex
    #plt.rcParams['text.usetex'] = True
    #plt.rcParams['font.family'] = 'serif' #sans-serif, monospace
    plt.rcParams['font.size'] = 11
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9

    #Change grid line properties
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.4

    #Change marker properties
    plt.rcParams['lines.markersize'] = 3.0

    #Change tick direction to inwards
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    #Define default legend options
    plt.rcParams['legend.framealpha'] = 1
    plt.rcParams['legend.edgecolor'] = '0'
    plt.rcParams['legend.handlelength'] = 2.2
    plt.rcParams['legend.fancybox'] = False
    plt.rcParams['legend.fontsize'] = 9

    #Deactivate open figure warning
    plt.rcParams.update({'figure.max_open_warning': 0})

    #Use constraint layout
    plt.rcParams['figure.constrained_layout.use'] = True
    #plt.rcParams['figure.constrained_layout.w_pad'] = 0.05
    #plt.rcParams['figure.constrained_layout.h_pad'] = 0.05

    #ipympl duplicate plot issue
    #https://issueexplorer.com/issue/matplotlib/ipympl/402
    plt.ioff()

    #Jupyter widget specific
    Canvas.header_visible.default_value = False
    Canvas.footer_visible.default_value = False
    #Canvas.resizable.default_value = False


def format_HTML(widget):
    with widget:
        display(HTML("""
        <style>
        .text_cell_render p {
            text-align: justify;
            text-justify: inter-word;
        }
        hr {
            background-color: #1a1a1a;
            border: none;
            height: 1px;
            margin: 1em 0;
        }
        input::-webkit-outer-spin-button,
        input::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
        input[type=number] {
            -moz-appearance: textfield;
        }
        </style>
        """
        ))