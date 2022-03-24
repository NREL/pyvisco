## Theory of linear viscoelasticity

[Viscoelastic](https://en.wikipedia.org/wiki/Viscoelasticity) materials exhibit both viscous and elastic characteristics when undergoing deformation. The theory of linear viscoelasticity describes ideal materials for which there is a linear relationship between mechanical stress and mechanical strain. An extensive review can be found in [Brinson H. F., Brinson L. C., Polymer Engineering Science and Viscoelasticity,
Springer, (2015)](https://link.springer.com/book/10.1007/978-1-4899-7485-3). A brief summary of the equations used in this notebook is provided below and was extracted from [Springer, M., and Bosco N. Prog Photovolt 28.7 (2020): 659-681](https://onlinelibrary.wiley.com/doi/full/10.1002/pip.3257). 


### Mathematical description

The rate-dependent material behavior can be described in either the time or frequency domain. In the time domain, the uniaxial, nonaging, isothermal stress-strain equation for a linear viscoelastic material can be represented by a Boltzmann superposition integral,

\begin{equation}
\sigma(t) = \int_0^t E(t-\tau) \frac{\mathrm{d \varepsilon(\tau)}}{\mathrm{d} \tau} \mathrm{d} \tau \quad , 
\end{equation}

where $\sigma(t)$ is the stress response over time, $t$; $E(t)$ is the relaxation modulus; $\varepsilon$ denotes the strain; and $\tau$ is the integration variable. The limiting moduli for the viscoelastic material are defined as the instantaneous modulus, $E(t=0) = E_0$, and the equilibrium modulus, i.e., $E(t) \rightarrow E_{\infty}$ for $t \rightarrow \infty$. 

The [Generalized Maxwell Model](https://en.wikipedia.org/wiki/Generalized_Maxwell_model) is commonly used to represent the stress-strain response of polymers. The relaxation modulus derived from this model is given by,

$$ E(t) = E_{\infty} + \sum\limits_{i=1}^{m} E_i \exp\left(-\frac{t}{\tau_i} \right) \quad , $$

where $\tau_i$ (relaxation times),  $E_i = E_0 \alpha_i$ (relaxation moduli) are material properties, and $m$ is the number of terms in the series. The above equation is often referred to as Prony series, and the equilibrium modulus can be defined by the Prony series as,

$$ E_{\infty} = E_0 \left[1-\sum_{i=1}^N \alpha_i \right] \quad . $$

The material properties can be directly obtained from relaxation or frequency-dependent test data. Material properties measured in the time domain can be converted into the frequency domain, and vice versa, by making use of a Fourier transformation,

$$ E'(\omega) = E_{\infty} + \sum\limits_{i=1}^{m}\frac{\omega^2 \tau_i^2 E_i}{\omega^2 \tau_i^2 + 1} \quad ,$$
$$ E''(\omega) = \sum\limits_{i=1}^{m} \frac{\omega \tau_i E_i}{\omega^2 \tau_i^2 +1} \quad . $$

Herein, $E'(\omega)$ is the storage modulus, $E''(\omega)$ is the loss modulus, and $\omega = 2 \pi f$ is the angular frequency, where $f=1/t$ is the frequency in Hertz and $t$ is the time period in seconds, respectively. The complex modulus, $E^*(\omega)$, and the loss factor, $\tan(\delta)$, are given as,

$$ E^*(\omega) = E'(\omega)  + i E''(\omega) \quad \text{and} $$
$$ \tan(\delta)  =  \frac{E''(\omega)}{E'(\omega)} \quad , $$

respectively.

### Experimental characterization

An efficient way to determine the storage modulus, $E'(\omega)$, and the loss modulus, $E''(\omega)$, is by dynamic mechanical analysis (DMA) or dynamic mechanical thermal analysis (DMTA). The material under test is excited to mechanical steady-state oscillations, either load- or displacement-controlled, and the corresponding response is measured. Alternatively, relaxation experiments in the time domain can be conducted to determine the relaxation modulus, $E(t)$.

Measurements at very low frequencies (long time periods) can be very time-consuming and might be unfeasible for practical applications. On the other side of the spectrum, measurements at very high frequencies can be limited by the instrumentation or unintended heating of the sample during cyclic deformation. To avoid such situations, the [time-temperature superposition principle (TTSP)](https://en.wikipedia.org/wiki/Time%E2%80%93temperature_superposition) is applied for thermo-rheologically simple materials. For such materials, the viscoelastic response at one temperature is related to the viscoelastic response at another temperature by changing the time scale (or frequency). This way, the time scale for the materials characterization can be extended by conducting the same frequency measurements at different temperatures. Afterward, a reference temperature is selected, and the isothermal measurements are shifted on a logarithmic time (or frequency) scale to form a so–called master curve.

![TTSP](https://raw.githubusercontent.com/NREL/pyvisco/main/figures/TTSP_small.png)

Time-temperature shift factors, $a_{\mathrm{T}}(\theta)$, are defined as the horizontal shift that must be applied to individual measurements at a constant temperature, $\theta_i$, to form the master curve at the reference temperature, $\theta_{\mathrm{ref}}$.

The determined shift factors, $a_{\mathrm{T}}$, are used to define a shift function that describes the temperature dependence of the viscoelastic material. Different shift functions for various materials are available in the literature. A commonly used shift function is the the [Williams-Landel-Ferry (WLF)](https://en.wikipedia.org/wiki/Williams%E2%80%93Landel%E2%80%93Ferry_equation),

$$ \log(a_{\mathrm{T}}) = -\frac{C_1\left(\theta-\theta_{\mathrm{ref}}\right)}{C_2 + \left(\theta-\theta_{\mathrm{ref}}\right)} $$

where $\theta_{\mathrm{ref}}$ is the reference temperature of the master curve. $\theta$ is the temperature of interest, and $C_1$ and $C_2$ are calibration constants.  The TTSP is based on the kinetic theory of polymers, which is strictly speaking only valid above the glass transition temperature, $\theta_g$. Although the TTSP is thought to be valid also for temperatures below $\theta_g$, the exact lower limit is not well defined, and the principle is commonly applied to temperatures below $\theta_g$ as long as the measurement data are shiftable to form a smooth master curve. Alternatively, different shift functions can be fitted. Below we provide a routine to fit the polynomial shift function up to degree four. 
