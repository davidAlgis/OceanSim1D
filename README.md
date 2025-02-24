# Ocean Waves 1D Simulation


# Theory

The part below summarizes an adaptation of Tessendorf’s method for simulating ocean waves—but in one spatial dimension rather than two. In this 1D formulation, the horizontal coordinate is denoted by $x$ and the simulation considers wave propagation along a line.


## Notations and Assumptions

- **Water Depth, $H$:**  
  $H\in\mathbb{R}^+$ denotes the water depth. Under the deep-water assumption, we consider $H\to+\infty$.

- **Water Surface Height, $h(x,t)$:**  
  The scalar function $h(x,t)$ gives the water surface elevation at horizontal position $x$ and time $t$.

- **Fluid Velocity, $v(x,y,t)$:**  
  In 1D, the fluid velocity is split into horizontal and vertical components:
  - Horizontal velocity: $v(x,y,t)$
  - Vertical velocity: $u(x,y,t)$  
  Here, $y$ represents the depth, with $y=0$ at the free surface.

- **Velocity Potential, $\phi(x,y,t)$:**  
  The function $\phi(x,y,t)$ is the potential of the fluid flow.

- **Gravitational Acceleration, $g$:**  
  $g=9.80665\,m/s^2$.

The approximated governing equations (derived from the Navier–Stokes equations under the potential flow assumption) become:
```math
\begin{aligned}
\frac{\partial \phi}{\partial t} &= -g\, h(x,t) \quad \text{for } y=0, \\
\Delta \phi &= 0 \quad \text{for } -H\le y\le 0, \\
\frac{\partial h}{\partial t} &= \frac{\partial \phi}{\partial y} \quad \text{for } y=0, \\
\frac{\partial \phi}{\partial y} &= 0 \quad \text{for } y=-H.
\end{aligned}
```

---

## Modeling the Water Surface

### Fourier Representation of $h(x,t)$

In 1D, the ocean surface is expressed as a sum of harmonic waves:
```math
h(x,t)=\sum_{k} \tilde{h}(k,t)\, e^{ikx},
```
where:
- The discrete wave number $k$ takes values such as $k=\frac{2\pi n}{L}$ for integers $n\in\left[-\frac{N}{2},\frac{N}{2}\right]$ over a domain of length $L$.
- The Fourier coefficient is defined by:
```math
\tilde{h}(k,t)=\tilde{h}_0(k)\, e^{i\omega(k)t} + \tilde{h}_0^*(-k)\, e^{-i\omega(k)t},
```
  with the dispersion relation:

```math
\omega(k)=\sqrt{g|k|}.
```

### Initial Amplitudes

The initial amplitudes $\tilde{h}_0(k)$ are often modeled via a statistical ocean spectrum. For example, one may write:
```math
\tilde{h}_0(k)=\xi_k\sqrt{\frac{4\pi}{L|k|}\,S(\omega)\left|\frac{d\omega}{dk}\right|},
```
where:
- $\xi_k$ are samples drawn from a Gaussian distribution (mean 0, standard deviation 1),
- $S(\omega)$ is the frequency spectrum (e.g., JONSWAP) defined below.

---

## Spectral Models

### JONSWAP Frequency Spectrum

For deep water conditions, the JONSWAP spectrum is given by:
```math
S(\omega)=\frac{\alpha g^2}{\omega^5}\exp\!\left(-\frac{5}{4}\left(\frac{\omega_p}{\omega}\right)^4\right)\gamma^r,
```
with:
- $\alpha=0.076\left(\frac{U^2}{F\,g}\right)^{0.22}$,
- $\omega_p=22\left(\frac{g^2}{UF}\right)$,
- $\gamma=3.3$,
- $r=\exp\!\left(-\frac{(\omega-\omega_p)^2}{2\sigma^2\omega_p^2}\right)$,
- 
```math
\sigma= 
\begin{cases} 
0.07, & \omega\le\omega_p, \\ 
0.09, & \omega>\omega_p. 
\end{cases}
```
---

## Horizontal Displacement and Choppy Waves

In the 2D case, a displacement vector is used to simulate choppiness. For 1D, we define a scalar horizontal displacement:
```math
D(x,t)=\sum_{k} \tilde{D}(k,t)\,e^{ikx},
```
with
```math
\tilde{D}(k,t)=\frac{i\,k}{|k|}\,\tilde{h}(k,t).
```
*Note:* For $k=0$ the displacement is defined as zero.

---

## Fluid Velocity

The velocity field has both horizontal and vertical components. In 1D, these are computed as:

### Horizontal Velocity
```math
v(x,y,t)=\sum_{k} E(k,y) \left[\tilde{h}_0(k)e^{i\omega(k)t}-\tilde{h}_0^*(-k)e^{-i\omega(k)t}\right]\left(-\frac{k\,g}{\omega(k)}\right)e^{ikx},
```

### Vertical Velocity
```math
u(x,y,t)=\sum_{k} E(k,y) \left[\tilde{h}_0(k)e^{i\omega(k)t}-\tilde{h}_0^*(-k)e^{-i\omega(k)t}\right]\left(i\omega(k)\right)e^{ikx},
```

where the attenuation function $E(k,y)$ is defined as:
```math
E(k,y)=
\begin{cases}
1+ky, & \text{if } y>0, \\
e^{ky}, & \text{if } y\le 0.
\end{cases}
```

---

## Implementation Details

### Fourier Transform

The wave field is efficiently computed by storing the Fourier coefficients in a 1D array of size $N$ and applying an inverse fast Fourier transform (IFFT) to obtain:
- The water surface $h(x,t)$,
- The displacement $D(x,t)$,
- The velocity components $v(x,y,t)$ and $u(x,y,t)$.

### Parametric Surface Reconstruction

Because the surface is described parametrically (with both $h(x,t)$ and $D(x,t)$), obtaining the exact water height at a given horizontal position requires an iterative procedure:
1. Estimate the corrected position $x'$ by subtracting $D(x,t)$ from $x$.
2. Recompute $h(x',t)$ until convergence.  
In practice, a fixed number of iterations (e.g., 4) offers a balance between accuracy and performance.

### Cascading Wave Components

To prevent the ocean surface from appearing overly repetitive due to limited resolution, the wave spectrum can be split into several cascades. In 1D, this involves:
- Dividing the spectrum into segments (e.g., long, intermediate, and short wavelengths).
- Assigning a different domain length $L_i$ to each cascade.
- Applying appropriate cutoff frequencies to avoid overlap between cascades.

### Interpolating Velocity in Depth

Calculating the velocity for every depth $y$ using IFFT may be computationally prohibitive. Instead:
- Compute the velocity at a discrete set of depths.
- Use interpolation to estimate $v(x,y,t)$ and $u(x,y,t)$ at any depth.
  
Given that the attenuation $E(k,y)$ decays exponentially with $y$, an exponential interpolation of the form
```math
f_e(y)=\alpha \exp(\beta y)
```
is well suited. A logarithmic spacing of depth samples can further improve accuracy near the free surface where gradients are largest.

---

## Conclusion

The 1D adaptation of Tessendorf’s method involves:

- Representing the water surface as a sum of Fourier components:
```math
h(x,t)=\sum_{k} \tilde{h}(k,t)\,e^{ikx}.
```
- Using a statistical spectrum (e.g., JONSWAP) to compute initial amplitudes.
- Introducing horizontal displacement $D(x,t)$ to model choppy waves.
- Computing horizontal and vertical velocities using an attenuation function $E(k,y)$ and Fourier synthesis.
- Efficiently implementing the simulation using IFFT and handling multiple wavelength cascades.
- Employing an iterative correction for the parametric description of the surface and interpolating velocities across depth.

This approach maintains the core physical insights of the original 2D method while reducing the complexity to a one-dimensional framework.
