## Adora: Automatic Differentiation fOr Response functions using jAx

Proof of concept of Stokes I inversion stuff using Jax

Using
```
- jax==0.7.1
- optax==0.2.5
- jaxopt==0.8.5
- interpax==0.3.10
```

Currently only have populations for Fe I, using the partition functions fo Irwin
(1981). Other Fe I Kurucz lines than the pair specified in
`kurucz_6301_6302.linelist` should work.


TODO:
- [ ] Use a better Van der Waals broadening (e.g. Unsoeld)
- [x] Zeeman splitting
- [x] Polarised formal solver (e.g. DELO-Constant)
- [ ] Partition functions for other elements
- [x] Second order fitter that uses fast gradient calculation (yes, via scipy)
- [x] Reconstruction from nodes/splines -- very basic
- [ ] EOS (neural?)
- [ ] Connecting to PINN for regularisation?
- [ ] Validate values against existing inversion code. -- synthesis validated against Lightweaver

NOTES:
- The `lte_pops.py` script is just a demo, not used for anything.
- `voigt.py` should be ready to go for the full Voigt-Faraday.
- `response_fn.py`: demo of just calculating responses.
- `iterate.py`: LM demo (uses slow gradients).
- `iterate_adam.py`: Uses a first-order ML optimiser.
- `iterate_scipy.py`: Uses the jax compiled jacobian with the scipy trf optimiser.
- `vector_response_fn`: Demo of full Stokes solver calculating responses through DELO-linear
- `nodes.py`: Super basic differentiable node system, used as the basis of an inversion.
- This work has only been tested on CPU, but should also perform well on GPUs with fast fp64 support (e.g. A/H/B100).
- Intensities are in units of kW/(m2 nm sr). Thus emissivities are kW/(m3 nm), and opacities are m-1. All other units are SI.

