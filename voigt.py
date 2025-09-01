import jax
import jax.numpy as jnp

def voigt_H_impl(a, v):
    l = 4.1195342878142354; # l=sqrt(n/sqrt(2.))  ! L = 2**(-1/4) * N**(1/2)
    ac = (
        -1.5137461654527820e-10, 4.9048215867870488e-09,
        1.3310461806370372e-09,  -3.0082822811202271e-08,
        -1.9122258522976932e-08, 1.8738343486619108e-07,
        2.5682641346701115e-07,  -1.0856475790698251e-06,
        -3.0388931839840047e-06, 4.1394617248575527e-06,
        3.0471066083243790e-05,  2.4331415462641969e-05,
        -2.0748431511424456e-04, -7.8166429956142650e-04,
        -4.9364269012806686e-04, 6.2150063629501763e-03,
        3.3723366855316413e-02,  1.0838723484566792e-01,
        2.6549639598807689e-01,  5.3611395357291292e-01,
        9.2570871385886788e-01,  1.3948196733791203e+00,
        1.8562864992055408e+00,  2.1978589365315417e+00
    )

    s = jnp.abs(v) + a
    # Region I
    z = jnp.complex_(a - v*1j)
    reg1 = (z * 0.5641896) / (0.5 + z * z)

    recLmZ = 1.0 / jnp.complex_(l + a - v*1j)
    t = jnp.complex_(l - a +  v*1j) * recLmZ
    wei24 = recLmZ * (
        0.5641896 + 2.0 * recLmZ * (
        ac[23]+(ac[22]+(ac[21]+(ac[20]+(ac[19]+(ac[18]+(ac[17]+(ac[16]+(ac[15]+(ac[14]+(ac[13]+(ac[12]+(ac[11]+(ac[10]+(ac[9]+(ac[8]+
        (ac[7]+(ac[6]+(ac[5]+(ac[4]+(ac[3]+(ac[2]+(ac[1]+ac[0]*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t
    ))
    result = jnp.where(s >= 15.0, reg1, wei24)
    return result.real, result.imag

# @jax.custom_jvp
def voigt_H(a, v):
    return voigt_H_impl(a, v)

# @voigt_H.defjvp
# def voigt_H_jvp(primals, tangents):
#     a, v = primals
#     a_dot, v_dot = tangents
#     grad = jax.jacrev(voigt_H_impl, argnums=(0, 1), holomorphic=True)
#     primal_out = voigt_H_impl(a, v)
#     jac = grad(jnp.complex_(a), jnp.complex_(v))
#     tangent_out = jac[0] * a_dot + jac[1] * v_dot
#     breakpoint()
#     return primal_out, tangent_out

def voigt_H_re(a, v):
    return voigt_H_impl(a, v)[0]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    try:
        get_ipython().run_line_magic("matplotlib", "")
    except:
        plt.ion()

    a = jnp.array([0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0])
    v = jnp.linspace(-400, 400, 101)

    ia = 3
    voigt_H_jit = jax.jit(jax.vmap(voigt_H))
    HF = voigt_H_jit(jnp.ones(101) * a[ia], v)
    H = HF[0]
    F = HF[1]
    plt.figure()
    plt.plot(v, H)
    plt.plot(v, F)

    dvoigt = jax.jit(jax.vmap(jax.jacrev(voigt_H, argnums=(0, 1))))
    grads = dvoigt(jnp.ones(101) * a[ia], v)
    dHda = grads[0][0]
    dFda = grads[0][1]
    dHdv = grads[1][0]
    dFdv = grads[1][1]

    dvoigt_re = jax.jit(jax.vmap(jax.jacrev(voigt_H_re, argnums=(0, 1))))
    grads_re = dvoigt_re(jnp.ones(101) * a[ia], v)
    dvda = grads_re[0]
    dvdv = grads_re[1]

    plt.figure()
    plt.plot(v, dHda)
    plt.plot(v, dHdv)
    plt.plot(v, dFda)
    plt.plot(v, dFdv)
    plt.plot(v, dvda, '--')
    plt.plot(v, dvdv, '--')
