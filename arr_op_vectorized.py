import math;
import cmath;
from numba import jit, vectorize, prange, uint64, float64, complex128;

@jit([(uint64, complex128, complex128[:]),
      (uint64, float64, float64[:])], nopython=True, nogil=True)
def arrFill(n, z, arr):
    for i in prange(n):
        arr[i] = z;

@jit([(uint64, complex128[:], complex128[:]), 
      (uint64, float64[:], float64[:])], nopython=True, nogil=True)
def arrCopy(n, from_arr, to_arr):
    for i in prange(n):
        to_arr[i] = from_arr[i];

@vectorize(["complex128(complex128)"], nopython=True)
def arrConj(v):
    return v.conjugate();

@vectorize("complex128(complex128)", nopython=True)
def arrRPart(v):
    return v.real;

@vectorize("complex128(complex128)", nopython=True)
def arrIPart(v):
    return v.imag;

@vectorize(["complex128(complex128, complex128)",
            "float64(float64, float64)"], nopython=True, target='parallel')
def arrAdd(v1, v2):
    return v1 + v2;

@vectorize(["complex128(complex128, complex128)",
            "float64(float64, float64)"], nopython=True, target='parallel')
def arrSub(v1, v2):
    return v1 - v2;

@vectorize(["complex128(complex128, complex128)",
            "float64(float64, float64)"], nopython=True, target='parallel')
def arrProd(v1, v2):
    return v1 * v2;

@vectorize(["complex128(complex128, complex128)",
            "float64(float64, float64)"], nopython=True)
def arrDiv(v1, v2):
    return v1 / v2;

@jit([(uint64, float64, float64[:], float64[:]),
      (uint64, complex128, complex128[:], complex128[:])], nopython=True, nogil=True)
def arrScalarMult(n, z, v, vz):
    for i in prange(n):
        vz[i] = z * v[i];

@jit([(uint64, float64, float64[:], float64[:]),
      (uint64, complex128, complex128[:], complex128[:])], nopython=True, nogil=True)
def arrScalarAdd(n, z, v, vz):
    for i in prange(n):
        vz[i] = z + v[i];

@jit([(uint64, float64[:], float64, float64[:], float64[:]),
      (uint64, complex128[:], complex128, complex128[:], complex128[:])],
      nopython=True, nogil=True)
def arrUpdate(n, v1, z, v2, v):
    for i in prange(n):
        v[i] = v1[i] + z * v2[i];

@vectorize(["float64(complex128)", "float64(float64)"], nopython=True)
def arrAbs(v):
    return abs(v);

@vectorize(["float64(complex128)", "float64(float64)"], nopython=True)
def arrAbs2(v):
    return v.real * v.real + v.imag * v.imag;

@jit([complex128(uint64, complex128[:], complex128[:])], nopython=True, nogil=True)
def arrDot(n, v1, v2):
    z = 0 + 0j;
    for i in prange(n):
        z += v1[i].conjugate() * v2[i];
    return z;

@jit([complex128(uint64, complex128[:], complex128[:])], nopython=True, nogil=True)
def arrDot2(n, v1, v2):
    z = 0 + 0j;
    for i in prange(n):
        z += v1[i] * v2[i];
    return z;

@jit([float64(uint64, complex128[:])], nopython=True, nogil=True)
def arrMod(n, v):
    mod = 0;
    for i in prange(n):
        mod += v[i].imag * v[i].imag + v[i].real * v[i].real;
    return math.sqrt(mod);

@jit([float64(uint64, complex128[:])], nopython=True, nogil=True)
def arrMod2(n, v):
    mod = 0;
    for i in prange(n):
        mod += v[i].imag * v[i].imag + v[i].real * v[i].real;
    return mod;



# ************************** ELEMENTARY FUNCTIONS ************************** #



@jit((uint64, complex128, complex128[:], complex128[:]), nopython=True, nogil=True)
def arrCExp(n, z, v, exp_v):
    for i in prange(n):
        exp_v[i] = cmath.exp(z * v[i]);

@jit((uint64, float64, float64[:], float64[:]), nopython=True, nogil=True)
def arrRExp(n, z, v, exp_v):
    for i in prange(n):
        exp_v[i] = math.exp(z * v[i]);
