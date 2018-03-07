import math;
import cmath;
from numba import jit, prange, uint64, float64, complex128;

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

@jit([(uint64, complex128[:], complex128[:])], nopython=True, nogil=True)
def arrConj(n, v, vconj):
    for i in prange(n):
        vconj[i] = v[i].conjugate();

@jit([(uint64, complex128[:], complex128[:])], nopython=True, nogil=True)
def arrRPart(n, v, vreal):
    for i in prange(n):
        vreal[i] = v[i].real;

@jit([(uint64, complex128[:], complex128[:])], nopython=True, nogil=True)
def arrIPart(n, v, vimag):
    for i in prange(n):
        vimag[i] = v[i].imag;

@jit([(uint64, complex128[:], complex128[:], complex128[:]),
      (uint64, float64[:], float64[:], float64[:])],
      nopython=True, nogil=True)
def arrAdd(n, v1, v2, v):
    for i in prange(n):
        v[i] = v1[i] + v2[i];

@jit([(uint64, complex128[:], complex128[:], complex128[:]),
      (uint64, float64[:], float64[:], float64[:])],
      nopython=True, nogil=True)
def arrSub(n, v1, v2, v):
    for i in prange(n):
        v[i] = v1[i] - v2[i];

@jit([(uint64, complex128[:], complex128[:], complex128[:]),
      (uint64, float64[:], float64[:], float64[:])],
      nopython=True, nogil=True)
def arrProd(n, v1, v2, v):
    for i in prange(n):
        v[i] = v1[i] * v2[i];

@jit([(uint64, complex128[:], complex128[:], complex128[:]),
      (uint64, float64[:], float64[:], float64[:])],
      nopython=True, nogil=True)
def arrDiv(n, v1, v2, v):
    for i in prange(n):
        v[i] = v1[i] / v2[i];

@jit([(uint64, complex128, complex128[:], complex128[:]),
      (uint64, float64, float64[:], float64[:])], nopython=True, nogil=True)
def arrScalarMult(n, z, v, vz):
    for i in prange(n):
        vz[i] = z * v[i];

@jit([(uint64, complex128, complex128[:], complex128[:]),
      (uint64, float64, float64[:], float64[:])], nopython=True, nogil=True)
def arrScalarAdd(n, z, v, vz):
    for i in prange(n):
        vz[i] = z + v[i];

@jit([(uint64, complex128[:], complex128, complex128[:], complex128[:]),
      (uint64, float64[:], float64, float64[:], float64[:])],
      nopython=True, nogil=True)
def arrUpdate(n, v1, z, v2, v):
    for i in prange(n):
        v[i] = v1[i] + z * v2[i];

@jit([(uint64, complex128[:], complex128[:]),
      (uint64, float64[:], float64[:])], nopython=True, nogil=True)
def arrAbs(n, v, vabs):
    for i in prange(n):
        vabs[i] = abs(v[i]);

@jit([(uint64, complex128[:], complex128[:]), 
      (uint64, float64[:], float64[:])], nopython=True, nogil=True)
def arrAbs2(n, v, vabs):
    for i in prange(n):
        vabs[i] = v[i].real * v[i].real + v[i].imag * v[i].imag;

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



@jit((uint64, complex128, complex128[:], complex128[:]),
     nopython=True, nogil=True, parallel=True)
def arrCExp(n, z, v, exp_v):
    for i in prange(n):
        exp_v[i] = cmath.exp(z * v[i]);

@jit((uint64, float64, float64[:], float64[:]), nopython=True, nogil=True)
def arrRExp(n, z, v, exp_v):
    for i in prange(n):
        exp_v[i] = math.exp(z * v[i]);
