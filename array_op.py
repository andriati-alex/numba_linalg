import math;
import cmath;
from numba import jit, prange, float64, complex128;

@jit([(complex128, complex128[:]), (float64, float64)])
def arrFill(z,  arr):
    for i in prange(arr.shape[0]):
        arr[i] = z;

@jit([(complex128[:], complex128[:]), (float64[:], float64[:])])
def arrCopy(from_arr, to_arr):
    for i in prange(from_arr.shape[0]):
        to_arr[i] = from_arr[i];

@jit([(complex128[:], complex128[:])])
def arrConj(v, vconj):
    for i in prange(v.shape[0]):
        vconj[i] = v[i].conj();

@jit([(complex128[:], complex128[:])])
def arrRPart(v, vreal):
    for i in prange(v.shape[0]):
        vreal[i] = v[i].real;

@jit([(complex128[:], complex128[:])])
def arrIPart(v, vimag):
    for i in prange(v.shape[0]):
        vreal[i] = v[i].imag;

@jit([(float64[:], float64[:], float64[:]),
      (complex128[:], complex128[:], complex128[:])])
def arrAdd(v1, v2, v):
    for i in prange(v.shape[0]):
        v[i] = v1[i] + v2[i];

@jit([(float64[:], float64[:], float64[:]),
      (complex128[:], complex128[:], complex128[:])])
def arrSub(v1, v2, v):
    for i in prange(v.shape[0]):
        v[i] = v1[i] - v2[i];

@jit([(float64[:], float64[:], float64[:]),
      (complex128[:], complex128[:], complex128[:])])
def arrProd(v1, v2, v):
    for i in prange(v.shape[0]):
        v[i] = v1[i] * v2[i];

@jit([(float64[:], float64[:], float64[:]),
      (complex128[:], complex128[:], complex128[:])])
def arrDiv(v1, v2, v):
    for i in prange(v.shape[0]):
        v[i] = v1[i] / v2[i];

@jit([(float64, float64[:], float64[:]),
      (complex128, complex128[:], complex128[:])])
def arrScalar(z, v, vz):
    for i in prange(v.shape[0]):
        vz[i] = z * v[i];

@jit([(float64[:], float64, float64[:], float64[:]),
    (complex128[:], complex128, complex128[:], complex128[:])])
def arrUpdate(v1, z, v2, v):
    for i in prange(v.shape[0]):
        v[i] = v1[i] + z * v2[i];

@jit([(float64[:], float64[:]), (complex128[:], complex128[:])])
def arrAbs(v, vabs):
    for i in prange(v.shape[0]):
        vabs[i] = abs(v[i]);

@jit([(float64[:], float64[:]), (complex128[:], complex128[:])])
def arrAbs2(v, vabs):
    for i in prange(v.shape[0]):
        vabs[i] = v[i].real * v[i].real + v[i].imag * v[i].imag;

@jit([complex128(complex128[:], complex128[:])])
def arrDot(v1, v2):
    z = 0 + 0j;
    for i in prange(v1.shape[0]):
        z += v1[i].conj() * v2[i];
    return z;

@jit([complex128(complex128[:], complex128[:])])
def arrDot2(v1, v2):
    z = 0 + 0j;
    for i in prange(v1.shape[0]):
        z += v1[i] * v2[i];
    return z;

@jit([float64(complex128[:])])
def arrMod(v):
    mod = 0;
    for i in prange(v.shape[0]):
        mod += v[i].imag * v[i].imag + v[i].real * v[i].real;
    return math.sqrt(mod);

@jit([float64(complex128[:])])
def arrMod2(v):
    mod = 0;
    for i in prange(v.shape[0]):
        mod += v[i].imag * v[i].imag + v[i].real * v[i].real;
    return mod;



# ************************** ELEMENTARY FUNCTIONS ************************** #



@jit((complex128, complex128[:], complex128[:]))
def arrCExp(z, v, exp_v):
    for i in prange(v.shape[0]):
        exp_v[i] = cmath.exp(z * v[i]);

@jit((float64, float64[:], float64[:]))
def arrRExp(z, v, exp_v):
    for i in prange(v.shape[0]):
        exp_v[i] = math.exp(z * v[i]);
