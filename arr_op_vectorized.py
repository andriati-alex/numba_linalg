"""
    NUMBA MODULE OF BASIC VECTOR OPERATIONS
    =======================================
"""


import math;
import cmath;
from numba import guvectorize, jit, vectorize, prange
from numba import uint64, float64, complex128;

#******************************************#
@jit(["(uint64, complex128, complex128[:])",
      "(uint64, float64, float64[:])"],
      nopython=True, nogil=True)
#******************************************#
def arrFill(n, z, arr):
    for i in prange(n):
        arr[i] = z;

#*********************************************#
@jit(["(uint64, complex128[:], complex128[:])", 
      "(uint64, float64[:], float64[:])"],
      nopython=True, nogil=True)
#*********************************************#
def arrCopy(n, from_arr, to_arr):
    for i in prange(n):
        to_arr[i] = from_arr[i];

@guvectorize(["void(uint64, complex128[:], complex128[:])",
              "void(uint64, float64[:], float64[:])"],
              "(),(n)->(n)",
              nopython=True, target='cpu')
def arrConj(n, v, vconj):
    for i in prange(n): vconj[i] = v[i].conjugate();

@vectorize("float64(complex128)", nopython=True, target='cpu')
def arrRPart(v):
    return v.real;

@vectorize("float64(complex128)", nopython=True, target='cpu')
def arrIPart(v):
    return v.imag;

@vectorize(["complex128(complex128, complex128)",
            "float64(float64, float64)"], nopython=True, target='cpu')
def arrAdd(v1, v2):
    return v1 + v2;

@vectorize(["complex128(complex128, complex128)",
            "float64(float64, float64)"], nopython=True, target='cpu')
def arrSub(v1, v2):
    return v1 - v2;

@vectorize(["complex128(complex128, complex128)",
            "float64(float64, float64)"], nopython=True, target='cpu')
def arrMultiply(v1, v2):
    return v1 * v2;

@vectorize(["complex128(complex128, complex128)",
            "float64(float64, float64)"], nopython=True, target='cpu')
def arrDivision(v1, v2):
    return v1 / v2;

@guvectorize(["void(uint64, float64, float64[:], float64[:])",
              "void(uint64, complex128, complex128[:], complex128[:])"],
              "(),(),(n)->(n)",
              nopython=True, target='cpu')
def arrScalarMultiply(n, z, v, vz):
    for i in prange(n):
        vz[i] = z * v[i];

@guvectorize(["void(uint64, float64, float64[:], float64[:])",
              "void(uint64, complex128, complex128[:], complex128[:])"],
              "(),(),(n)->(n)",
              nopython=True, target='cpu')
def arrScalarAdd(n, z, v, vz):
    for i in prange(n):
        vz[i] = z + v[i];

######################################################################
@guvectorize(["void(uint64, complex128[:], complex128, complex128[:],\
               complex128[:])",
              "void(uint64, complex128[:], float64, complex128[:],\
               complex128[:])",
              "void(uint64, float64[:], float64, float64[:], float64[:])"],
              "(),(n),(),(n)->(n)",
              nopython=True, target='cpu')
######################################################################
def arrUpdate(n, v1, z, v2, v):
    for i in prange(n):
        v[i] = v1[i] + z * v2[i];

@vectorize(["float64(complex128)", "float64(float64)"],
            nopython=True, target='parallel')
def arrAbs(v):
    return abs(v);

@vectorize(["float64(complex128)", "float64(float64)"], nopython=True,
           target='parallel')
def arrAbs2(v):
    return v.real * v.real + v.imag * v.imag;

@jit([complex128(uint64, complex128[:], complex128[:])],
     nopython=True, nogil=True, parallel=True)
def arrDot(n, v1, v2):
    z = 0 + 0j;
    for i in prange(n):
        z += v1[i].conjugate() * v2[i];
    return z;

@jit([complex128(uint64, complex128[:], complex128[:])],
     nopython=True, nogil=True, parallel=True)
def arrDot2(n, v1, v2):
    z = 0 + 0j;
    for i in prange(n):
        z += v1[i] * v2[i];
    return z;

@jit([float64(uint64, complex128[:])],
     nopython=True, nogil=True, parallel=True)
def arrMod(n, v):
    mod = 0;
    for i in prange(n):
        mod += v[i].imag * v[i].imag + v[i].real * v[i].real;
    return math.sqrt(mod);

@jit([float64(uint64, complex128[:])],
     nopython=True, nogil=True, parallel=True)
def arrMod2(n, v):
    mod = 0;
    for i in prange(n):
        mod += v[i].imag * v[i].imag + v[i].real * v[i].real;
    return mod;



# ************************** ELEMENTARY FUNCTIONS ************************** #



@vectorize(["complex128(complex128)",
            "complex128(complex128)"],
            nopython=True, target='parallel')
def arrCExp(v):
    return cmath.exp(v);

@jit((uint64, float64, float64[:], float64[:]),
      nopython=True, nogil=True, parallel=True)
def arrRExp(n, z, v, exp_v):
    for i in prange(n):
        exp_v[i] = math.exp(z * v[i]);
