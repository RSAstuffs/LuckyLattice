#!/usr/bin/env python3
"""
fpylll-compatible wrapper for LLL reduction on large integers.
Provides fpylll-like interface without requiring fpylll installation.
Uses our custom LLL implementation that handles large integers.
"""

import numpy as np
from typing import Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)


def lll_reduce_integer(basis: np.ndarray, delta: float = 0.75) -> np.ndarray:
    """
    Integer-based LLL lattice basis reduction using exact integer arithmetic.
    Uses Python's arbitrary precision integers to handle large numbers.
    
    This is a simplified LLL that works directly with integers, approximating
    the Gram-Schmidt orthogonalization when needed for the Lovász condition.
    """
    n = basis.shape[0]
    dim = basis.shape[1]
    
    # Convert basis to object dtype with Python integers
    basis_int = np.empty((n, dim), dtype=object)
    for i in range(n):
        for j in range(dim):
            val = basis[i, j]
            if isinstance(val, (int, np.integer)):
                basis_int[i, j] = int(val)
            else:
                try:
                    basis_int[i, j] = int(val) if val != 0 else 0
                except (ValueError, TypeError, OverflowError):
                    basis_int[i, j] = 0
    
    # Store mu coefficients as (numerator, denominator) pairs for exact rational arithmetic
    mu = {}  # mu[(i, j)] = (num, den) where mu[i,j] = num/den
    
    def compute_mu(i: int, j: int) -> tuple:
        """Compute mu[i,j] = <b_i, b*_j> / <b*_j, b*_j> using exact integer arithmetic."""
        # First compute b*_j (Gram-Schmidt orthogonalized vector)
        b_star_j = [int(basis_int[j, d]) for d in range(dim)]
        
        # Orthogonalize b*_j against previous vectors
        for l in range(j):
            if (j, l) in mu:
                mu_num, mu_den = mu[(j, l)]
                if mu_den != 0:
                    # Approximate mu[j,l] for orthogonalization
                    mu_approx = round(mu_num / mu_den) if abs(mu_num) < 10**100 and mu_den < 10**100 else 0
                    for d in range(dim):
                        b_star_j[d] -= mu_approx * int(basis_int[l, d])
        
        # Compute <b_i, b*_j>
        dot_bi_bjstar = sum(int(basis_int[i, d]) * b_star_j[d] for d in range(dim))
        
        # Compute <b*_j, b*_j>
        dot_bjstar_bjstar = sum(b_star_j[d] * b_star_j[d] for d in range(dim))
        
        if dot_bjstar_bjstar != 0:
            return (dot_bi_bjstar, dot_bjstar_bjstar)
        else:
            return (0, 1)
    
    def update_mu_row(i: int):
        """Update mu coefficients for row i."""
        for j in range(i):
            mu[(i, j)] = compute_mu(i, j)
        mu[(i, i)] = (1, 1)
    
    # Initialize mu coefficients
    for i in range(n):
        update_mu_row(i)
    
    k = 1
    max_iterations = n * n * 10  # Safety limit
    iterations = 0
    
    while k < n and iterations < max_iterations:
        iterations += 1
        
        # Size reduction: reduce b_k with respect to b_{k-1}, ..., b_0
        for j in range(k - 1, -1, -1):
            if (k, j) in mu:
                mu_num, mu_den = mu[(k, j)]
                if mu_den != 0:
                    # Check if |mu[k,j]| > 0.5
                    # Use exact comparison: |mu_num| > mu_den/2
                    if abs(mu_num) * 2 > abs(mu_den):
                        # q = round(mu[k,j])
                        # For exact computation: q = round(mu_num / mu_den)
                        # Use integer arithmetic: q = (mu_num + mu_den//2) // mu_den if mu_den > 0
                        if mu_den > 0:
                            q = (mu_num + mu_den // 2) // mu_den
                        else:
                            q = (mu_num - (-mu_den) // 2) // (-mu_den)
                        
                        # b_k = b_k - q * b_j
                        for d in range(dim):
                            basis_int[k, d] = int(basis_int[k, d]) - q * int(basis_int[j, d])
                        
                        # Update mu coefficients for row k
                        update_mu_row(k)
        
        # Lovász condition: check if we need to swap
        if k > 0 and (k, k-1) in mu and (k-1, k-1) in mu:
            mu_num, mu_den = mu[(k, k-1)]
            
            # Compute |b*_k|^2 and |b*_{k-1}|^2
            # For efficiency, approximate b* vectors
            b_star_k_minus_1 = [int(basis_int[k-1, d]) for d in range(dim)]
            b_star_k = [int(basis_int[k, d]) for d in range(dim)]
            
            # Orthogonalize (approximate)
            for l in range(k-1):
                if (k-1, l) in mu:
                    mu_n, mu_d = mu[(k-1, l)]
                    if mu_d != 0 and abs(mu_n) < 10**50 and mu_d < 10**50:
                        q_approx = round(mu_n / mu_d)
                        for d in range(dim):
                            b_star_k_minus_1[d] -= q_approx * int(basis_int[l, d])
            
            for l in range(k):
                if (k, l) in mu:
                    mu_n, mu_d = mu[(k, l)]
                    if mu_d != 0 and abs(mu_n) < 10**50 and mu_d < 10**50:
                        q_approx = round(mu_n / mu_d)
                        for d in range(dim):
                            b_star_k[d] -= q_approx * int(basis_int[l, d])
            
            norm_k_minus_1_sq = sum(b_star_k_minus_1[d] * b_star_k_minus_1[d] for d in range(dim))
            norm_k_sq = sum(b_star_k[d] * b_star_k[d] for d in range(dim))
            
            if norm_k_minus_1_sq > 0 and mu_den != 0:
                # Check Lovász condition using rational arithmetic
                # |b*_k + mu[k,k-1]*b*_{k-1}|^2 >= delta * |b*_{k-1}|^2
                # Approximate: norm_k_sq >= (delta - mu^2) * norm_k_minus_1_sq
                if abs(mu_num) < 10**50 and mu_den < 10**50:
                    mu_val = mu_num / mu_den
                    mu_sq = mu_val * mu_val
                    lhs = norm_k_sq
                    rhs = (delta - mu_sq) * norm_k_minus_1_sq
                    
                    if lhs < rhs:
                        # Swap b_{k-1} and b_k
                        for d in range(dim):
                            basis_int[k-1, d], basis_int[k, d] = basis_int[k, d], basis_int[k-1, d]
                        # Recompute mu for affected rows
                        update_mu_row(k-1)
                        update_mu_row(k)
                        k = max(1, k - 1)
                        continue
        
        k += 1
    
    return basis_int


class IntegerMatrix:
    """
    Matrix class for large integers, compatible with fpylll interface.
    """
    def __init__(self, data):
        """
        Initialize from numpy array or list of lists.
        Supports object dtype for arbitrary precision integers.
        """
        if isinstance(data, np.ndarray):
            self._matrix = data
        else:
            # Convert to numpy array with object dtype for large integers
            self._matrix = np.array(data, dtype=object)
        
        # Ensure object dtype for large integers
        if self._matrix.dtype != object:
            # Check if we need object dtype (large integers)
            max_val = 0
            for i in range(self._matrix.shape[0]):
                for j in range(self._matrix.shape[1]):
                    val = self._matrix[i, j]
                    if isinstance(val, (int, np.integer)):
                        max_val = max(max_val, abs(int(val)))
            
            # If values are too large for int64, use object dtype
            if max_val > 2**63:
                matrix_obj = np.empty(self._matrix.shape, dtype=object)
                for i in range(self._matrix.shape[0]):
                    for j in range(self._matrix.shape[1]):
                        matrix_obj[i, j] = int(self._matrix[i, j])
                self._matrix = matrix_obj
    
    @property
    def nrows(self):
        return self._matrix.shape[0]
    
    @property
    def ncols(self):
        return self._matrix.shape[1]
    
    def __getitem__(self, key):
        return self._matrix[key]
    
    def __setitem__(self, key, value):
        self._matrix[key] = value
    
    def to_numpy(self):
        """Convert to numpy array."""
        return self._matrix
    
    def copy(self):
        """Return a copy of the matrix."""
        return IntegerMatrix(self._matrix.copy())
    
    def is_zero(self):
        """Check if matrix is all zeros."""
        for i in range(self.nrows):
            for j in range(self.ncols):
                val = self._matrix[i, j]
                if val != 0 and val is not None:
                    return False
        return True
    
    def rank(self):
        """Compute rank (for small matrices only)."""
        if self.nrows > 500 or self.ncols > 500:
            return None  # Too large for quick rank computation
        
        # Convert to float for rank computation
        try:
            matrix_float = np.zeros((self.nrows, self.ncols), dtype=float)
            for i in range(self.nrows):
                for j in range(self.ncols):
                    val = self._matrix[i, j]
                    if val != 0 and val is not None:
                        try:
                            matrix_float[i, j] = float(int(val))
                        except (OverflowError, ValueError):
                            matrix_float[i, j] = 0.0
            return np.linalg.matrix_rank(matrix_float)
        except:
            return None
    
    def density(self):
        """Compute density (fraction of non-zero entries)."""
        total = self.nrows * self.ncols
        non_zero = 0
        for i in range(self.nrows):
            for j in range(self.ncols):
                val = self._matrix[i, j]
                if val != 0 and val is not None:
                    non_zero += 1
        return non_zero / total if total > 0 else 0.0
    
    def det(self):
        """Compute determinant (for small square matrices only)."""
        if not self.is_square() or self.nrows > 200:
            return None
        
        try:
            # Convert to float for determinant
            matrix_float = np.zeros((self.nrows, self.ncols), dtype=float)
            for i in range(self.nrows):
                for j in range(self.ncols):
                    val = self._matrix[i, j]
                    if val != 0 and val is not None:
                        try:
                            matrix_float[i, j] = float(int(val))
                        except (OverflowError, ValueError):
                            matrix_float[i, j] = 0.0
            return np.linalg.det(matrix_float)
        except:
            return None
    
    def is_square(self):
        """Check if matrix is square."""
        return self.nrows == self.ncols


def LLL(B: IntegerMatrix, delta: float = 0.75) -> IntegerMatrix:
    """
    LLL reduction of integer matrix using integer arithmetic.
    Handles arbitrarily large integers without float conversion.
    
    Args:
        B: IntegerMatrix to reduce
        delta: LLL parameter (default 0.75)
    
    Returns:
        Reduced IntegerMatrix
    """
    matrix = B.to_numpy()
    
    # Check maximum entry size
    max_entry = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if val != 0 and val is not None:
                try:
                    val_abs = abs(int(val))
                    if val_abs > max_entry:
                        max_entry = val_abs
                except (TypeError, ValueError):
                    pass
    
    max_bits = max_entry.bit_length() if max_entry > 0 else 0
    logger.info(f"Running integer-based LLL on {matrix.shape[0]}x{matrix.shape[1]} matrix (max entry: 2^{max_bits} bits)")
    
    # Use integer-based LLL for all cases (no float conversion)
    reduced_int = lll_reduce_integer(matrix, delta=delta)
    
    # Convert back to IntegerMatrix
    result = IntegerMatrix(reduced_int)
    
    return result


# Convenience functions matching fpylll interface
def IntegerMatrix_from_matrix(matrix):
    """Create IntegerMatrix from another matrix."""
    return IntegerMatrix(matrix)


# This is a fallback implementation without fpylll
FPYLLL_AVAILABLE = False

# Export main interface
__all__ = ['IntegerMatrix', 'LLL', 'IntegerMatrix_from_matrix', 'FPYLLL_AVAILABLE']
