#!/usr/bin/env python3
"""
Vesica Piscis LLL: A geometric lattice reduction through circular projection.

The algorithm literally projects the lattice basis onto a circle, then:
1. FOLD: Circle folds horizontally, creating vesica piscis - vectors organize themselves
2. EXPAND: As vesica expands back to circle, the southern wave pulls outliers inward
3. COMPRESS: Circle compresses to singularity, contracting all vectors

The transformation happens in a geometric space where angular and radial positions
determine reduction operations.
"""

import numpy as np
from typing import Tuple, List
import math
import logging

logger = logging.getLogger(__name__)


def project_to_circle(vector: np.ndarray, radius: float = 1.0) -> Tuple[float, float]:
    """
    Project a lattice vector onto a circle.
    
    Returns:
        (angle, magnitude) - position on circle
    """
    # Compute vector norm
    norm = math.sqrt(sum(int(v)**2 for v in vector))
    if norm == 0:
        return (0.0, 0.0)
    
    # Angle from first two dimensions (or hash of all dimensions)
    angle = math.atan2(float(vector[1] if len(vector) > 1 else 0), 
                       float(vector[0]))
    
    # Magnitude (radial distance from origin)
    magnitude = norm
    
    return (angle, magnitude)


def fold_circle_to_vesica(circle_points: List[Tuple[int, float, float]], 
                         fold_amount: float) -> List[Tuple[int, float, float, bool]]:
    """
    Fold the circle horizontally to create vesica piscis.
    
    fold_amount: 0.0 (full circle) to 1.0 (completely folded)
    
    Returns:
        List of (index, angle, magnitude, in_vesica) where in_vesica indicates
        if the vector is in the overlapping vesica region
    """
    vesica_points = []
    
    # Folding creates two overlapping circles
    # The vesica is where they overlap (angles near 0, ±π)
    for idx, angle, mag in circle_points:
        # Normalize angle to [-π, π]
        norm_angle = ((angle + math.pi) % (2 * math.pi)) - math.pi
        
        # Distance from fold line (horizontal axis)
        # Vectors near 0° and 180° are in the vesica overlap region
        dist_from_fold = min(abs(norm_angle), abs(abs(norm_angle) - math.pi))
        
        # In vesica if within the folded region
        vesica_threshold = math.pi * (1.0 - fold_amount)
        in_vesica = dist_from_fold < vesica_threshold
        
        vesica_points.append((idx, angle, mag, in_vesica))
    
    return vesica_points


def compute_wave_displacement(angle: float, fold_amount: float) -> float:
    """
    Compute how far the 'southern wave' displaces vectors outward.
    
    The wave is strongest at 90° and 270° (perpendicular to fold).
    As fold_amount increases, the wave pushes vectors outward.
    """
    # Normalize angle
    norm_angle = ((angle + math.pi) % (2 * math.pi)) - math.pi
    
    # Wave is strongest perpendicular to fold (at ±π/2)
    wave_strength = abs(math.sin(norm_angle))
    
    # Displacement increases with fold amount
    displacement = wave_strength * fold_amount * 2.0
    
    return displacement


def vesica_piscis_lll(basis: np.ndarray, delta: float = 0.75, 
                     n_cycles: int = 20) -> np.ndarray:
    """
    Geometric LLL reduction through circular projection and transformation.
    
    Process:
    1. Project all vectors onto a circle (by angle and magnitude)
    2. FOLD circle → vesica piscis (organize vectors)
    3. EXPAND vesica → circle (wave drags outliers inward)
    4. COMPRESS circle → singularity (contract all vectors)
    
    Args:
        basis: Input lattice basis (rows are basis vectors)
        delta: LLL reduction parameter
        n_cycles: Number of fold-expand-compress cycles
    
    Returns:
        Reduced basis
    """
    n = basis.shape[0]
    dim = basis.shape[1]
    
    # Convert to object dtype for arbitrary precision
    basis_int = np.empty((n, dim), dtype=object)
    for i in range(n):
        for j in range(dim):
            val = basis[i, j]
            basis_int[i, j] = int(val) if val != 0 else 0
    
    logger.info(f"Starting Vesica Piscis LLL with {n} vectors in {dim} dimensions")
    
    for cycle in range(n_cycles):
        logger.info(f"\n=== CYCLE {cycle + 1}/{n_cycles} ===")
        
        # === STEP 1: PROJECT TO CIRCLE ===
        circle_points = []
        for i in range(n):
            angle, mag = project_to_circle(basis_int[i])
            circle_points.append((i, angle, mag))
            logger.debug(f"Vector {i}: angle={angle:.3f}, magnitude={mag:.2e}")
        
        # === STEP 2: FOLD - Create Vesica Piscis ===
        logger.info("FOLDING: Circle → Vesica Piscis")
        
        # Fold amount increases in early cycles, then decreases
        fold_progress = cycle / (n_cycles - 1) if n_cycles > 1 else 0.5
        if fold_progress < 0.5:
            fold_amount = fold_progress * 2  # 0 → 1
        else:
            fold_amount = 2 - fold_progress * 2  # 1 → 0
        
        vesica_points = fold_circle_to_vesica(circle_points, fold_amount)
        
        # ORGANIZE: Vectors in vesica region should be size-reduced against each other
        vesica_indices = [idx for idx, _, _, in_v in vesica_points if in_v]
        logger.info(f"Vesica contains {len(vesica_indices)} vectors")
        
        for i in vesica_indices:
            for j in vesica_indices:
                if i > j:
                    # Size reduce basis_int[i] with respect to basis_int[j]
                    dot_product = sum(int(basis_int[i, d]) * int(basis_int[j, d]) 
                                    for d in range(dim))
                    norm_sq_j = sum(int(basis_int[j, d])**2 for d in range(dim))
                    
                    if norm_sq_j > 0:
                        mu = dot_product / norm_sq_j
                        if abs(mu) > 0.5:
                            q = round(mu)
                            for d in range(dim):
                                basis_int[i, d] = int(basis_int[i, d]) - q * int(basis_int[j, d])
        
        # === STEP 3: EXPAND - Vesica → Circle, Wave Drags Inward ===
        logger.info("EXPANDING: Vesica → Circle (wave drags outliers)")
        
        # Re-project after size reduction
        circle_points = []
        for i in range(n):
            angle, mag = project_to_circle(basis_int[i])
            circle_points.append((i, angle, mag))
        
        # Compute wave displacements
        wave_displaced = []
        for idx, angle, mag in circle_points:
            displacement = compute_wave_displacement(angle, fold_amount)
            effective_mag = mag * (1.0 + displacement)
            wave_displaced.append((idx, angle, mag, effective_mag))
        
        # Vectors with high effective magnitude (pushed by wave) need to be pulled back
        # This is done by swapping them with shorter vectors (Lovász condition)
        for k in range(1, n):
            # Get current and previous vector magnitudes
            curr_mag = wave_displaced[k][3]  # effective magnitude with wave
            prev_mag = wave_displaced[k-1][3]
            
            # Check Lovász-style condition with wave consideration
            if curr_mag < prev_mag * math.sqrt(delta):
                # SWAP: Pull the outlier (wave-displaced vector) inward
                logger.debug(f"SWAP {k-1} ↔ {k} (wave pulling inward)")
                for d in range(dim):
                    basis_int[k-1, d], basis_int[k, d] = basis_int[k, d], basis_int[k-1, d]
        
        # === STEP 4: COMPRESS - Circle → Singularity ===
        logger.info("COMPRESSING: Circle → Singularity")
        
        # Compute all vector norms
        norms = []
        for i in range(n):
            norm_sq = sum(int(basis_int[i, d])**2 for d in range(dim))
            norms.append((i, norm_sq))
        
        # Sort by norm to find shortest vectors
        norms.sort(key=lambda x: x[1])
        
        # Compress: move shortest vectors to front (toward singularity/center)
        compression_strength = 1.0 - abs(fold_amount - 0.5) * 2  # Peak at fold_amount=0.5
        
        if compression_strength > 0.3:
            # Find shortest vector not already at front
            for rank, (idx, norm_sq) in enumerate(norms):
                if idx > rank:
                    # Move this short vector toward its natural position
                    # Swap with vector currently at its rank position
                    target = rank
                    current = idx
                    
                    if current > target:
                        logger.debug(f"COMPRESS: moving vector {current} → {target}")
                        # Bubble it forward
                        for pos in range(current, target, -1):
                            for d in range(dim):
                                basis_int[pos, d], basis_int[pos-1, d] = \
                                    basis_int[pos-1, d], basis_int[pos, d]
                        break  # Only one move per cycle for stability
        
        # Log cycle statistics
        total_norm = sum(math.sqrt(n_sq) for _, n_sq in norms)
        avg_norm = total_norm / n
        logger.info(f"Cycle {cycle + 1} complete. Avg vector norm: {avg_norm:.2e}")
    
    logger.info("\nVesica Piscis LLL reduction complete")
    return basis_int


class IntegerMatrix:
    """Matrix class for large integers, compatible with fpylll interface."""
    
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            self._matrix = data
        else:
            self._matrix = np.array(data, dtype=object)
        
        if self._matrix.dtype != object:
            max_val = 0
            for i in range(self._matrix.shape[0]):
                for j in range(self._matrix.shape[1]):
                    val = self._matrix[i, j]
                    if isinstance(val, (int, np.integer)):
                        max_val = max(max_val, abs(int(val)))
            
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
        return self._matrix
    
    def copy(self):
        return IntegerMatrix(self._matrix.copy())


def LLL(B: IntegerMatrix, delta: float = 0.75) -> IntegerMatrix:
    """
    Vesica Piscis LLL reduction through geometric circular transformation.
    
    Args:
        B: IntegerMatrix to reduce
        delta: LLL parameter (default 0.75)
    
    Returns:
        Reduced IntegerMatrix
    """
    matrix = B.to_numpy()
    
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
    logger.info(f"Vesica Piscis LLL: {matrix.shape[0]}x{matrix.shape[1]} matrix (max: 2^{max_bits} bits)")
    
    reduced = vesica_piscis_lll(matrix, delta=delta)
    
    return IntegerMatrix(reduced)


def IntegerMatrix_from_matrix(matrix):
    """Create IntegerMatrix from another matrix."""
    return IntegerMatrix(matrix)


FPYLLL_AVAILABLE = False

__all__ = ['IntegerMatrix', 'LLL', 'IntegerMatrix_from_matrix', 'FPYLLL_AVAILABLE']
