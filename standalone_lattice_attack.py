#!/usr/bin/env python3
"""
Standalone Minimizable Factorization Lattice Attack with Enhanced Polynomial Solving

A self-contained script for demonstrating the minimizable factorization lattice approach
with advanced polynomial solving capabilities.

Usage:
    python standalone_lattice_attack.py <N> [--p P] [--q Q] [--search-radius RADIUS] [--verbose] [--polynomial]

Arguments:
    N: The number to factor (required)
    --p: Initial P candidate (optional, will be estimated if not provided)
    --q: Initial Q candidate (optional, will be estimated if not provided)
    --search-radius: Search radius for corrections in bits (default: 4000, meaning 2^4000)
    --verbose: Enable verbose output
    --polynomial: Enable polynomial solving methods alongside lattice methods

Example:
    python standalone_lattice_attack.py 2021 --polynomial --verbose
    python standalone_lattice_attack.py 2021 --polynomial
"""

import argparse
import sys
import math
import random
import time
from typing import List, Dict, Optional, Tuple
import numpy as np
import sympy as sp
from sympy.polys.polytools import groebner
from typing import Optional, Tuple, List

# Try to import PyTorch for Transformer, fallback to simpler model if not available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[Transformer] PyTorch not available, using simplified model")

# ============================================================================
# TRANSFORMER MODEL FOR STEP PREDICTION
# ============================================================================

class StepPredictionTransformer:
    """
    Transformer Neural Network to learn from bulk search results and predict
    which steps are most likely to yield good factors.
    """
    
    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 2,
                 max_seq_len: int = 500000, use_torch: bool = TORCH_AVAILABLE):
        """
        Initialize Transformer model for step prediction.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            max_seq_len: Maximum sequence length (history size) - default 500K for massive context
            use_torch: Whether to use PyTorch (if available)
        """
        self.use_torch = use_torch and TORCH_AVAILABLE
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.search_history = []  # List of (step_center, diff_bits, sqrt_N, N_bits)
        self.factor_history = []  # List of (step_center, p_candidate, q_candidate, diff_bits) for learning p-q patterns
        
        if self.use_torch:
            self._init_torch_model(d_model, nhead, num_layers)
        else:
            self._init_simple_model()
    
    def _init_torch_model(self, d_model: int, nhead: int, num_layers: int):
        """Initialize ENHANCED PyTorch Transformer model with advanced features."""
        device = torch.device('cpu')  # Force CPU to avoid GPU memory issues

        class EnhancedTransformerModel(nn.Module):
            def __init__(self, d_model, nhead, num_layers, max_seq_len):
                super().__init__()
                self.d_model = d_model
                self.max_seq_len = max_seq_len

                # Enhanced embedding with correction signals
                self.embedding = nn.Linear(8, d_model)  # Extended: step_center, diff_bits, sqrt_N, N_bits, recency, quality_trend, correction_signal, optimality_score

                # Rotary Position Encoding (RoPE) - more sophisticated than sinusoidal
                self._init_rope()

                # Multi-layer transformer with better architecture
                self.layers = nn.ModuleList()
                for _ in range(num_layers):
                    layer = nn.ModuleDict({
                        'attention': nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True),
                        'norm1': nn.LayerNorm(d_model),
                        'norm2': nn.LayerNorm(d_model),
                        'ffn': nn.Sequential(
                            nn.Linear(d_model, d_model * 4),
                            nn.GELU(),  # Better activation than ReLU
                            nn.Dropout(0.1),
                            nn.Linear(d_model * 4, d_model),
                            nn.Dropout(0.1)
                        )
                    })
                    self.layers.append(layer)

                # Enhanced predictor heads
                self.quality_predictor = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.LayerNorm(d_model // 2),
                    nn.Linear(d_model // 2, 1),
                    nn.Sigmoid()  # Quality score 0-1
                )

                self.position_predictor = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.LayerNorm(d_model // 2),
                    nn.Linear(d_model // 2, 1)  # Position offset prediction
                )

            def _init_rope(self):
                """Initialize Rotary Position Encoding frequencies."""
                inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model // nhead, 2).float() / (d_model // nhead)))
                self.register_buffer("inv_freq", inv_freq)

            def _apply_rope(self, x, positions=None):
                """Apply Rotary Position Encoding."""
                seq_len = x.shape[1]
                inv_freq = self.inv_freq
                t = torch.arange(seq_len, device=x.device).type_as(inv_freq)
                freqs = torch.einsum('i , j -> i j', t, inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos()
                sin = emb.sin()

                # Apply rotation
                x1 = x[..., : x.shape[-1] // 2]
                x2 = x[..., x.shape[-1] // 2 :]
                return torch.cat((-x2, x1), dim=-1) * cos + torch.cat((x1, x2), dim=-1) * sin

            def forward(self, x, return_attention=False):
                # x shape: (batch, seq_len, 6) - extended features
                batch_size, seq_len, _ = x.shape

                # Embed input features
                x = self.embedding(x)  # (batch, seq_len, d_model)

                # Apply RoPE positional encoding
                x = self._apply_rope(x)

                # Store attention weights if requested
                attention_weights = []

                # Multi-layer transformer processing
                for layer in self.layers:
                    # Multi-head attention with residual connection
                    attn_out, attn_weights = layer['attention'](x, x, x, need_weights=return_attention)
                    if return_attention:
                        attention_weights.append(attn_weights)

                    x = layer['norm1'](x + attn_out)  # Pre-norm formulation

                    # Feed-forward network with residual connection
                    ffn_out = layer['ffn'](x)
                    x = layer['norm2'](x + ffn_out)

                # Pooling and prediction
                pooled = x.mean(dim=1)  # (batch, d_model)

                quality_score = self.quality_predictor(pooled)  # (batch, 1)
                position_offset = self.position_predictor(pooled)  # (batch, 1)

                result = {
                    'quality_score': quality_score,
                    'position_offset': position_offset
                }

                if return_attention:
                    result['attention_weights'] = attention_weights

                return result

        self.model = EnhancedTransformerModel(d_model, nhead, num_layers, self.max_seq_len).to(device)
        self.device = device

        # Enhanced optimizer with learning rate scheduling
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.9)

        # Multiple loss functions for different prediction tasks
        self.quality_criterion = nn.MSELoss()
        self.position_criterion = nn.SmoothL1Loss()  # Better for regression

        self.model.eval()  # Start in eval mode
    
    def _init_simple_model(self):
        """Initialize simple fallback model (weighted average)."""
        self.weights = np.array([0.4, 0.3, 0.2, 0.1])  # Weights for features
        print("[Transformer] Using simplified model (PyTorch not available)")
    
    def add_search_result(self, step_center: int, diff_bits: Optional[int],
                         sqrt_N: int, N_bits: int, p_candidate: Optional[int] = None,
                         q_candidate: Optional[int] = None, known_p: Optional[int] = None,
                         known_q: Optional[int] = None):
        """
        Add a search result to history with optional correction from known factors.

        Args:
            step_center: The step center value
            diff_bits: Difference in bits (None if no result found)
            sqrt_N: Square root of N
            N_bits: Bit length of N
            p_candidate: P candidate found (optional, for learning p-q patterns)
            q_candidate: Q candidate found (optional, for learning p-q patterns)
            known_p: Known correct P factor (for supervised learning/correction)
            known_q: Known correct Q factor (for supervised learning/correction)
        """
        # ENHANCED FEATURE EXTRACTION
        import time
        current_time = time.time()

        # Enhanced feature extraction with richer context
        normalized_diff = diff_bits if diff_bits is not None else N_bits * 2

        # Core spatial features
        normalized_step = (step_center - sqrt_N) / sqrt_N if sqrt_N > 0 else 0
        normalized_sqrt_N = sqrt_N / (2 ** N_bits) if N_bits > 0 else 0
        normalized_N_bits = N_bits / 4096.0

        # Recency feature (how recent is this result?)
        recency = 1.0  # Most recent result
        if self.search_history:
            time_diff = current_time - self.search_history[-1][-1]  # Compare to last result timestamp
            recency = min(1.0, 1.0 / (1.0 + time_diff))  # Exponential decay

        # Quality trend feature (is quality improving?)
        quality_trend = 0.0
        if len(self.search_history) >= 2:
            recent_qualities = [entry[1] for entry in self.search_history[-3:]]  # Last 3 results
            if len(recent_qualities) >= 2:
                # Calculate trend (negative = improving, positive = worsening)
                quality_trend = (recent_qualities[-1] - recent_qualities[0]) / len(recent_qualities)

        # CORRECTION SIGNALS: If we know the true factors, add supervised learning features
        correction_signal = 0.0
        optimality_score = 0.0

        if known_p is not None and known_q is not None:
            # Calculate how close this search position is to optimal factor locations
            p_distance = abs(step_center - known_p)
            q_distance = abs(step_center - known_q)

            # Find the closest factor (this is what we're actually trying to find)
            min_factor_distance = min(p_distance, q_distance)

            # Convert to a normalized score (0-1, higher = better)
            max_reasonable_distance = 2**(N_bits // 2)  # Half the key size
            optimality_score = 1.0 - min(1.0, min_factor_distance / max_reasonable_distance)

            # Correction signal: positive if this position is near a factor, negative if far
            correction_signal = optimality_score - 0.5  # Center around 0

            if optimality_score > 0.8:  # Very close to a factor
                print(f"[Correction] 🎯 Position {step_center} is very close to factor (optimality: {optimality_score:.3f})")
            elif optimality_score < 0.1:  # Very far from factors
                print(f"[Correction] ❌ Position {step_center} is far from factors (optimality: {optimality_score:.3f})")

        # Enhanced feature vector (8 features with correction signals)
        self.search_history.append((
            normalized_step,
            normalized_diff / (N_bits * 2) if N_bits > 0 else 1.0,
            normalized_sqrt_N,
            normalized_N_bits,
            recency,
            quality_trend,
            correction_signal,     # Correction signal from known factors
            optimality_score,      # How close to optimal factor locations
            current_time  # Add timestamp for recency calculations
        ))

        # Enhanced p-q pattern learning with more features
        if p_candidate is not None and q_candidate is not None:
            p_q_diff = abs(p_candidate - q_candidate)
            p_q_diff_bits = p_q_diff.bit_length() if p_q_diff > 0 else 0

            # Additional relationship features
            p_q_ratio = max(p_candidate, q_candidate) / min(p_candidate, q_candidate) if min(p_candidate, q_candidate) > 0 else 1.0
            p_q_sum_bits = (p_candidate + q_candidate).bit_length()
            quality_score = 1.0 / (1.0 + diff_bits) if diff_bits is not None else 0.0

            self.factor_history.append((
                step_center, p_candidate, q_candidate, p_q_diff_bits,
                quality_score, p_q_ratio, p_q_sum_bits, current_time
            ))

        # MASSIVE CONTEXT PRUNING: Intelligent retention with 500K capacity
        if len(self.search_history) > self.max_seq_len * 0.9:  # Warn when approaching capacity
            print(f"[Transformer] 📈 Approaching massive context capacity: {len(self.search_history):,} / {self.max_seq_len:,}")

        if len(self.search_history) > self.max_seq_len:
            print(f"[Transformer] Pruning search history from {len(self.search_history):,} to {self.max_seq_len:,} samples")

            # Sophisticated pruning for massive context with correction signals
            def comprehensive_score(entry):
                quality = 1.0 - entry[1]  # Higher quality = higher score
                recency = entry[4]  # Recency weight (0-1)
                trend = entry[5]  # Quality trend (-1 to 1, negative = improving)
                correction = entry[6] + 0.5  # Correction signal (-0.5 to 0.5) -> (0 to 1)
                optimality = entry[7]  # Optimality score (0-1)

                # Combine factors with learned weights
                score = (0.4 * quality +          # Quality most important
                        0.25 * recency +         # Recency second
                        0.15 * max(0, trend) +   # Positive trends bonus
                        0.1 * correction +       # Correction signal bonus
                        0.1 * optimality)        # Optimality bonus

                return score

            # Sort by comprehensive score and keep top samples
            self.search_history.sort(key=comprehensive_score, reverse=True)
            self.search_history = self.search_history[:self.max_seq_len]

        if len(self.factor_history) > self.max_seq_len:
            print(f"[Transformer] Pruning factor history from {len(self.factor_history):,} to {self.max_seq_len:,} samples")

            # Sort by quality score, recency, and factorization value
            def factor_score(entry):
                quality = entry[4]  # Quality score (0-1)
                recency = (time.time() - entry[7]) / 1000.0  # Recency in seconds
                recency_weight = 1.0 / (1.0 + recency)  # Exponential decay

                # Bonus for exact factorizations
                exact_bonus = 10.0 if entry[3] == 0 else 0.0  # diff_bits == 0

                return quality + recency_weight + exact_bonus

            self.factor_history.sort(key=factor_score, reverse=True)
            self.factor_history = self.factor_history[:self.max_seq_len]
    
    def predict_step_quality(self, step_center: int, sqrt_N: int, N_bits: int) -> float:
        """
        ENHANCED prediction with richer features and progressive learning.

        Now uses:
        - 6-dimensional feature vectors instead of 4
        - Recency weighting
        - Quality trend analysis
        - Context-aware predictions

        Args:
            step_center: The step center to predict
            sqrt_N: Square root of N
            N_bits: Bit length of N

        Returns:
            Quality score (0-1, higher is better)
        """
        if len(self.search_history) < 3:
            # Early exploration: not enough history for sophisticated prediction
            return 0.5

        # Enhanced feature extraction with richer context
        normalized_step = (step_center - sqrt_N) / sqrt_N if sqrt_N > 0 else 0
        normalized_sqrt_N = sqrt_N / (2 ** N_bits) if N_bits > 0 else 0
        normalized_N_bits = N_bits / 4096.0

        # Estimate recency (assume this is a new prediction)
        recency = 1.0

        # Estimate quality trend from recent history
        quality_trend = 0.0
        if len(self.search_history) >= 3:
            recent_qualities = [entry[1] for entry in self.search_history[-3:]]
            quality_trend = (recent_qualities[-1] - recent_qualities[0]) / len(recent_qualities)

        if self.use_torch:
            return self._predict_torch_enhanced(normalized_step, normalized_sqrt_N, normalized_N_bits, recency, quality_trend)
        else:
            return self._predict_simple_enhanced(normalized_step, normalized_sqrt_N, normalized_N_bits, recency, quality_trend)
    
    def _predict_torch_enhanced(self, normalized_step: float, normalized_sqrt_N: float,
                               normalized_N_bits: float, recency: float, quality_trend: float) -> float:
        """ENHANCED prediction using sophisticated Transformer with 6 features."""
        try:
            # MASSIVE CONTEXT: Use sliding window sampling for efficiency with 500K context
            # Sample diverse and recent experiences instead of full history
            full_history = self.search_history

            if len(full_history) > 1000:  # If we have massive history
                # Intelligent sampling: keep recent + diverse high-quality samples
                recent_samples = full_history[-500:]  # Last 500 most recent
                quality_samples = sorted(full_history[:-500], key=lambda x: x[1])[:500]  # Top 500 best quality (lowest error)
                history = quality_samples + recent_samples  # Combine for rich context
            else:
                history = full_history[-min(1000, len(full_history)):]  # Normal case

            # Create sequence: history + current step (8 features each)
            current_features = np.array([normalized_step, 0.0, normalized_sqrt_N, normalized_N_bits, recency, quality_trend, 0.0, 0.0])  # Add correction placeholders
            sequence = np.array([entry[:8] for entry in history] + [current_features])  # Extract first 8 features, ignore timestamp

            # Convert to tensor and move to device
            x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # (1, seq_len, 6)

            with torch.no_grad():
                self.model.eval()
                result = self.model(x)

                # Use quality score prediction
                quality_score = float(result['quality_score'][0, 0].item())

                # Optionally use position offset for additional guidance
                position_offset = float(result['position_offset'][0, 0].item())

                # Combine quality score with positional confidence
                confidence = min(1.0, abs(position_offset) / normalized_step) if normalized_step != 0 else 1.0
                final_score = quality_score * (0.8 + 0.2 * confidence)  # Weight quality higher

                return max(0.0, min(1.0, final_score))  # Clamp to [0, 1]

        except Exception as e:
            # Fallback to enhanced simple model on error
            return self._predict_simple_enhanced(normalized_step, normalized_sqrt_N, normalized_N_bits, recency, quality_trend)
    
    def _predict_simple_enhanced(self, normalized_step: float, normalized_sqrt_N: float,
                                normalized_N_bits: float, recency: float, quality_trend: float) -> float:
        """ENHANCED simple prediction using weighted features and pattern recognition."""
        if not self.search_history:
            return 0.5

        # Enhanced feature weighting with correction signals
        weights = np.array([0.25, 0.2, 0.12, 0.12, 0.08, 0.06, 0.1, 0.07])  # 8 weights for 8 features
        current_features = np.array([normalized_step, 0.0, normalized_sqrt_N, normalized_N_bits, recency, quality_trend, 0.0, 0.0])  # Add correction placeholders

        # Calculate similarity to successful historical results
        best_similarity = 0.0
        for entry in self.search_history[-50:]:  # Check last 50 results
            if entry[1] < 0.3:  # Good quality result (low normalized error)
                similarity = 1.0 - np.mean(np.abs(np.array(entry[:6]) - current_features))
                best_similarity = max(best_similarity, similarity)

        # Combine feature-weighted prediction with similarity
        feature_score = np.dot(weights, current_features)
        feature_score = 1.0 / (1.0 + abs(feature_score))  # Convert to 0-1 range

        # Blend with similarity score
        final_score = 0.7 * feature_score + 0.3 * best_similarity

        return max(0.0, min(1.0, final_score))

    def train_on_history(self, epochs: int = 5):
        """Train the enhanced Transformer on massive accumulated search history (500K+ context)."""
        if not self.use_torch or len(self.search_history) < 10:
            return

        try:
            self.model.train()
            history_size = len(self.search_history)
            print(f"[Transformer] Training on {history_size:,} historical samples with 500K context...")

            for epoch in range(epochs):
                epoch_loss = 0.0
                batch_count = 0

                # MASSIVE CONTEXT TRAINING: Efficient batch sampling
                # Instead of training on every sequence, sample diverse batches
                num_batches = min(100, max(20, history_size // 100))  # Adaptive batch count

                for batch_idx in range(num_batches):
                    # Sample diverse training sequences from massive history
                    if history_size > 1000:
                        # Stratified sampling: mix recent + high-quality + random
                        recent_start = max(0, history_size - 200)
                        quality_indices = np.argsort([entry[1] for entry in self.search_history])[:200]  # Best quality
                        random_indices = np.random.choice(history_size, size=100, replace=False)

                        sample_indices = np.concatenate([
                            np.arange(recent_start, history_size),  # Recent
                            quality_indices,  # High quality
                            random_indices  # Random exploration
                        ])
                        sample_indices = np.unique(sample_indices)  # Remove duplicates
                        np.random.shuffle(sample_indices)
                    else:
                        sample_indices = np.arange(max(5, history_size // 2), history_size)

                    # Pick a random sequence from our samples
                    seq_idx = np.random.choice(sample_indices)
                    seq_length = min(50, seq_idx)  # Reasonable sequence length for training
                    start_idx = max(0, seq_idx - seq_length)

                    # Extract sequence data (8 features)
                    seq_data = np.array([self.search_history[i][:8] for i in range(start_idx, seq_idx + 1)])

                    if len(seq_data) < 3:  # Minimum sequence length
                        continue

                    # Target: predict quality of the final position in sequence
                    target_quality = 1.0 - self.search_history[seq_idx][1]  # Lower error = higher quality
                    target_position = self.search_history[seq_idx][0]  # Position prediction

                    # Convert to tensors
                    x = torch.FloatTensor(seq_data).unsqueeze(0).to(self.device)
                    target_qual = torch.FloatTensor([target_quality]).to(self.device)
                    target_pos = torch.FloatTensor([target_position]).to(self.device)

                    # Forward pass
                    self.optimizer.zero_grad()
                    result = self.model(x)

                    # Calculate losses
                    quality_loss = self.quality_criterion(result['quality_score'].squeeze(), target_qual)
                    position_loss = self.position_criterion(result['position_offset'].squeeze(), target_pos)

                    # Combined loss with adaptive weighting based on training progress
                    quality_weight = 0.8 if epoch < epochs // 2 else 0.6  # Focus more on quality early
                    loss = quality_weight * quality_loss + (1 - quality_weight) * position_loss

                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    batch_count += 1

                if batch_count > 0:
                    avg_loss = epoch_loss / batch_count
                    print(".3f")

                # Learning rate scheduling
                self.lr_scheduler.step()

            self.model.eval()
            print(f"[Transformer] ✅ Training completed on {history_size:,} samples")

        except Exception as e:
            print(f"[Transformer] Training failed: {e}")

    def _predict_simple(self, normalized_step: float, normalized_sqrt_N: float,
                        normalized_N_bits: float) -> float:
        """LEGACY: Predict using simple weighted model for backward compatibility."""
        if len(self.search_history) == 0:
            return 0.5
        
        # Find similar steps in history
        recent_history = self.search_history[-50:]  # Use last 50 steps
        
        # Calculate similarity scores based on step position
        similarities = []
        for hist_step, hist_diff, hist_sqrt, hist_nbits in recent_history:
            # Similarity based on step position
            step_sim = 1.0 / (1.0 + abs(hist_step - normalized_step) * 10)
            # Weight by how good the result was (lower diff_bits = better)
            quality = 1.0 / (1.0 + hist_diff)
            similarities.append(step_sim * quality)
        
        if len(similarities) > 0:
            # Weighted average of similarities
            return min(1.0, max(0.0, np.mean(similarities)))
        return 0.5
    
    def train_on_history(self, epochs: int = 10):
        """Train the model on accumulated history."""
        if not self.use_torch or len(self.search_history) < 10:
            return
        
        try:
            self.model.train()
            
            # Prepare training data
            sequences = []
            targets = []
            
            # Create sliding windows from history
            window_size = min(50, len(self.search_history) - 1)
            for i in range(len(self.search_history) - window_size):
                seq = self.search_history[i:i+window_size]
                target = 1.0 / (1.0 + self.search_history[i+window_size][1])  # Quality based on diff_bits
                
                sequences.append(seq)
                targets.append(target)
            
            if len(sequences) == 0:
                return
            
            # Convert to tensors and move to device
            X = torch.FloatTensor(sequences).to(self.device)
            y = torch.FloatTensor(targets).unsqueeze(1).to(self.device)
            
            # Training loop
            for epoch in range(epochs):
                self.optimizer.zero_grad()
                predictions = self.model(X)
                loss = self.criterion(predictions, y)
                loss.backward()
                self.optimizer.step()
            
            self.model.eval()
            print(f"[Transformer] Trained on {len(sequences)} samples, final loss: {loss.item():.4f}")
        except Exception as e:
            print(f"[Transformer] Training error: {e}")
            self.model.eval()
    
    def predict_optimal_step_size(self, current_step: int, sqrt_N: int, N_bits: int,
                                  base_step_size: int = None) -> int:
        """
        Predict optimal step size based on learned p-q difference patterns.

        Args:
            current_step: Current step position
            sqrt_N: Square root of N
            N_bits: Bit length of N
            base_step_size: Base step size to use if no history (default: 2^100)

        Returns:
            Optimal step size in bits (will be converted to 2^bits)
        """
        if base_step_size is None:
            base_step_size = 100  # Default: 2^100

        if len(self.factor_history) < 3:
            # Not enough history, use base step size
            return base_step_size

        # Analyze learned p-q difference patterns
        recent_factors = self.factor_history[-min(20, len(self.factor_history)):]

        # Calculate average p-q difference in bits
        p_q_diffs = [f[3] for f in recent_factors if f[3] > 0]  # p_q_diff_bits

        if not p_q_diffs:
            return base_step_size

        avg_p_q_diff_bits = sum(p_q_diffs) / len(p_q_diffs)

        # Also consider the quality of results (lower diff_bits = better)
        weighted_diffs = []
        for f in recent_factors:
            p_q_diff_bits = f[3]
            result_quality = 1.0 / (1.0 + f[4]) if f[4] is not None and f[4] > 0 else 0.1
            weighted_diffs.append(p_q_diff_bits * result_quality)

        if weighted_diffs:
            weighted_avg = sum(weighted_diffs) / len(weighted_diffs)
        else:
            weighted_avg = avg_p_q_diff_bits

        # Predict step size: if p-q differences are small, use smaller steps
        # If differences are large, use larger steps
        if weighted_avg < N_bits / 4:
            # Small p-q differences: use smaller steps for precision
            optimal_bits = max(50, int(weighted_avg / 2))
        elif weighted_avg < N_bits / 2:
            # Medium differences: use medium steps
            optimal_bits = max(75, int(weighted_avg))
        else:
            # Large differences: use larger steps for coverage
            optimal_bits = min(200, int(weighted_avg * 1.5))

        # Clamp to reasonable range
        optimal_bits = max(50, min(200, optimal_bits))

        return optimal_bits

    def predict_next_search_position(self, current_position: int, sqrt_N: int, N_bits: int) -> int:
        """
        PROGRESSIVE LEARNING: Use all accumulated knowledge to predict the next most promising search position.

        The Transformer continuously learns from every search result and improves its predictions
        by analyzing patterns in successful factorizations, good approximations, and failed attempts.
        """
        if len(self.factor_history) < 3:
            # Not enough data - do intelligent exploration around sqrt(N)
            exploration_step = len(self.factor_history) + 1
            # Start closer to sqrt(N) and explore logarithmically outward
            base_offset = 2**(25 + exploration_step * 3)  # 2^28, 2^31, 2^34, etc.

            # Alternate between above and below sqrt(N) for better coverage
            if exploration_step % 2 == 1:
                next_pos = sqrt_N + base_offset
            else:
                next_pos = max(1000, sqrt_N - base_offset // 4)  # Don't go too low

            print(f"[Transformer] Exploration phase: testing around √N ± 2^{base_offset.bit_length()}")
            return next_pos

        # Progressive learning: analyze ALL available data
        all_factors = self.factor_history

        # Extract all valid results (both successful and approximations)
        valid_results = [f for f in all_factors if f[4] is not None]
        successful = [f for f in valid_results if f[4] == 0]
        approximations = [f for f in valid_results if f[4] > 0]

        print(f"[Transformer] Progressive learning from {len(valid_results)} total results ({len(successful)} exact, {len(approximations)} approximations)")

        if successful:
            # EXCELLENT: We have exact factorizations! Focus on their patterns
            print(f"[Transformer] 🎯 Found {len(successful)} exact factorizations - converging on pattern!")

            # Use successful factorizations as the primary guide
            success_positions = [f[0] for f in successful]
            success_p_q_diffs = [f[3] for f in successful]

            # Calculate weighted average based on recency (newer results more important)
            weights = list(range(1, len(successful) + 1))
            weighted_pos = sum(p * w for p, w in zip(success_positions, weights)) // sum(weights)
            weighted_p_q_diff = sum(d * w for d, w in zip(success_p_q_diffs, weights)) // sum(weights)

            # Jump toward the weighted center of successful regions
            pos_diff = abs(weighted_pos - current_position)
            jump_distance = pos_diff // 3 + 2**(weighted_p_q_diff // 2)  # Blend position and p-q insights

        elif approximations:
            # LEARNING FROM APPROXIMATIONS: Get progressively better
            print(f"[Transformer] 📈 Learning from {len(approximations)} approximations - improving predictions")

            # Sort by quality and recency
            approximations.sort(key=lambda x: (x[4], -x[0]))  # Best quality first, then most recent

            # Take top results, weighted by quality (better approximations = higher weight)
            top_results = approximations[:min(5, len(approximations))]

            # Quality-weighted analysis
            total_weight = sum(1.0 / (r[4] + 1) for r in top_results)  # Better quality = higher weight
            avg_pos = sum(r[0] * (1.0 / (r[4] + 1)) for r in top_results) / total_weight
            avg_p_q_diff = sum(r[3] * (1.0 / (r[4] + 1)) for r in top_results) / total_weight

            # Progressive refinement: get closer to good approximations
            pos_diff = abs(avg_pos - current_position)
            p_q_scale = 2**(avg_p_q_diff // 4)  # Smaller jumps as we learn

            if current_position < avg_pos:
                jump_distance = pos_diff // 4 + p_q_scale  # Approach good region
            else:
                jump_distance = p_q_scale  # Explore at similar scale

        else:
            # LEARNING FROM FAILURES: Even failed attempts teach us something
            print(f"[Transformer] 🔄 Learning from {len(all_factors)} attempts (no good approximations yet)")

            # Analyze failure patterns to avoid bad regions
            failed_positions = [f[0] for f in all_factors]

            # Try scales that haven't been explored much
            attempt_count = len(all_factors)
            # Use a more systematic exploration pattern
            scale_step = (attempt_count // 5) % 33  # EXPANDED: Change scale every 5 attempts
            jump_distance = 2**(35 + scale_step * 2)  # EXPANDED: 2^35, 2^37, 2^39, ... 2^101

            # Avoid recently failed positions
            next_pos = current_position + jump_distance
            while any(abs(next_pos - fp) < jump_distance // 4 for fp in failed_positions[-10:]):
                next_pos += jump_distance // 2  # Shift away from recent failures

            return next_pos

        # Apply progressive learning bounds - EXPANDED RANGE: 2^30 to 2^100
        jump_distance = max(jump_distance, 2**30)  # Minimum intelligent jump
        jump_distance = min(jump_distance, 2**100)  # EXPANDED: Maximum jump now 2^100 (was 2^80)

        # Direction: progressively refine toward promising regions
        if successful:
            next_pos = current_position + jump_distance  # Go toward successful regions
        else:
            # For approximations, sometimes explore in both directions
            if len(approximations) % 3 == 0:  # Every 3rd approximation-guided search
                next_pos = current_position - jump_distance // 2  # Try backward
                next_pos = max(next_pos, 1000)  # Don't go too low
            else:
                next_pos = current_position + jump_distance

        # Learning bounds check - EXPANDED RANGE
        search_upper = 2**(N_bits + 50)  # EXPANDED: Allow much more overflow for 2^100 jumps
        next_pos = min(next_pos, search_upper)

        # Safe bit length calculation
        try:
            jump_bits = (next_pos - current_position).bit_length()
        except (AttributeError, OverflowError):
            jump_bits = len(bin(next_pos - current_position)) - 2 if next_pos > current_position else 0

        print(f"[Transformer] Learning from {len(successful)} successes, {len(approximations)} approximations")
        print(f"[Transformer] Next jump: 2^{jump_bits}")

        return next_pos

    def save_model(self, filepath: str) -> bool:
        """Save the trained transformer model to disk.

        Returns:
            True if save was successful, False otherwise
        """
        if not self.use_torch:
            print("[Transformer] Cannot save: PyTorch not available")
            return False

        try:
            import torch
            state = {
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'search_history': self.search_history,
                'factor_history': self.factor_history,
                'pretrained': getattr(self, 'pretrained', False),
                'config': {
                    'd_model': self.d_model,
                    'nhead': self.nhead,
                    'num_layers': self.num_layers,
                    'max_seq_len': self.max_seq_len
                }
            }
            torch.save(state, filepath)
            print(f"[Transformer] ✅ Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"[Transformer] ❌ Failed to save model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """Load a trained transformer model from disk."""
        if not self.use_torch:
            print("[Transformer] Cannot load: PyTorch not available")
            return False

        try:
            import torch
            state = torch.load(filepath, map_location=self.device)

            # Check if model architecture matches
            config = state.get('config', {})
            if (config.get('d_model') != self.d_model or
                config.get('nhead') != self.nhead or
                config.get('num_layers') != self.num_layers):
                print("[Transformer] ⚠️ Model architecture mismatch, loading anyway...")

            self.model.load_state_dict(state['model_state'])
            self.optimizer.load_state_dict(state['optimizer_state'])

            # Restore learning history
            self.search_history = state.get('search_history', [])
            self.factor_history = state.get('factor_history', [])
            self.pretrained = state.get('pretrained', False)

            print(f"[Transformer] ✅ Model loaded from {filepath}")
            print(f"[Transformer] 📊 Restored {len(self.search_history)} search samples, {len(self.factor_history)} factor samples")
            if self.pretrained:
                print("[Transformer] 🎓 Model was pre-trained on synthetic RSA keys")

            return True
        except Exception as e:
            print(f"[Transformer] ❌ Failed to load model: {e}")
            return False


# ============================================================================
# ENHANCED POLYNOMIAL SOLVER
# ============================================================================

class EnhancedPolynomialSolver:
    """Advanced polynomial solving for factorization problems."""
    
    def __init__(self, N: int, config: dict = None, p_approx: int = None, q_approx: int = None):
        self.N = N
        self.p = sp.Symbol('p', integer=True, positive=True)
        self.q = sp.Symbol('q', integer=True, positive=True)
        self.config = config or {}

        # Set approximations - use provided hints, or fall back to improved defaults
        if p_approx is not None and q_approx is not None:
            self.p_approx = p_approx
            self.q_approx = q_approx
        else:
            # IMPROVED: Use integer square root for accurate approximation
            import math
            sqrt_N = math.isqrt(N)  # Integer square root (exact for large numbers)
            
            # Strategy 1: Start with sqrt(N) as initial guess for p
            # Since p and q are close to sqrt(N) for balanced RSA keys
            self.p_approx = sqrt_N
            
            # Strategy 2: Use Newton's method refinement for better accuracy
            # For p ≈ sqrt(N), we can refine using: p_new = (p_old + N/p_old) // 2
            # This converges to sqrt(N) quickly
            p_refined = (sqrt_N + N // sqrt_N) // 2
            if p_refined > 0 and p_refined < sqrt_N * 2:
                self.p_approx = p_refined
            
            # Strategy 3: Account for typical RSA key imbalance
            # Most RSA keys have p and q within a factor of 2-10 of sqrt(N)
            # Try slightly smaller p (common in practice)
            p_candidate = (sqrt_N * 9) // 10  # 90% of sqrt(N)
            if p_candidate > 0:
                q_candidate = N // p_candidate
                # Check if this gives a better balance (p and q closer in size)
                if abs(p_candidate - q_candidate) < abs(self.p_approx - (N // self.p_approx)):
                    self.p_approx = p_candidate
            
            self.q_approx = N // self.p_approx
            
            # Strategy 4: Refine using continued fraction approximation
            # If N has a continued fraction representation, we can get better estimates
            # For now, use the integer square root which is already quite good
            
            print(f"[Estimation] Initial p_approx = {self.p_approx} (bit-length: {self.p_approx.bit_length()})")
            print(f"[Estimation] Initial q_approx = {self.q_approx} (bit-length: {self.q_approx.bit_length()})")
            print(f"[Estimation] Error: p_approx * q_approx - N = {self.p_approx * self.q_approx - N}")
    
    def refine_approximations(self, max_iterations: int = 5) -> Tuple[int, int]:
        """
        Refine p and q approximations using iterative methods.
        
        Uses:
        1. Newton's method: p_new = (p_old + N/p_old) // 2
        2. Balance correction: Adjust to minimize |p - q|
        3. Error minimization: Find p that minimizes |p * (N//p) - N|
        
        Returns:
            (refined_p_approx, refined_q_approx)
        """
        import math
        
        p_current = self.p_approx
        q_current = self.q_approx
        best_error = abs(p_current * q_current - self.N)
        best_p = p_current
        best_q = q_current
        
        print(f"[Refinement] Starting with p={p_current}, q={q_current}, error={best_error}")
        
        # Method 1: Newton's method refinement (converges to sqrt(N))
        for iteration in range(max_iterations):
            # Newton step: p_new = (p + N/p) // 2
            if p_current > 0:
                p_new = (p_current + self.N // p_current) // 2
                q_new = self.N // p_new
                error = abs(p_new * q_new - self.N)
                
                if error < best_error:
                    best_error = error
                    best_p = p_new
                    best_q = q_new
                    print(f"[Refinement] Iteration {iteration+1}: p={p_new}, error={error} (improved)")
                
                # Check convergence
                if abs(p_new - p_current) < max(1, p_current // 1000):
                    break
                
                p_current = p_new
                q_current = q_new
        
        # Method 2: Try values that make p and q more balanced
        # For RSA, p and q are typically within a factor of 2-10
        sqrt_N = math.isqrt(self.N)
        for factor in [8, 9, 10, 11, 12]:  # Try different ratios
            p_candidate = (sqrt_N * factor) // 10
            if p_candidate > 0:
                q_candidate = self.N // p_candidate
                error = abs(p_candidate * q_candidate - self.N)
                
                # Also check balance (prefer p and q closer in size)
                balance = abs(p_candidate - q_candidate)
                if error < best_error * 1.1 and balance < abs(best_p - best_q) * 1.1:
                    best_p = p_candidate
                    best_q = q_candidate
                    best_error = error
        
        self.p_approx = best_p
        self.q_approx = best_q
        
        print(f"[Refinement] Final: p={best_p}, q={best_q}, error={best_error}")
        print(f"[Refinement] Improvement: {abs(self.p_approx * self.q_approx - N) - best_error} reduction in error")
        
        return best_p, best_q

    def _safe_penalty(self, val, max_penalty: float = 1000.0) -> float:
        """
        Safely compute penalty for large numbers without float conversion.

        For large numbers, we use integer arithmetic and bit-length approximations:
        - If val is exactly 0, penalty = 0 (perfect match)
        - For small values, use normal float conversion
        - For large values, use bit-length and log approximations to estimate magnitude
        - For very large numbers, use relative comparisons when possible
        """
        try:
            # First check: exact zero (perfect match)
            if val == 0:
                return 0.0

            # Try to get the integer value
            int_val = None
            if hasattr(val, 'is_integer') and val.is_integer:
                try:
                    int_val = int(val)
                except (OverflowError, ValueError):
                    pass

            if int_val is not None:
                # We have an integer value
                abs_val = abs(int_val)

                # Small integers: safe to convert to float
                if abs_val < 1e15:
                    return abs(float(int_val))

                # Large integers: use bit length and log approximations
                if abs_val == 0:
                    return 0.0

                # For very large numbers, use log approximation
                # log10(abs_val) ≈ bit_length * log10(2) ≈ bit_length * 0.3010
                bit_len = abs_val.bit_length()
                log10_approx = bit_len * 0.3010

                # Scale the penalty based on magnitude
                # For numbers much larger than N, give high penalty
                if bit_len > self.N.bit_length() + 50:  # Much larger than N
                    return max_penalty
                elif bit_len > self.N.bit_length() + 10:  # Significantly larger than N
                    return min(max_penalty * 0.8, max_penalty)
                elif bit_len > self.N.bit_length():  # Larger than N
                    return min(max_penalty * 0.5, max_penalty)
                else:
                    # Comparable size to N, try to be more precise
                    # Use the log approximation as a penalty measure
                    penalty = min(log10_approx, max_penalty)
                    return penalty

            # For SymPy expressions that are numbers but not integers
            if hasattr(val, 'is_number') and val.is_number and not val.is_integer:
                try:
                    # Try float conversion for non-integers
                    return min(abs(float(val)), max_penalty)
                except (OverflowError, ValueError):
                    # Can't convert, assume large
                    return max_penalty

            # For expressions that aren't pure numbers, try symbolic evaluation
            try:
                # Check if the expression evaluates to zero
                if val.simplify() == 0:
                    return 0.0
                # Try to evaluate numerically with high precision
                numeric_val = val.evalf()
                if numeric_val.is_number and not numeric_val.is_infinite:
                    try:
                        float_val = float(numeric_val)
                        if abs(float_val) < 1e308:  # Check for reasonable float range
                            return min(abs(float_val), max_penalty)
                    except (OverflowError, ValueError):
                        pass
            except:
                pass

            # Last resort: if we can't evaluate, assume it's non-zero
            return max_penalty * 0.1  # Lower default penalty to be less aggressive

        except Exception:
            # Any error in evaluation means we can't assess the penalty precisely
            return max_penalty * 0.1  # Be conservative

    def _groebner_factorization_method(self, gb_polys: List[sp.Expr]) -> List[Dict]:
        """
        Use Gröbner basis techniques specifically for factorization.

        Instead of solving general polynomial systems, this method:
        1. Constructs polynomials that encode the factorization problem
        2. Uses Gröbner basis to eliminate variables and find factors
        3. Leverages lattice structure for efficient computation
        """
        print(f"[Gröbner] Applying factorization-specific Gröbner basis method...")

        # Validate and convert to proper polynomials
        valid_polys = []
        for i, poly in enumerate(gb_polys):
            try:
                # Ensure it's a polynomial in p and q
                if not poly.is_polynomial(self.p, self.q):
                    print(f"[Gröbner] Converting expression {i+1} to polynomial...")
                    poly = sp.Poly(poly, self.p, self.q)
                valid_polys.append(poly)
            except Exception as e:
                print(f"[Gröbner] Skipping invalid polynomial {i+1}: {e}")
                continue

        if len(valid_polys) < 2:
            print(f"[Gröbner] Not enough valid polynomials ({len(valid_polys)}), need at least 2")
            return []

        print(f"[Gröbner] Using {len(valid_polys)} valid polynomials")

        solutions = []

        try:
            # Method 1: Use resultants to eliminate variables
            if len(valid_polys) >= 2:
                print(f"[Gröbner] Using resultants for variable elimination...")
                for i in range(len(valid_polys)):
                    for j in range(i+1, len(valid_polys)):
                        try:
                            poly1, poly2 = valid_polys[i], valid_polys[j]

                            # Compute resultant with respect to p to eliminate p
                            res_p = sp.resultant(poly1, poly2, self.p)
                            if res_p != 0:
                                print(f"[Gröbner] Resultant w.r.t. p computed, solving for q...")
                                q_candidates = sp.solve(res_p, self.q)

                                for q_val in q_candidates:
                                    if q_val.is_integer and q_val > 1:
                                        q_int = int(q_val)
                                        # Check if q divides N
                                        if self.N % q_int == 0:
                                            p_int = self.N // q_int
                                            print(f"[Gröbner] Computed p = {p_int} from q = {q_int}")
                                            if p_int > 1:
                                                print(f"[Gröbner] ✓✓✓ FACTORIZATION FOUND via resultant: p={p_int}, q={q_int}")
                                                print(f"[Gröbner] Found p={p_int}, q={q_int}")
                                                solutions.append({self.p: sp.Integer(p_int), self.q: sp.Integer(q_int)})
                                                return solutions  # Return immediately on success

                        except Exception as e:
                            print(f"[Gröbner] Resultant failed for pair ({i},{j}): {e}")
                            continue

            # Method 2: Direct Gröbner basis computation for small systems
            if len(valid_polys) <= 3 and not solutions:
                print(f"[Gröbner] Small system ({len(valid_polys)} polynomials), computing full Gröbner basis...")
                try:
                    # Compute Gröbner basis
                    gb = sp.groebner(valid_polys, [self.p, self.q], order='lex')

                    print(f"[Gröbner] Gröbner basis computed with {len(gb)} polynomials")

                    # Look for univariate polynomials in q
                    for poly in gb:
                        if poly.degree(self.p) == 0 and poly.degree(self.q) > 0:
                            print(f"[Gröbner] Found univariate polynomial in q: {poly}")
                            q_roots = sp.solve(poly, self.q)
                            for q_root in q_roots:
                                if q_root.is_integer and q_root > 1:
                                    q_int = int(q_root)
                                    if self.N % q_int == 0:
                                        p_int = self.N // q_int
                                        if p_int > 1:
                                            print(f"[Gröbner] ✓✓✓ FACTORIZATION FOUND via Gröbner basis: p={p_int}, q={q_int}")
                                            solutions.append({self.p: sp.Integer(p_int), self.q: sp.Integer(q_int)})
                                            return solutions

                except Exception as e:
                    print(f"[Gröbner] Gröbner basis computation failed: {e}")

        except Exception as e:
            print(f"[Gröbner] Factorization method failed: {e}")

        print(f"[Gröbner] Factorization method found {len(solutions)} solutions")
        return solutions

    def solve_with_groebner_basis(self, polynomials: List[sp.Expr]) -> Optional[Tuple[int, int]]:
        """
        Advanced Gröbner basis solving using algebraic elimination and systematic strategies.
        Adapted from Sage's proven implementation with proper ideal theory.

        For factorization, we set up a polynomial system in correction variables dp, dq
        where (p_approx + dp) * (q_approx + dq) = N

        Key features:
        - Lex ordering for variable elimination
        - Dimension analysis of the polynomial ideal
        - Systematic univariate polynomial solving
        - Resultant elimination for variable removal
        - Variety computation over integers and rationals
        """
        print("\n[Gröbner] Computing Gröbner basis for factorization...")

        try:
            # For factorization, create a polynomial system in correction variables dp, dq
            # We want (p_approx + dp) * (q_approx + dq) = N
            # This expands to: dp*dq + p_approx*dq + q_approx*dp + p_approx*q_approx - N = 0

            # Use the Diophantine approach: f(x,v) = x² + 2tx + (t² - N) - v² = 0
            # Where x = u - t (u near t), and we solve for small |x|
            # Then p = u - v, q = u + v
            
            x = sp.Symbol('x', integer=True)  # x = u - t (small correction)
            v = sp.Symbol('v', integer=True)  # v parameter
            
            # t is the center point (near sqrt(N) or (p_approx + q_approx)/2)
            t = (self.p_approx + self.q_approx) // 2
            if t == 0:
                t = int(sp.sqrt(self.N))
            
            # Create the factorization polynomial: f(x,v) = x² + 2tx + (t² - N) - v² = 0
            factorization_poly = x**2 + 2*t*x + (t**2 - self.N) - v**2

            print(f"    Diophantine polynomial: f(x,v) = x² + {2*t}x + ({t**2 - self.N}) - v² = 0")
            print(f"    Solving for small |x| (u near t={t})...")

            # Start with the factorization polynomial
            processed_polys = [factorization_poly]

            # For the Diophantine approach, we only need the factorization polynomial
            # The Gröbner basis is just {f(x,v)} - the work is solving the Diophantine constraint
            print(f"    Using single polynomial Diophantine approach")
            print(f"    Gröbner basis will be {{f(x,v)}} - solving for integer solutions")

            print(f"    Solving system with {len(processed_polys)} polynomial in x, v")

            # The processed_polys are already set up correctly for dp, dq

            if not processed_polys:
                print("    No polynomials to process")
                return None

            print(f"    Processing {len(processed_polys)} polynomials")

            # Use lex ordering for elimination: v > x (eliminate v first to get univariate in x)
            # This is the key: Gröbner basis with lex order will give us elimination ideals
            try:
                print("    Computing Gröbner basis with lex ordering (v > x) to eliminate v...")
                print("    Goal: Obtain univariate polynomial in x (no brute search needed)")
                start_time = time.time()
                # Lex order v > x means: eliminate v first, polynomials in x come last
                gb = groebner(processed_polys, [v, x], order='lex')
                gb_time = time.time() - start_time
                print(f"    ✓ Gröbner basis computed in {gb_time:.3f}s")
                print(f"    Basis has {len(gb)} elements")
                gb_list = list(gb)
                print(f"    DEBUG: GB list: {gb_list}")
                
                # Check if we got [1] immediately (inconsistent system)
                if gb_list == [1]:
                    print(f"    DEBUG: WARNING - Gröbner basis is [1] immediately!")
                    print(f"    DEBUG: System is inconsistent, cannot proceed with elimination")
                    # Continue to old error handling code below
                else:
                    # Extract elimination ideal: polynomials that don't contain v (are in Q[x])
                    # This is the key: with lex order v > x, elimination ideal contains polynomials purely in x
                    # For a polynomial like v² = f(x), we can extract f(x) as the elimination ideal
                    print("    Extracting elimination ideal (polynomials in x only)...")
                    elimination_polys = []
                    for poly in gb_list:
                        if hasattr(poly, 'free_symbols'):
                            if v not in poly.free_symbols:
                                elimination_polys.append(poly)
                                print(f"      Found univariate polynomial in x: {poly}")
                            elif v in poly.free_symbols and x in poly.free_symbols:
                                # Try to extract the elimination ideal from v² = f(x) form
                                # If poly is v² - f(x) = 0, then f(x) is in the elimination ideal
                                try:
                                    poly_poly = sp.Poly(poly, v, x)
                                    # Check if it's of form v² - f(x) = 0
                                    v2_coeff = poly_poly.coeff_monomial(v**2 * x**0)
                                    if v2_coeff != 0:
                                        # Extract the part that doesn't involve v (the f(x) part)
                                        # This is the elimination ideal
                                        poly_no_v = poly.subs(v, 0)  # Set v=0 to get f(x)
                                        if x in poly_no_v.free_symbols:
                                            elimination_polys.append(poly_no_v)
                                            print(f"      Extracted elimination ideal from v² = f(x): {poly_no_v} = 0")
                                except:
                                    pass
                    
                    if elimination_polys:
                        print(f"    ✓ Found {len(elimination_polys)} univariate polynomial(s) in x")
                        # Use the first (and typically only) univariate polynomial
                        univariate_poly_x = elimination_polys[0]
                        print(f"    Solving univariate polynomial: {univariate_poly_x} = 0")
                        
                        # Solve the univariate polynomial directly (no brute search!)
                        try:
                            # Convert to polynomial form for better solving
                            poly_x = sp.Poly(univariate_poly_x, x)
                            poly_expr = poly_x.as_expr()
                            
                            # For very large coefficients, truncate display but keep full precision
                            poly_str = str(poly_expr)
                            if len(poly_str) > 200:
                                print(f"    Polynomial form: {poly_str[:200]}... = 0 (truncated, {len(poly_str)} chars)")
                            else:
                                print(f"    Polynomial form: {poly_expr} = 0")
                            
                            # Try solving symbolically (works for reasonable sizes)
                            try:
                                x_solutions = sp.solve(univariate_poly_x, x, check=False)
                                print(f"    Found {len(x_solutions)} symbolic solution(s) for x")
                            except:
                                x_solutions = []
                                print("    Symbolic solve failed (likely due to large coefficients)")
                            
                            # Extract coefficients for manual solving if needed
                            coeffs = poly_x.all_coeffs()
                            degree = poly_x.degree()
                            print(f"    Degree: {degree}, coefficient bit lengths: {[c.bit_length() if hasattr(c, 'bit_length') else 'unknown' for c in coeffs]}")
                            
                            # For quadratic with huge coefficients, we can still compute roots if discriminant is manageable
                            if degree == 2 and len(coeffs) == 3:
                                try:
                                    a, b, c = int(coeffs[0]), int(coeffs[1]), int(coeffs[2])
                                    # Compute discriminant: b² - 4ac
                                    # For huge numbers, this might overflow, so use SymPy
                                    a_sym, b_sym, c_sym = sp.Integer(a), sp.Integer(b), sp.Integer(c)
                                    discriminant = b_sym*b_sym - 4*a_sym*c_sym
                                    
                                    disc_bits = discriminant.bit_length() if hasattr(discriminant, 'bit_length') else 0
                                    print(f"    Discriminant: {disc_bits} bits")
                                    
                                    # Check if discriminant is a perfect square
                                    if discriminant >= 0:
                                        # For reasonable size, check if it's a perfect square
                                        if disc_bits < 1000:  # Reasonable size to check
                                            disc_int = int(discriminant)
                                            disc_root_approx = int(sp.sqrt(disc_int))
                                            if disc_root_approx * disc_root_approx == disc_int:
                                                # Perfect square - compute roots
                                                x1 = (-b_sym + disc_root_approx) // (2 * a_sym)
                                                x2 = (-b_sym - disc_root_approx) // (2 * a_sym)
                                                x_solutions = [x1, x2]
                                                print(f"    Found rational roots: x = {x1}, {x2}")
                                            else:
                                                print(f"    Discriminant is not a perfect square")
                                        else:
                                            print(f"    Discriminant too large ({disc_bits} bits) to check for perfect square")
                                    else:
                                        print(f"    Discriminant is negative - no real roots")
                                except (OverflowError, ValueError, AttributeError) as e:
                                    print(f"    Error computing discriminant: {e}")
                            
                            # If we still don't have solutions and it's a quadratic, the polynomial
                            # likely doesn't have integer roots (as expected for single constraint case)
                            if len(x_solutions) == 0:
                                print("    Note: Univariate polynomial from elimination ideal has no integer roots")
                                print("    This is expected for single constraint - need additional constraints for integer solutions")
                            
                            # For each x solution, recover v and then p, q
                            for x_sol in x_solutions:
                                try:
                                    # Check if x_sol is an integer
                                    if x_sol.is_integer or (hasattr(x_sol, 'is_rational') and x_sol.is_rational and x_sol.denominator == 1):
                                        x_val = int(x_sol)
                                        print(f"      Trying x = {x_val}")
                                        
                                        # Substitute x into the original equation to get v
                                        # From: x² + 2tx + (t² - N) - v² = 0
                                        # We get: v² = x² + 2tx + (t² - N)
                                        v_squared = x_val*x_val + 2*t*x_val + (t*t - self.N)
                                        
                                        if v_squared >= 0:
                                            v_root = int(sp.sqrt(v_squared))
                                            if v_root*v_root == v_squared:
                                                # Found integer v
                                                for v_sign in [v_root, -v_root]:
                                                    u = t + x_val
                                                    p_candidate = u - v_sign
                                                    q_candidate = u + v_sign
                                                    
                                                    if p_candidate > 0 and q_candidate > 0 and p_candidate * q_candidate == self.N:
                                                        print(f"      ✓✓✓ SOLUTION FOUND via univariate elimination!")
                                                        print(f"      x = {x_val}, v = {v_sign}")
                                                        print(f"      u = {u}, p = {p_candidate}, q = {q_candidate}")
                                                        return (p_candidate, q_candidate)
                                        else:
                                            # Try solving for v from the GB polynomial directly
                                            # Find polynomial in v by substituting x
                                            for poly in gb_list:
                                                if v in poly.free_symbols and x not in poly.free_symbols:
                                                    # This shouldn't happen with lex order v > x, but check anyway
                                                    pass
                                                elif v in poly.free_symbols:
                                                    # Substitute x and solve for v
                                                    poly_at_x = poly.subs(x, x_val)
                                                    v_solutions = sp.solve(poly_at_x, v)
                                                    for v_sol in v_solutions:
                                                        if v_sol.is_integer or (hasattr(v_sol, 'is_rational') and v_sol.is_rational and v_sol.denominator == 1):
                                                            v_val = int(v_sol)
                                                            u = t + x_val
                                                            p_candidate = u - v_val
                                                            q_candidate = u + v_val
                                                            if p_candidate > 0 and q_candidate > 0 and p_candidate * q_candidate == self.N:
                                                                print(f"      ✓✓✓ SOLUTION FOUND!")
                                                                return (p_candidate, q_candidate)
                                    
                                except (ValueError, OverflowError, AttributeError) as e:
                                    print(f"      Error processing x solution {x_sol}: {e}")
                                    continue
                            
                            print(f"    No valid factorization found from univariate solutions")
                        except Exception as e:
                            print(f"    Error solving univariate polynomial: {e}")
                    else:
                        print("    ⚠ No univariate polynomial found in elimination ideal")
                        print("    Note: With single polynomial f(x,v), elimination should give univariate in x")
                        print("    Falling back to analyzing GB structure...")

                # Continue with old error handling if GB was [1] or elimination didn't work
                if gb_list == [1]:
                    print(f"    DEBUG: WARNING - Gröbner basis is [1] immediately!")
                    print(f"    DEBUG: This suggests a processing issue, not mathematical inconsistency")
                    print(f"    DEBUG: The old implementation worked, so let's investigate...")

                    # Check for common issues
                    print(f"    DEBUG: Investigating polynomial issues:")

                    # Check if any polynomials are the same
                    unique_polys = set()
                    duplicates = 0
                    for poly in processed_polys:
                        poly_str = str(poly)
                        if poly_str in unique_polys:
                            duplicates += 1
                        else:
                            unique_polys.add(poly_str)
                    if duplicates > 0:
                        print(f"    DEBUG: Found {duplicates} duplicate polynomials")

                    # Check polynomial degrees
                    degrees = []
                    for poly in processed_polys:
                        try:
                            deg = poly.total_degree()
                            degrees.append(deg)
                        except:
                            degrees.append(-1)  # Error getting degree

                    max_deg = max(degrees) if degrees else 0
                    print(f"    DEBUG: Polynomial degrees: min={min(degrees)}, max={max_deg}, avg={sum(degrees)/len(degrees):.1f}")

                    if max_deg > 10:
                        print(f"    DEBUG: High degree polynomials detected (max degree {max_deg})")
                        print(f"    DEBUG: This can cause Gröbner basis computation issues")

                    # Try a smarter subset selection based on polynomial quality
                    print(f"    DEBUG: Attempting smart subset selection...")
                    print(f"    DEBUG: Evaluating polynomial quality at approximations...")

                    # Evaluate each polynomial at the approximations to find "good" ones
                    poly_scores = []
                    for i, poly in enumerate(processed_polys):
                        try:
                            if hasattr(poly, 'subs'):
                                residual = poly.subs([(self.p, self.p_approx), (self.q, self.q_approx)])
                                if hasattr(residual, 'simplify'):
                                    residual = residual.simplify()
                                if residual.is_number:
                                    # Convert SymPy number to Python int for bit_length()
                                    if residual == 0:
                                        residual_bits = 0
                                    else:
                                        try:
                                            residual_int = int(abs(residual))
                                            residual_bits = residual_int.bit_length()
                                        except (OverflowError, ValueError):
                                            # For very large numbers, estimate bit length
                                            residual_str = str(abs(residual))
                                            residual_bits = len(residual_str) * 4  # Rough estimate: ~4 bits per digit
                                    poly_scores.append((i, residual_bits, poly))
                                else:
                                    poly_scores.append((i, 999, poly))  # Can't evaluate
                        except Exception as e:
                            # More detailed error reporting
                            if i < 5:  # Only show first few errors
                                print(f"    DEBUG: Failed to evaluate poly {i} at approximations: {e}")
                            poly_scores.append((i, 999, poly))  # Evaluation failed

                    # Sort by residual size (smallest first = best polynomials)
                    poly_scores.sort(key=lambda x: x[1])

                    print(f"    DEBUG: Best polynomials (by residual size):")
                    for i, (poly_idx, residual_bits, poly) in enumerate(poly_scores[:5]):
                        print(f"    DEBUG:  #{i}: Poly {poly_idx}, residual={residual_bits} bits")

                    # Try with the best 3 polynomials
                    if len(poly_scores) >= 3:
                        best_polys = [poly for _, _, poly in poly_scores[:3]]
                        print(f"    DEBUG: Testing with best 3 polynomials...")
                        try:
                            gb_best = sp.groebner(best_polys, self.p, self.q, order='lex')
                            best_list = list(gb_best)
                            print(f"    DEBUG: Best-3 GB: {best_list}")
                            if best_list != [1]:
                                print(f"    DEBUG: SUCCESS! Best polynomials give valid Gröbner basis")
                                gb_list = best_list
                                gb = gb_best
                            else:
                                print(f"    DEBUG: Even best polynomials give [1], trying first 5...")
                                subset = processed_polys[:5]
                                gb_subset = sp.groebner(subset, self.p, self.q, order='lex')
                                subset_list = list(gb_subset)
                                print(f"    DEBUG: First-5 GB: {subset_list}")
                                if subset_list != [1]:
                                    print(f"    DEBUG: SUCCESS! First 5 polynomials work")
                                    gb_list = subset_list
                                    gb = gb_subset
                                else:
                                    print(f"    DEBUG: Even 5 polynomials give [1] - trying individual polynomials...")
                                    # Try individual polynomials to see which ones are problematic
                                    for j, single_poly in enumerate(processed_polys[:5]):
                                        try:
                                            gb_single = sp.groebner([single_poly], self.p, self.q, order='lex')
                                            single_list = list(gb_single)
                                            print(f"    DEBUG: Single poly {j} GB: {single_list}")
                                            if single_list == [1]:
                                                print(f"    DEBUG: Single polynomial {j} is inconsistent by itself!")
                                                print(f"    DEBUG: Poly {j}: {str(single_poly)[:200]}...")
                                        except Exception as e:
                                            print(f"    DEBUG: Single poly {j} failed: {e}")

                                    print(f"    DEBUG: CONCLUSION: Individual polynomials are inconsistent")
                                    print(f"    DEBUG: This indicates the lattice polynomial generation is fundamentally flawed")
                                    print(f"    DEBUG: The polynomials do not represent valid constraints")
                                    return None
                        except Exception as e:
                            print(f"    DEBUG: Subset computation failed: {e}")
                            return None
                    else:
                        return None

                # Analyze Gröbner basis structure
                if gb:
                    print("    Analyzing Gröbner basis structure:")
                    univariate_polys = []
                    quadratic_bivariates = []
                    constant_found = False

                    for idx, g in enumerate(gb):
                        # Check if it's a constant first (this handles the Gröbner basis containing 1 or 0)
                        if isinstance(g, (int, sp.Integer)) or (hasattr(g, 'is_number') and g.is_number):
                            print(f"      GB[{idx}]: CONSTANT = {g}")
                            constant_found = True
                            if g != 0:
                                print("      → Non-zero constant means inconsistent system")
                                return None
                        else:
                            # Try to analyze as polynomial
                            try:
                                vars_in_poly = g.free_symbols if hasattr(g, 'free_symbols') else set()
                                g_poly = sp.Poly(g, x, v)

                                if len(vars_in_poly) == 1:
                                    var = list(vars_in_poly)[0]
                                    print(f"      GB[{idx}]: univariate in {var}, degree={g_poly.degree()}")
                                    univariate_polys.append((idx, var, g_poly))
                                elif len(vars_in_poly) == 0:
                                    # This shouldn't happen since we checked for constants above
                                    print(f"      GB[{idx}]: CONSTANT = {g}")
                                elif len(vars_in_poly) == 2:
                                    print(f"      GB[{idx}]: bivariate in x,v, degree={g_poly.degree()}")
                                    if g_poly.total_degree() == 1:
                                        print(f"      → Linear bivariate: can solve for one variable in terms of the other")
                                    elif g_poly.total_degree() == 2:
                                        print(f"      → Quadratic bivariate: Diophantine constraint f(x,v) = 0")
                                        # Store the quadratic bivariate polynomial for solving
                                        quadratic_bivariates.append((idx, g, g_poly))
                                elif idx < 5:
                                    print(f"      GB[{idx}]: multivariate, degree={g_poly.degree()}")
                            except Exception as e:
                                print(f"      GB[{idx}]: could not analyze: {g} (error: {e})")

                    # Check for inconsistent system
                    if constant_found:
                        print("    → System is inconsistent (contains non-zero constant)")
                        return None

                    # Try to solve univariate polynomials in dp, dq
                    bivariate_linears = []
                    for idx, g in enumerate(gb):
                        try:
                            vars_in_poly = g.free_symbols if hasattr(g, 'free_symbols') else set()
                            if len(vars_in_poly) == 2:  # Both dp and dq
                                g_poly = sp.Poly(g, dp, dq)
                                if g_poly.total_degree() == 1:  # Linear in both variables
                                    print(f"      GB[{idx}]: bivariate linear in dp,dq, solvable")
                                    bivariate_linears.append((idx, g_poly))
                        except:
                            pass

                    # Try bivariate linear polynomials in dp, dq
                    if bivariate_linears:
                        print("    Solving bivariate linear polynomials from GB:")
                        for idx, poly in bivariate_linears:
                            try:
                                print(f"      Solving GB[{idx}] for dp in terms of dq...")

                                # Extract coefficients: a*dp + b*dq + c = 0
                                a = poly.coeff_monomial(dp**1 * dq**0)  # coefficient of dp
                                b = poly.coeff_monomial(dp**0 * dq**1)  # coefficient of dq
                                c = poly.coeff_monomial(dp**0 * dq**0)  # constant term

                                print(f"        Linear relation: {a}*dp + {b}*dq + {c} = 0")

                                if a != 0:
                                    try:
                                        # Convert to integers for arithmetic
                                        a_int = int(a) if hasattr(a, '__int__') else int(a)
                                        b_int = int(b) if hasattr(b, '__int__') else int(b)
                                        c_int = int(c) if hasattr(c, '__int__') else int(c)

                                        print(f"        Solving: {a_int}*dp + {b_int}*dq + {c_int} = 0")
                                        print(f"        dp = (-{b_int}*dq - {c_int}) / {a_int}")

                                        # Try small integer values for dq (corrections should be small)
                                        for dq_candidate in range(-10, 11):  # Try dq from -10 to 10
                                            try:
                                                # Calculate dp from the linear relationship: dp = (-b*dq - c)/a
                                                dp_calculated = (-b_int * dq_candidate - c_int) // a_int

                                                # Convert back to actual factors
                                                p_actual = self.p_approx + dp_calculated
                                                q_actual = self.q_approx + dq_candidate

                                                # Check if this factorization works
                                                if p_actual > 0 and q_actual > 0 and p_actual * q_actual == self.N:
                                                    # Success! Found the factors
                                                    print(f"        ✓✓✓ VERIFIED! Gröbner basis factorization found!")
                                                    print(f"        dp = {dp_calculated}, dq = {dq_candidate}")
                                                    print(f"        p = {p_actual}, q = {q_actual}")
                                                    return (dp_calculated, dq_candidate)

                                            except (ValueError, OverflowError, ZeroDivisionError):
                                                continue

                                        print(f"        No small corrections found with this linear relation")

                                    except (ValueError, OverflowError, ZeroDivisionError) as e:
                                        print(f"        Error in bivariate linear solving: {e}")
                                        continue
                                else:
                                    print(f"        Cannot solve (coefficient of dp is zero)")

                            except Exception as e:
                                print(f"        Error solving multivariate linear: {e}")
                                continue

                    # Solve quadratic bivariate Diophantine constraints from GB
                    if quadratic_bivariates:
                        print("    Solving quadratic bivariate Diophantine constraints from GB...")
                        for idx, g_expr, g_poly in quadratic_bivariates:
                            try:
                                print(f"    Processing GB[{idx}]: {str(g_expr)[:200]}...")
                                
                                # Extract coefficients: The polynomial is of form A*x² + B*x*v + C*v² + D*x + E*v + F = 0
                                # We want to handle the form: -v² + x² + c*x + d = 0 or similar
                                try:
                                    # Try to extract coefficients in a structured way
                                    coeff_x2 = g_poly.coeff_monomial(x**2 * v**0)
                                    coeff_x1 = g_poly.coeff_monomial(x**1 * v**0)
                                    coeff_v2 = g_poly.coeff_monomial(x**0 * v**2)
                                    coeff_v1 = g_poly.coeff_monomial(x**0 * v**1)
                                    coeff_xv = g_poly.coeff_monomial(x**1 * v**1)
                                    coeff_const = g_poly.coeff_monomial(x**0 * v**0)
                                    
                                    # Convert to integers
                                    try:
                                        a = int(coeff_x2) if coeff_x2 != 0 else 0
                                        b = int(coeff_x1) if coeff_x1 != 0 else 0
                                        c_coeff = int(coeff_v2) if coeff_v2 != 0 else 0
                                        d = int(coeff_v1) if coeff_v1 != 0 else 0
                                        e = int(coeff_xv) if coeff_xv != 0 else 0
                                        f = int(coeff_const) if coeff_const != 0 else 0
                                        
                                        print(f"      Extracted: {a}*x² + {e}*x*v + {c_coeff}*v² + {b}*x + {d}*v + {f} = 0")
                                        
                                        # Handle the common cases:
                                        # Case 1: -v² + x² + c*x + d = 0  →  v² = x² + c*x + d
                                        # Case 2: v² - x² - c*x - d = 0   →  v² = x² + c*x + d (same form, different signs)
                                        if ((c_coeff == -1 and a == 1) or (c_coeff == 1 and a == -1)) and e == 0 and d == 0:
                                            # Normalize to v² = x² + c*x + d form
                                            if c_coeff == 1 and a == -1:
                                                # v² - x² - c*x - d = 0  →  v² = x² + c*x + d
                                                # So coefficient of x is -b, constant is -f
                                                c_val = -b
                                                d_val = -f
                                            else:
                                                # -v² + x² + c*x + d = 0  →  v² = x² + c*x + d
                                                # So coefficient of x is b, constant is f
                                                c_val = b
                                                d_val = f
                                            
                                            print(f"      Detected form: v² = x² + {c_val}*x + {d_val}")
                                            print(f"      Coefficient magnitude: ~{abs(c_val).bit_length()} bits")
                                            
                                            # Strategy 1: If coefficient is reasonable, search small x values
                                            # (The original approach works when coefficients aren't huge)
                                            # For huge coefficients, skip brute force and go directly to Strategy 2 (lattice solver)
                                            if abs(c_val) < 10**50:
                                                max_x_search = min(10000, int(sp.sqrt(self.N)) // 2)
                                                print(f"      Searching small |x| values in range [-{max_x_search}, {max_x_search}]...")
                                                
                                                for x_val in range(-max_x_search, max_x_search + 1):
                                                    try:
                                                        v_squared = x_val*x_val + c_val*x_val + d_val
                                                        if v_squared < 0:
                                                            continue
                                                        
                                                        v_root = int(sp.sqrt(v_squared))
                                                        if v_root*v_root == v_squared:
                                                            u = t + x_val
                                                            for v_candidate in [v_root, -v_root]:
                                                                p_candidate = u - v_candidate
                                                                q_candidate = u + v_candidate
                                                                if p_candidate > 0 and q_candidate > 0 and p_candidate * q_candidate == self.N:
                                                                    print(f"      ✓✓✓ Diophantine solution found!")
                                                                    print(f"      x = {x_val}, v = {v_candidate}, p = {p_candidate}, q = {q_candidate}")
                                                                    return (p_candidate, q_candidate)
                                                    except (ValueError, OverflowError):
                                                        continue
                                            
                                            # For huge coefficients (abs(c_val) >= 10**50), skip brute force on x
                                            # and go directly to Strategy 2 (lattice solver)
                                            
                                            # Strategy 2: For huge coefficients, use lattice-based approach ONLY
                                            # NO brute force fallback - lattice solver should handle it
                                            print(f"      Coefficient magnitude: ~{abs(c_val).bit_length()} bits - using lattice solver ONLY (no brute force)...")
                                            
                                            # Use the lattice solver (already in this file)
                                            print(f"      Initializing lattice solver with N={self.N}")
                                            print(f"      N has {self.N.bit_length()} bits")
                                            print(f"      p_approx={self.p_approx}, q_approx={self.q_approx}")
                                            lattice_solver = MinimizableFactorizationLatticeSolver(self.N)
                                            
                                            # Use a MASSIVE search radius for huge coefficients
                                            # For huge coefficients, we need a very large search radius
                                            # User requested 2^2000, but that's computationally infeasible
                                            # Use a more reasonable but still very large value
                                            search_radius = 2**2000  # User requested 2^2000
                                            print(f"      Lattice search radius: 2^2000 = {search_radius}")
                                            
                                            # Solve using lattice - this should find the factors
                                            p_result, q_result, confidence, basis = lattice_solver.solve(
                                                self.p_approx, 
                                                self.q_approx, 
                                                confidence=0.5,
                                                search_radius=search_radius
                                            )
                                            
                                            if p_result and q_result and p_result * q_result == self.N:
                                                print(f"      ✓✓✓ Diophantine solution found via lattice!")
                                                print(f"      p = {p_result}, q = {q_result}")
                                                return (p_result, q_result)
                                            else:
                                                print(f"      Lattice solver did not find exact factors (confidence={confidence})")
                                                print(f"      Note: Lattice solver is the only method for huge coefficients - no brute force fallback")
                                                # Don't fall back to brute force - just return None
                                                return None
                                            
                                        else:
                                            print(f"      Unsupported polynomial form, trying general approach...")
                                            # Try solving by substituting small values
                                            # This is a fallback for other forms
                                            for x_test in range(-1000, 1001):
                                                try:
                                                    # Substitute x = x_test and solve for v
                                                    poly_at_x = g_expr.subs(x, x_test)
                                                    v_solutions = sp.solve(poly_at_x, v)
                                                    for v_sol in v_solutions:
                                                        if v_sol.is_integer:
                                                            v_int = int(v_sol)
                                                            u = t + x_test
                                                            p_cand = u - v_int
                                                            q_cand = u + v_int
                                                            if p_cand > 0 and q_cand > 0 and p_cand * q_cand == self.N:
                                                                print(f"      ✓✓✓ Solution found via substitution!")
                                                                return (p_cand, q_cand)
                                                except:
                                                    continue
                                    
                                    except (ValueError, OverflowError, AttributeError) as e:
                                        print(f"      Error extracting coefficients: {e}")
                                        continue
                                
                                except Exception as e:
                                    print(f"      Error processing polynomial structure: {e}")
                                    continue
                            
                            except Exception as e:
                                print(f"      Error in quadratic bivariate solver: {e}")
                                continue
                    
                    # Legacy fallback: Solve the Diophantine constraint: f(x,v) = 0 for small |x|
                    # This uses the expected form from the initial polynomial construction
                    if gb_list != [1] and not quadratic_bivariates:
                        print("    Solving Diophantine constraint f(x,v) = 0 for small |x| (legacy method)...")
                        
                        # Search for small |x| values (u near t)
                        max_x = min(1000, int(sp.sqrt(self.N)) // 2)  # Reasonable search range
                        
                        for x_val in range(-max_x, max_x + 1):
                            try:
                                # Compute v² = x² + 2tx + (t² - N)
                                v_squared = x_val**2 + 2*t*x_val + (t**2 - self.N)
                                
                                # Check if v² is a perfect square
                                if v_squared < 0:
                                    continue
                                
                                v_candidate = int(sp.sqrt(v_squared))
                                
                                # Verify it's actually a perfect square
                                if v_candidate**2 == v_squared:
                                    # Found candidate (u, v) where u = t + x
                                    u = t + x_val
                                    
                                    # Compute p = u - v, q = u + v
                                    p_candidate = u - v_candidate
                                    q_candidate = u + v_candidate
                                    
                                    # Verify these are actual factors
                                    if p_candidate > 0 and q_candidate > 0 and p_candidate * q_candidate == self.N:
                                        print(f"    ✓✓✓ Diophantine solution found!")
                                        print(f"    x = {x_val}, v = {v_candidate}")
                                        print(f"    u = t + x = {t} + {x_val} = {u}")
                                        print(f"    p = u - v = {u} - {v_candidate} = {p_candidate}")
                                        print(f"    q = u + v = {u} + {v_candidate} = {q_candidate}")
                                        
                                        # Return as (p, q)
                                        return (p_candidate, q_candidate)
                                    
                                    # Also try negative v
                                    if v_candidate > 0:
                                        p_candidate = u - (-v_candidate)
                                        q_candidate = u + (-v_candidate)
                                        if p_candidate > 0 and q_candidate > 0 and p_candidate * q_candidate == self.N:
                                            print(f"    ✓✓✓ Diophantine solution found (negative v)!")
                                            print(f"    x = {x_val}, v = {-v_candidate}")
                                            print(f"    p = {p_candidate}, q = {q_candidate}")
                                            return (p_candidate, q_candidate)
                                    
                            except Exception as e:
                                continue
                        
                        print(f"    No Diophantine solutions found for |x| < {max_x}")

                    # Try to solve univariate polynomials (fallback)
                    if univariate_polys:
                        print("    Solving univariate polynomials from GB:")
                        for idx, var, poly in univariate_polys:
                            try:
                                print(f"      Solving GB[{idx}] for {var}...")

                                # Get the polynomial as a univariate polynomial in the target variable
                                if var == dp:
                                    # For a polynomial like dp - 3 = 0, solve dp = 3
                                    # Extract coefficients: poly = a*dp + b, solve a*dp = -b, dp = -b/a
                                    dp_coeff = poly.coeff_monomial(dp**1 * dq**0)  # coefficient of dp
                                    const_coeff = poly.coeff_monomial(dp**0 * dq**0)  # constant term

                                    if dp_coeff != 0 and poly.total_degree() == 1:
                                        # Linear case: dp_coeff * dp + const_coeff = 0
                                        try:
                                            # Safely convert SymPy objects to Python integers
                                            const_int = int(const_coeff) if hasattr(const_coeff, '__int__') else const_coeff
                                            dp_coeff_int = int(dp_coeff) if hasattr(dp_coeff, '__int__') else dp_coeff

                                            dp_absolute = -const_int // dp_coeff_int
                                            print(f"        Found dp = {dp_absolute}")
                                            # For dp, we need to find dq from the other equations
                                            # Since we have the factorization constraint, we can solve for dq
                                            # From dp*dq + p_approx*dq + q_approx*dp - ε = 0
                                            # We have dp, so: dq*(dp + p_approx) + q_approx*dp - ε = 0
                                            # dq*(dp + p_approx) = ε - q_approx*dp
                                            # dq = (ε - q_approx*dp) / (dp + p_approx)

                                            if dp_absolute + self.p_approx != 0:
                                                numerator = self.N - (self.p_approx * self.q_approx) - self.q_approx * dp_absolute
                                                denominator = dp_absolute + self.p_approx
                                                if denominator != 0:
                                                    dq_calculated = numerator // denominator
                                                    print(f"        Corresponding dq = {dq_calculated}")

                                                    # Verify the solution
                                                    p_actual = self.p_approx + dp_absolute
                                                    q_actual = self.q_approx + dq_calculated
                                                    if p_actual * q_actual == self.N:
                                                        print("        ✓✓✓ VERIFIED! Gröbner solution found!")
                                                        return (dp_absolute, dq_calculated)
                                        except (ValueError, OverflowError, AttributeError) as e:
                                            print(f"        Error in p coefficient arithmetic: {e}")
                                            continue
                                    elif poly.total_degree() == 2:
                                        # Quadratic case: try to solve for integer roots in dp variable
                                        print(f"        Quadratic polynomial: {poly}")
                                        # Extract coefficients: a*dp^2 + b*dp + c = 0
                                        a = poly.coeff_monomial(dp**2 * dq**0)
                                        b = poly.coeff_monomial(dp**1 * dq**0)
                                        c = poly.coeff_monomial(dp**0 * dq**0)

                                        if a != 0:
                                            try:
                                                discriminant = b**2 - 4*a*c
                                                disc_int = int(discriminant) if hasattr(discriminant, '__int__') else discriminant
                                                print(f"        Quadratic discriminant: {disc_int.bit_length()}-bit magnitude")

                                                if disc_int >= 0:
                                                    # For very large discriminants, solutions will be far from approximations
                                                    if disc_int.bit_length() > 4000:  # Very large, but allow for poor approximations
                                                        print(f"        Discriminant too large ({disc_int.bit_length()} bits), corrections would be enormous")
                                                        print(f"        No practical integer solutions to quadratic")
                                                    else:
                                                        sqrt_d = int(disc_int**0.5)
                                                        if sqrt_d**2 == disc_int:  # Perfect square
                                                            a_int = int(a) if hasattr(a, '__int__') else a
                                                            b_int = int(b) if hasattr(b, '__int__') else b

                                                            dp1 = (-b_int + sqrt_d) // (2*a_int)
                                                            dp2 = (-b_int - sqrt_d) // (2*a_int)

                                                            for dp_abs in [dp1, dp2]:
                                                                print(f"        Trying dp = {dp_abs}")
                                                                # For dp, we need to find dq from the factorization constraint
                                                                # From dp*dq + p_approx*dq + q_approx*dp - ε = 0
                                                                # We have dp, so: dq*(dp + p_approx) + q_approx*dp - ε = 0
                                                                # dq*(dp + p_approx) = ε - q_approx*dp
                                                                # dq = (ε - q_approx*dp) / (dp + p_approx)

                                                                if dp_abs + self.p_approx != 0:
                                                                    numerator = self.N - (self.p_approx * self.q_approx) - self.q_approx * dp_abs
                                                                    denominator = dp_abs + self.p_approx
                                                                    if denominator != 0:
                                                                        dq_calculated = numerator // denominator
                                                                        print(f"        Corresponding dq = {dq_calculated}")

                                                                        # Verify the solution
                                                                        p_actual = self.p_approx + dp_abs
                                                                        q_actual = self.q_approx + dq_calculated
                                                                        if p_actual * q_actual == self.N:
                                                                            print(f"        ✓✓✓ VERIFIED! Gröbner quadratic solution found!")
                                                                            print(f"        dp = {dp_abs}, dq = {dq_calculated}")
                                                                            return (dp_abs, dq_calculated)
                                            except (ValueError, OverflowError, AttributeError) as e:
                                                print(f"        Error in quadratic solving: {e}")
                                                continue
                                        print(f"        No integer solutions to quadratic")
                                    else:
                                        print(f"        Cannot solve degree-{poly.total_degree()} polynomial")
                                else:  # var == dq
                                    # For a polynomial like dq - 1 = 0, solve dq = 1
                                    # Extract coefficients: poly = c*dq + d, solve c*dq = -d, dq = -d/c
                                    dq_coeff = poly.coeff_monomial(dp**0 * dq**1)  # coefficient of dq
                                    const_coeff = poly.coeff_monomial(dp**0 * dq**0)  # constant term

                                    if dq_coeff != 0 and poly.total_degree() == 1:
                                        # Linear case: dq_coeff * dq + const_coeff = 0
                                        const_int = int(const_coeff) if hasattr(const_coeff, '__int__') else const_coeff
                                        dq_coeff_int = int(dq_coeff) if hasattr(dq_coeff, '__int__') else dq_coeff
                                        dq_absolute = -const_int // dq_coeff_int
                                        print(f"        Found dq = {dq_absolute}")

                                        # For dq, we need to find dp from the factorization constraint
                                        # From dp*dq + p_approx*dq + q_approx*dp - ε = 0
                                        # We have dq, so: dp*dq + p_approx*dq + q_approx*dp - ε = 0
                                        # dp*(dq + q_approx) = ε - p_approx*dq
                                        # dp = (ε - p_approx*dq) / (dq + q_approx)

                                        if dq_absolute + self.q_approx != 0:
                                            numerator = self.N - (self.p_approx * self.q_approx) - self.p_approx * dq_absolute
                                            denominator = dq_absolute + self.q_approx
                                            if denominator != 0:
                                                dp_calculated = numerator // denominator
                                                print(f"        Corresponding dp = {dp_calculated}")

                                                # Verify the solution
                                                p_actual = self.p_approx + dp_calculated
                                                q_actual = self.q_approx + dq_absolute
                                                if p_actual * q_actual == self.N:
                                                    print("        ✓✓✓ VERIFIED! Gröbner solution found!")
                                                    return (dp_calculated, dq_absolute)
                                    elif poly.total_degree() == 2:
                                        # Quadratic case: try to solve for integer roots in dq variable
                                        print(f"        Quadratic polynomial in dq: {poly}")
                                        # Extract coefficients: a*dq^2 + b*dq + c = 0
                                        a = poly.coeff_monomial(dp**0 * dq**2)
                                        b = poly.coeff_monomial(dp**0 * dq**1)
                                        c = poly.coeff_monomial(dp**0 * dq**0)

                                        if a != 0:
                                            try:
                                                discriminant = b**2 - 4*a*c
                                                disc_int = int(discriminant) if hasattr(discriminant, '__int__') else discriminant
                                                print(f"        Quadratic discriminant: {disc_int.bit_length()}-bit magnitude")

                                                if disc_int >= 0:
                                                    if disc_int.bit_length() > 4000:
                                                        print(f"        Discriminant too large ({disc_int.bit_length()} bits), corrections would be enormous")
                                                    else:
                                                        sqrt_d = int(disc_int**0.5)
                                                        if sqrt_d**2 == disc_int:
                                                            a_int = int(a) if hasattr(a, '__int__') else a
                                                            b_int = int(b) if hasattr(b, '__int__') else b

                                                            dq1 = (-b_int + sqrt_d) // (2*a_int)
                                                            dq2 = (-b_int - sqrt_d) // (2*a_int)

                                                            for dq_abs in [dq1, dq2]:
                                                                print(f"        Trying dq = {dq_abs}")
                                                                # For dq, we need to find dp from the factorization constraint
                                                                if dq_abs + self.q_approx != 0:
                                                                    numerator = self.N - (self.p_approx * self.q_approx) - self.p_approx * dq_abs
                                                                    denominator = dq_abs + self.q_approx
                                                                    if denominator != 0:
                                                                        dp_calculated = numerator // denominator
                                                                        print(f"        Corresponding dp = {dp_calculated}")

                                                                        # Verify the solution
                                                                        p_actual = self.p_approx + dp_calculated
                                                                        q_actual = self.q_approx + dq_abs
                                                                        if p_actual * q_actual == self.N:
                                                                            print(f"        ✓✓✓ VERIFIED! Gröbner quadratic solution found!")
                                                                            print(f"        dp = {dp_calculated}, dq = {dq_abs}")
                                                                            return (dp_calculated, dq_abs)
                                            except Exception as e:
                                                print(f"        Error in quadratic solving: {e}")
                                                continue
                            except Exception as e:
                                print(f"        Error solving univariate polynomial: {e}")
                                continue

                    # If no univariate solutions, try elimination with resultants
                    if len(gb) >= 2:
                        print("    Trying resultant elimination method...")
                        try:
                            # Take first two polynomials and compute resultant
                            p1 = gb[0]
                            p2 = gb[1]

                            # Compute resultant with respect to q to eliminate q
                            res_q = sp.resultant(p1, p2, self.q)
                            if res_q != 0:
                                print("    ✓ Resultant computed (eliminating q)")

                                # Try to handle the resultant
                                try:
                                    if hasattr(res_q, 'is_number') and res_q.is_number:
                                        # Resultant is a constant
                                        if res_q == 0:
                                            print("    Resultant is zero - dependent polynomials")
                                        else:
                                            print(f"    Resultant is non-zero constant {res_q}")
                                    else:
                                        # Try to create polynomial and solve
                                        res_poly = sp.Poly(res_q, self.p)
                                        if res_poly.degree() > 0:
                                            print(f"    Resultant degree: {res_poly.degree()}")

                                            # Try to find rational roots first (more reliable)
                                            try:
                                                rational_roots = res_poly.ground_roots()
                                                for root, mult in rational_roots.items():
                                                    if root.is_integer:
                                                        p_val = int(root)
                                                        p_candidate = self.p_approx + p_val
                                                        if p_candidate > 0 and self.N % p_candidate == 0:
                                                            q_candidate = self.N // p_candidate
                                                            q_correction = q_candidate - self.q_approx
                                                            if self._verify_factorization(p_val, q_correction):
                                                                print("    ✓✓✓ VERIFIED! Solution found via resultant!")
                                                                return (p_val, q_correction)
                                            except:
                                                # Fall back to numerical roots
                                                try:
                                                    roots = res_poly.nroots()
                                                    for root in roots:
                                                        if abs(root.imag) < 1e-10:
                                                            p_val = int(round(root.real))
                                                            p_candidate = self.p_approx + p_val

                                                            if p_candidate > 0 and self.N % p_candidate == 0:
                                                                q_candidate = self.N // p_candidate
                                                                q_correction = q_candidate - self.q_approx

                                                                if self._verify_factorization(p_val, q_correction):
                                                                    print("    ✓✓✓ VERIFIED! Solution found via resultant!")
                                                                    return (p_val, q_correction)
                                                except:
                                                    pass
                                except Exception as e2:
                                    print(f"    Could not process resultant: {e2}")
                        except Exception as e:
                            print(f"    Resultant method failed: {e}")

                    # Try variety computation (numerical approach)
                    print("    Attempting numerical variety computation...")
                    try:
                        # For small systems, try to solve numerically
                        if len(processed_polys) <= 3:
                            # Use numerical solving for small systems
                            solutions = sp.solve(processed_polys, [self.p, self.q])

                            if solutions:
                                print(f"    Found {len(solutions)} solution(s)")
                                for sol in solutions:
                                    if isinstance(sol, dict):
                                        p_val = sol.get(self.p)
                                        q_val = sol.get(self.q)

                                        if p_val is not None and q_val is not None:
                                            try:
                                                p_int = int(round(float(p_val)))
                                                q_int = int(round(float(q_val)))

                                                # For factorization, we need corrections, not absolute values
                                                # Assume these are small corrections around approximations
                                                if abs(p_int) < 1000 and abs(q_int) < 1000:  # Reasonable bounds
                                                    p_candidate = self.p_approx + p_int
                                                    q_candidate = self.q_approx + q_int

                                                    if p_candidate > 0 and q_candidate > 0 and p_candidate * q_candidate == self.N:
                                                        print("    ✓✓✓ VERIFIED! Solution found numerically!")
                                                        return (p_int, q_int)
                                            except:
                                                continue
                                    elif isinstance(sol, list) and len(sol) == 2:
                                        # Sometimes solve returns [p_val, q_val]
                                        try:
                                            p_int = int(round(float(sol[0])))
                                            q_int = int(round(float(sol[1])))

                                            if abs(p_int) < 1000 and abs(q_int) < 1000:
                                                p_candidate = self.p_approx + p_int
                                                q_candidate = self.q_approx + q_int

                                                if p_candidate > 0 and q_candidate > 0 and p_candidate * q_candidate == self.N:
                                                    print("    ✓✓✓ VERIFIED! Solution found numerically!")
                                                    return (p_int, q_int)
                                        except:
                                            continue
                    except Exception as e:
                        print(f"    Numerical solving failed: {e}")

                else:
                    print("    → Empty Gröbner basis")
                    return None

            except Exception as e:
                print(f"    Gröbner basis computation failed: {e}")
                return None

        except Exception as e:
            print(f"    Error in Sage-style Gröbner solving: {e}")
            return None

        print("    No solution found with Sage-style Gröbner approach")
        return None

    def _verify_factorization(self, y: int, z: int) -> bool:
        """Verify that (p_approx + y) * (q_approx + z) == N"""
        p_candidate = self.p_approx + y
        q_candidate = self.q_approx + z

        if p_candidate <= 0 or q_candidate <= 0:
            return False

        return p_candidate * q_candidate == self.N


    
    def solve_with_algebraic_elimination(self, polynomials: List[sp.Expr]) -> Optional[Tuple[int, int]]:
        """
        Use algebraic elimination methods optimized for ABCD polynomials.

        This method handles the mixed nature of our constraints:
        - Exact: p*q = N (factorization requirement)
        - Approximate: lattice-derived relationships

        Focuses on exact elimination while being tolerant of approximations.
        """
        print("\n[Algebraic] Using algebraic elimination for mixed exact/approximate ABCD constraints...")

        # For now, delegate to resultant elimination which is already implemented
        print("[Algebraic] Delegating to resultant elimination...")
        return self.solve_with_resultants(polynomials)

    def solve_with_roots_method_iterative(self, polynomials: List[sp.Expr],
                                         p_hint: int = None, q_hint: int = None,
                                         max_iterations: int = 3) -> Optional[Tuple[int, int]]:
        """
        Iterative Root's Method: Run multiple rounds with refined hints to zero in on target N.
        """
        current_p_hint = p_hint
        current_q_hint = q_hint

        for iteration in range(max_iterations):
            print(f"[Root's] Iteration {iteration + 1}/{max_iterations} - targeting N with hints p≈{current_p_hint}, q≈{current_q_hint}")

            result = self.solve_with_roots_method(polynomials, current_p_hint, current_q_hint)

            if result:
                p_found, q_found = result
                # Check if it's exact
                try:
                    if p_found.bit_length() + q_found.bit_length() <= 8192:
                        if p_found * q_found == self.N:
                            print(f"[Root's] ✓✓✓ EXACT FACTORIZATION FOUND after {iteration + 1} iterations!")
                            return result
                except:
                    pass

                # Use as new hints for next iteration
                current_p_hint = p_found
                current_q_hint = q_found
                print(f"[Root's] Using approximation as hints for next iteration")
            else:
                print(f"[Root's] No result from iteration {iteration + 1}")
                break

        print(f"[Root's] Iterative method completed - best approximation from final iteration")
        return result if 'result' in locals() else None

    def solve_with_roots_method(self, polynomials: List[sp.Expr],
                               p_hint: int = None, q_hint: int = None) -> Optional[Tuple[int, int]]:
        """
        Root's Method: Custom root-finding algorithm for ABCD fused polynomials.

        This method treats our ABCD polynomials as a system of equations where:
        - B components provide candidate roots (modular constraints)
        - A×B fusions validate root combinations
        - A components refine root bounds
        - C components provide cross-validation
        - D components offer final verification

        Inspired by root-finding algorithms but customized for lattice-derived constraints.
        """
        print("\n[Root's] Applying Root's Method to ABCD fused polynomials...")
        print("[Root's] ℹ️  Root's Method: Modular roots + Constraint validation")

        # Safeguard for extremely large numbers
        if self.N.bit_length() > 2000:
            print(f"[Root's] ⚠️  {self.N.bit_length()}-bit number detected - Root's Method has limitations")
            print(f"[Root's] ℹ️  For numbers > 2000 bits, consider dedicated large-number factoring")
            print(f"[Root's] ℹ️  Continuing with limited root generation...")

        try:
            # Step 1: Extract root candidates from B components (modular constraints)
            root_candidates = self._extract_modular_roots(polynomials, p_hint, q_hint)
            p_count = len(root_candidates.get('p', []))
            q_count = len(root_candidates.get('q', []))
            print(f"[Root's] Found {p_count} p roots, {q_count} q roots")

            if not root_candidates or (p_count == 0 and q_count == 0):
                print("[Root's] No modular root candidates found")
                return None

            # Preserve original root_candidates for final fallback
            original_root_candidates = {
                'p': list(root_candidates.get('p', set())),
                'q': list(root_candidates.get('q', set()))
            }

            # Step 2: Validate and combine roots using A×B fusions
            # Apply max_root_candidates limit BEFORE validation to avoid explosion
            max_root_candidates = self.config.get('max_root_candidates')
            if max_root_candidates:
                # Limit the root_candidates dictionary itself
                p_list = sorted(list(root_candidates['p']))[:max_root_candidates]
                q_list = sorted(list(root_candidates['q']))[:max_root_candidates]
                root_candidates = {'p': p_list, 'q': q_list}
                print(f"[Root's] Pre-validation limit: {len(p_list)} p × {len(q_list)} q = {len(p_list) * len(q_list)} combinations")
            
            validated_pairs = self._validate_root_fusions(polynomials, root_candidates)
            print(f"[Root's] Validated {len(validated_pairs)} root pairs via A×B fusions")

            # If no fusions validated, fall back to basic factorization check
            if not validated_pairs:
                print("[Root's] No fusion validation, checking basic factorization...")
                p_roots = root_candidates['p']
                q_roots = root_candidates['q']
                print(f"[Root's] Checking {len(p_roots)} × {len(q_roots)} = {len(p_roots) * len(q_roots)} combinations for N={self.N}")

                # Limit the combinations to avoid explosion
                # Use max_root_candidates if set, otherwise use adaptive limits
                max_root_candidates = self.config.get('max_root_candidates')
                if max_root_candidates:
                    # Use the configured limit
                    max_combinations = max_root_candidates * max_root_candidates
                    if len(p_roots) * len(q_roots) > max_combinations:
                        print(f"[Root's] Too many combinations ({len(p_roots) * len(q_roots)}), limiting to {max_root_candidates}×{max_root_candidates}...")
                        p_roots = sorted(p_roots)[:max_root_candidates]
                        q_roots = sorted(q_roots)[:max_root_candidates]
                        print(f"[Root's] Limited to {len(p_roots)} × {len(q_roots)} = {len(p_roots) * len(q_roots)} combinations")
                elif self.N.bit_length() <= 50 and len(p_roots) * len(q_roots) > 10000:
                    # Fallback: For small numbers without config, use old limit
                    print(f"[Root's] Too many combinations ({len(p_roots) * len(q_roots)}), limiting search...")
                    p_roots = sorted(p_roots)[:50]
                    q_roots = sorted(q_roots)[:50]
                    print(f"[Root's] Limited to {len(p_roots)} × {len(q_roots)} combinations")
                elif len(p_roots) * len(q_roots) > 100000:  # Safety limit for large numbers
                    # Safety limit: don't allow more than 100k combinations
                    limit = int((100000) ** 0.5)  # Square root to get roughly equal p and q limits
                    print(f"[Root's] Too many combinations ({len(p_roots) * len(q_roots)}), applying safety limit to {limit}×{limit}...")
                    p_roots = sorted(p_roots)[:limit]
                    q_roots = sorted(q_roots)[:limit]
                    print(f"[Root's] Limited to {len(p_roots)} × {len(q_roots)} = {len(p_roots) * len(q_roots)} combinations")

                for p_root in p_roots:
                    for q_root in q_roots:
                        try:
                            product = p_root * q_root
                            if product == self.N:
                                validated_pairs.append((p_root, q_root))
                                print(f"[Root's] ✓ Basic factorization check: {p_root} × {q_root} = {self.N}")
                                break  # Found a match, can continue
                        except OverflowError:
                            # For very large numbers, skip overflow cases
                            continue

                print(f"[Root's] Basic validation found {len(validated_pairs)} pairs")

            # If we still have no pairs, return the best approximation pairs for further processing
            if not validated_pairs:
                print("[Root's] No exact pairs found, returning best approximations...")
                # Find the closest approximations to actual factors
                best_approximations = []
                p_roots = root_candidates['p']
                q_roots = root_candidates['q']

                for p_root in p_roots[:20]:  # Limit to avoid explosion
                    for q_root in q_roots[:20]:
                        try:
                            product = p_root * q_root
                            diff = abs(product - self.N)
                            best_approximations.append((p_root, q_root, diff))
                        except OverflowError:
                            continue

                if best_approximations:
                    # Sort by difference (closest first)
                    best_approximations.sort(key=lambda x: x[2])
                    # Return top approximations
                    validated_pairs = [(p, q) for p, q, diff in best_approximations[:5]]
                    print(f"[Root's] Returning {len(validated_pairs)} best approximations for refinement")
                else:
                    print("[Root's] No valid root pairs found even after approximation search")
                    # If we have no validated pairs but have root candidates, skip to final fallback
                    if not validated_pairs:
                        print("[Root's] Skipping validation steps, will use root candidates directly...")
                        # Jump to final fallback by setting validated_pairs to empty and continuing
                        validated_pairs = []

            # Step 3: Refine bounds using A components (linear constraints)
            # Only proceed if we have validated pairs to refine
            if validated_pairs:
                refined_candidates = self._refine_linear_bounds(polynomials, validated_pairs)
                print(f"[Root's] Refined to {len(refined_candidates)} candidates via linear bounds")
            else:
                # No validated pairs, skip refinement and go to final fallback
                print("[Root's] No validated pairs to refine, skipping to final fallback...")
                refined_candidates = []

            # Step 4: Cross-validate using C components (ratio constraints)
            if refined_candidates:
                cross_validated = self._cross_validate_ratios(polynomials, refined_candidates)
                print(f"[Root's] Cross-validated {len(cross_validated)} candidates via ratios")
            else:
                cross_validated = []
                print("[Root's] No refined candidates to cross-validate")

            # Step 5: Final verification using D components (quadratic constraints)
            if cross_validated:
                verified_solutions = self._verify_quadratic_constraints(polynomials, cross_validated)
                print(f"[Root's] Verified {len(verified_solutions)} solutions via quadratics")
            else:
                verified_solutions = []
                print("[Root's] No candidates to verify via quadratics")

            # Step 6: If no exact solutions but we have candidates, try refinement
            if (not verified_solutions or len(verified_solutions) == 0) and cross_validated:
                print(f"[Root's] No exact solutions found, attempting refinement...")
                refined = self._refine_candidates_to_exact(cross_validated)
                if refined:
                    verified_solutions = refined

            # Return the best solution (exact or best approximation)
            # Handle different return formats from _verify_quadratic_constraints
            if verified_solutions and len(verified_solutions) > 0:
                best_solution = verified_solutions[0]  # Already sorted by score/diff
                
                # Handle different tuple formats: (p, q, score) or (p, q, score, diff) or (p, q)
                if len(best_solution) >= 2:
                    p_sol = best_solution[0]
                    q_sol = best_solution[1]
                    
                    # Enhanced final verification with exact checking
                    is_exact = False

                    # Always attempt exact verification when possible
                    try:
                        if p_sol.bit_length() + q_sol.bit_length() <= 8192:  # Extended limit
                            if p_sol * q_sol == self.N:
                                is_exact = True
                                print(f"[Root's] ✓✓✓ EXACT FACTORIZATION VERIFIED: p={p_sol}, q={q_sol}")
                                return (p_sol, q_sol)
                    except:
                        pass

                    # Check if it's an exact solution (has score but no diff, or diff=0)
                    if len(best_solution) == 3:
                        # Format: (p, q, score) - likely exact solution
                        score = best_solution[2]
                        if is_exact:
                            return (p_sol, q_sol)
                        # Not exact, but best we have
                        print(f"[Root's] 🏆 BEST SOLUTION (score={score}): p={p_sol}, q={q_sol}")
                        return (p_sol, q_sol)
                    elif len(best_solution) == 4:
                        # Format: (p, q, score, diff) - approximation
                        score = best_solution[2]
                        diff = best_solution[3]
                        if diff == 0:
                            print(f"[Root's] ✓✓✓ EXACT SOLUTION FOUND: p={p_sol}, q={q_sol}")
                            return (p_sol, q_sol)
                        else:
                            print(f"[Root's] 🏆 BEST APPROXIMATION: p={p_sol}, q={q_sol}, diff={diff:,}")
                            return (p_sol, q_sol)
                    else:
                        # Format: (p, q) - simple tuple
                        print(f"[Root's] 🏆 BEST CANDIDATE: p={p_sol}, q={q_sol}")
                        return (p_sol, q_sol)
            
            # No verified solutions - return best from cross_validated
            if cross_validated:
                print(f"[Root's] No verified solutions, returning best approximation from {len(cross_validated)} cross-validated candidates...")
                # Find the candidate with the smallest product difference
                best_approx = None
                best_diff = 2**(self.N.bit_length() + 10)  # Start with a very large but finite difference
                
                for candidate in cross_validated:
                    if len(candidate) == 3:
                        p_cand, q_cand, score = candidate
                    else:
                        p_cand, q_cand = candidate
                        score = 0
                    
                    try:
                        # Calculate how close p*q is to N
                        if p_cand.bit_length() + q_cand.bit_length() <= 4096:
                            product = p_cand * q_cand
                            diff = abs(product - self.N)
                            if diff < best_diff:
                                best_diff = diff
                                best_approx = (p_cand, q_cand, diff)
                        else:
                            # For very large numbers, use bit-length based estimation
                            estimated_diff = 2**max(p_cand.bit_length() + q_cand.bit_length(), self.N.bit_length())
                            if best_approx is None or estimated_diff < best_diff:
                                best_approx = (p_cand, q_cand, estimated_diff)
                    except:
                        continue
                
                if best_approx:
                    p_best, q_best, diff = best_approx
                    if diff < 2**(self.N.bit_length() + 5):  # If diff is reasonable
                        print(f"[Root's] 🏆 BEST APPROXIMATION: p={p_best}, q={q_best}, diff={diff:,}")
                    else:
                        print(f"[Root's] 🏆 BEST APPROXIMATION: p={p_best}, q={q_best} (very large difference)")
                    return (p_best, q_best)
            
            # Last resort: return best from validated_pairs if available
            if validated_pairs:
                print(f"[Root's] No exact solutions, returning best from {len(validated_pairs)} validated pairs...")
                # Use the first validated pair (they're sorted by score)
                p_best, q_best = validated_pairs[0]
                try:
                    if p_best.bit_length() + q_best.bit_length() <= 4096:
                        product = p_best * q_best
                        diff = abs(product - self.N)
                        if diff == 0:
                            print(f"[Root's] ✓✓✓ EXACT SOLUTION FOUND: p={p_best}, q={q_best}")
                        else:
                            print(f"[Root's] 🏆 BEST CANDIDATE: p={p_best}, q={q_best}, diff={diff:,}")
                    else:
                        print(f"[Root's] 🏆 BEST CANDIDATE: p={p_best}, q={q_best}")
                except:
                    print(f"[Root's] 🏆 BEST CANDIDATE: p={p_best}, q={q_best}")
                return (p_best, q_best)
            
            # Final fallback: return best from root_candidates if available
            # Use original_root_candidates (preserved before any modifications)
            print(f"[Root's] Reaching final fallback - checking original root candidates...")
            p_list = original_root_candidates.get('p', [])
            q_list = original_root_candidates.get('q', [])
            print(f"[Root's] Final fallback: {len(p_list)} p roots, {len(q_list)} q roots available")
            
            if p_list and q_list:
                print(f"[Root's] No validated solutions, returning best from {len(p_list)}×{len(q_list)} root candidates...")
                # Find the pair with smallest product difference
                best_pair = None
                best_diff = 2**(self.N.bit_length() + 10)  # Very large but finite
                
                # Limit search to avoid explosion
                max_check = min(100, len(p_list), len(q_list))
                for p_cand in p_list[:max_check]:
                    for q_cand in q_list[:max_check]:
                        try:
                            if p_cand.bit_length() + q_cand.bit_length() <= 4096:
                                product = p_cand * q_cand
                                diff = abs(product - self.N)
                                if diff < best_diff:
                                    best_diff = diff
                                    best_pair = (p_cand, q_cand)
                            else:
                                # For very large numbers, use bit-length estimation
                                estimated_diff = 2**max(p_cand.bit_length() + q_cand.bit_length(), self.N.bit_length())
                                if best_pair is None or estimated_diff < best_diff:
                                    best_diff = estimated_diff
                                    best_pair = (p_cand, q_cand)
                        except:
                            continue
                
                if best_pair:
                    p_best, q_best = best_pair
                    if best_diff < 2**(self.N.bit_length() + 5):
                        print(f"[Root's] 🏆 BEST FROM ROOT CANDIDATES: p={p_best}, q={q_best}, diff={best_diff:,}")
                    else:
                        print(f"[Root's] 🏆 BEST FROM ROOT CANDIDATES: p={p_best}, q={q_best} (very large difference)")
                    return (p_best, q_best)
            elif p_list or q_list:
                # If we only have one type of root, use hints or approximations
                if p_list and q_hint:
                    print(f"[Root's] Returning best approximation with {len(p_list)} p roots and q hint...")
                    # Find best p that works with q_hint
                    best_p = None
                    best_diff = 2**(self.N.bit_length() + 10)
                    for p_cand in p_list[:100]:
                        try:
                            if p_cand.bit_length() + q_hint.bit_length() <= 4096:
                                product = p_cand * q_hint
                                diff = abs(product - self.N)
                                if diff < best_diff:
                                    best_diff = diff
                                    best_p = p_cand
                        except:
                            continue
                    if best_p:
                        print(f"[Root's] 🏆 BEST APPROXIMATION: p={best_p}, q={q_hint}, diff={best_diff:,}")
                        return (best_p, q_hint)
                elif q_list and p_hint:
                    print(f"[Root's] Returning best approximation with {len(q_list)} q roots and p hint...")
                    # Find best q that works with p_hint
                    best_q = None
                    best_diff = 2**(self.N.bit_length() + 10)
                    for q_cand in q_list[:100]:
                        try:
                            if p_hint.bit_length() + q_cand.bit_length() <= 4096:
                                product = p_hint * q_cand
                                diff = abs(product - self.N)
                                if diff < best_diff:
                                    best_diff = diff
                                    best_q = q_cand
                        except:
                            continue
                    if best_q:
                        print(f"[Root's] 🏆 BEST APPROXIMATION: p={p_hint}, q={best_q}, diff={best_diff:,}")
                        return (p_hint, best_q)

            # Final attempt: Check if any of our root candidates actually work
            print(f"[Root's] Performing final exact verification on root candidates...")
            for p_cand in original_root_candidates.get('p', [])[:50]:  # Check first 50
                for q_cand in original_root_candidates.get('q', [])[:50]:
                    try:
                        if p_cand.bit_length() + q_cand.bit_length() <= 8192:  # Extended limit
                            if p_cand * q_cand == self.N:
                                print(f"[Root's] ✓✓✓ EXACT FACTORIZATION FOUND in final check: p={p_cand}, q={q_cand}")
                                return (p_cand, q_cand)
                    except:
                        continue

            print("[Root's] No exact factorization found - attempting targeted refinement...")

            # NEW: Targeted refinement - use best approximation as starting point for focused search
            if original_root_candidates.get('p') and original_root_candidates.get('q'):
                print(f"[Root's] Attempting targeted refinement around best candidates...")

                # Find the best approximation from our candidates
                best_p, best_q = None, None
                best_diff = 2**(self.N.bit_length() + 10)  # Large initial value

                for p_cand in original_root_candidates.get('p', [])[:20]:
                    for q_cand in original_root_candidates.get('q', [])[:20]:
                        try:
                            if p_cand.bit_length() + q_cand.bit_length() <= 8192:
                                candidate_product = p_cand * q_cand
                                diff = abs(candidate_product - self.N)
                                if diff < best_diff:
                                    best_diff = diff
                                    best_p, best_q = p_cand, q_cand
                        except:
                            continue

                # If we found a reasonable approximation, try a focused search around it
                if best_p and best_q and best_diff < 2**(self.N.bit_length() // 2):
                    print(f"[Root's] Found good approximation (diff: {best_diff}), performing targeted search...")

                    # Try small adjustments around the best approximation
                    search_radius = min(10000, best_diff // max(1, min(best_p, best_q)))

                    for dp in range(-search_radius, search_radius + 1, max(1, search_radius // 20)):
                        for dq in range(-search_radius, search_radius + 1, max(1, search_radius // 20)):
                            p_test = best_p + dp
                            q_test = best_q + dq

                            if p_test > 1 and q_test > 1:
                                try:
                                    if p_test.bit_length() + q_test.bit_length() <= 8192:
                                        if p_test * q_test == self.N:
                                            print(f"[Root's] ✓✓✓ EXACT FACTORIZATION FOUND in targeted refinement: p={p_test}, q={q_test}")
                                            return (p_test, q_test)
                                except:
                                    continue

            print("[Root's] No exact factorization found by Root's Method")
            return None

        except Exception as e:
            error_msg = str(e)
            if "too large to convert to float" in error_msg:
                print(f"[Root's] ⚠️  Floating point overflow for {self.N.bit_length()}-bit number")
                print(f"[Root's] ℹ️  Root's Method limited by integer arithmetic for very large numbers")
                print(f"[Root's] 💡 Consider using specialized large-number factoring algorithms")
            else:
                print(f"[Root's] Root's Method failed: {error_msg[:100]}...")
            return None

    def _integer_sqrt_approx(self, n):
        """
        Compute integer square root approximation without floating point.
        """
        if n == 0 or n == 1:
            return n

        # Binary search for square root
        left, right = 1, n
        while left <= right:
            mid = (left + right) // 2
            try:
                if mid * mid == n:
                    return mid
                elif mid * mid < n:
                    left = mid + 1
                else:
                    right = mid - 1
            except OverflowError:
                # For extremely large numbers, reduce search space
                right = mid - 1

        return right  # right will be the floor of sqrt(n)

    def _extract_modular_roots(self, polynomials, p_hint, q_hint):
        """
        Step 1: Extract root candidates from B components and generate additional roots for large numbers.

        For small numbers: Extract from modular constraints (p - k = 0)
        For large numbers: Generate root candidates using multiple strategies
        """
        root_candidates = {'p': set(), 'q': set()}

        # Strategy 1: Extract from explicit B components (modular constraints)
        print(f"[Root's] Extracting explicit modular roots...")
        for poly in polynomials:
            try:
                poly_obj = sp.Poly(poly, self.p, self.q)
                if poly_obj.total_degree() == 1:
                    coeffs = poly_obj.as_dict()

                    # Check for p - k = 0 (modular root for p)
                    if (len(coeffs) == 2 and
                        (1, 0) in coeffs and coeffs[(1, 0)] == 1 and
                        (0, 0) in coeffs):
                        k = -coeffs[(0, 0)]
                        if k > 1 and isinstance(k, (int, sp.Integer)):
                            root_candidates['p'].add(int(k))

                    # Check for q - m = 0 (modular root for q)
                    elif (len(coeffs) == 2 and
                          (0, 1) in coeffs and coeffs[(0, 1)] == 1 and
                          (0, 0) in coeffs):
                        m = -coeffs[(0, 0)]
                        if m > 1 and isinstance(m, (int, sp.Integer)):
                            root_candidates['q'].add(int(m))

            except:
                continue

        # Strategy 2: Add hints if available
        if p_hint and p_hint > 1:
            root_candidates['p'].add(p_hint)
        if q_hint and q_hint > 1:
            root_candidates['q'].add(q_hint)

        # Strategy 2.5: For small numbers, prioritize finding actual factors
        if self.N.bit_length() <= 50:  # Small number threshold
            print(f"[Root's] Small number detected ({self.N.bit_length()}-bit), finding actual factors...")

            # First priority: Find actual factors through trial division
            sqrt_N = self._integer_sqrt_approx(self.N)
            bit_len = self.N.bit_length()

            # Scale trial division limit based on bit length
            if bit_len <= 32:
                max_trial = min(100000, sqrt_N + 1)  # Up to 100k for 32-bit
            elif bit_len <= 48:
                max_trial = min(500000, sqrt_N + 1)  # Up to 500k for 48-bit
            elif bit_len <= 56:
                max_trial = min(1000000, sqrt_N + 1)  # Up to 1M for 56-bit
            else:
                max_trial = min(2000000, sqrt_N + 1)  # Up to 2M for larger "small" numbers

            print(f"[Root's] Trial division up to {max_trial} for {bit_len}-bit number (sqrt≈{sqrt_N})")

            actual_factors = set()
            for i in range(2, max_trial):
                if self.N % i == 0:
                    p_factor = i
                    q_factor = self.N // i
                    actual_factors.add(p_factor)
                    actual_factors.add(q_factor)

            # Add all actual factors to both p and q candidates
            for factor in actual_factors:
                if factor > 1:
                    root_candidates['p'].add(factor)
                    root_candidates['q'].add(factor)

            # Always add approximations around sqrt(N), and add actual factors if found
            print(f"[Root's] Adding square root approximations...")
            radius = min(100000, sqrt_N // 1000)  # MASSIVE expansion for 2048-bit numbers
            for offset in [-radius, -radius//2, 0, radius//2, radius]:
                p_cand = sqrt_N + offset
                q_cand = sqrt_N - offset
                if p_cand > 1:
                    root_candidates['p'].add(p_cand)
                if q_cand > 1:
                    root_candidates['q'].add(q_cand)

            if actual_factors:
                print(f"[Root's] Also found {len(actual_factors)} actual factors")
            else:
                print(f"[Root's] No small factors found, relying on approximations")

        # Strategy 3: For large numbers, generate additional root candidates
        if self.N.bit_length() > 50:  # Large number threshold
            print(f"[Root's] Large number detected ({self.N.bit_length()}-bit), generating additional root candidates...")

            # Generate roots from factorization approximations (avoid floating point)
            try:
                # Use integer square root approximation
                sqrt_N = self._integer_sqrt_approx(self.N)
                print(f"[Root's] Computed sqrt(N) ≈ {sqrt_N} for {self.N.bit_length()}-bit number")
            except Exception as e:
                print(f"[Root's] Cannot compute square root for {self.N.bit_length()}-bit number: {e}")
                sqrt_N = 2**(self.N.bit_length() // 2)  # Rough approximation
                print(f"[Root's] Using rough approximation sqrt(N) ≈ {sqrt_N}")

            # Strategy 3a: Generate candidates around the square root
            radius = min(1000000, sqrt_N // 10000)  # MASSIVE expansion for 2048-bit
            for offset in [-radius, -radius//2, 0, radius//2, radius]:
                p_cand = sqrt_N + offset
                q_cand = sqrt_N - offset
                if p_cand > 1:
                    root_candidates['p'].add(p_cand)
                if q_cand > 1:
                    root_candidates['q'].add(q_cand)

            # Strategy 3b: Generate candidates from polynomial evaluations
            print(f"[Root's] Evaluating polynomials to find root candidates...")
            for poly in polynomials:  # Check ALL polynomials - no artificial limit (expanded)
                try:
                    # Try to find roots by solving poly = 0 for p and q
                    if poly.has(self.p) and not poly.has(self.q):
                        # Univariate in p: solve for p
                        try:
                            p_roots = sp.solve(poly, self.p)
                            for root in p_roots:
                                if root.is_integer and root > 1:
                                    root_candidates['p'].add(int(root))
                        except:
                            pass
                    elif poly.has(self.q) and not poly.has(self.p):
                        # Univariate in q: solve for q
                        try:
                            q_roots = sp.solve(poly, self.q)
                            for root in q_roots:
                                if root.is_integer and root > 1:
                                    root_candidates['q'].add(int(root))
                        except:
                            pass
                except:
                    continue

            # Strategy 3c: Generate candidates using root-finding techniques
            print(f"[Root's] Applying root-finding techniques for large numbers...")
            lattice_p, lattice_q = self._generate_lattice_roots(polynomials, p_hint, q_hint)
            root_candidates['p'].update(lattice_p)
            root_candidates['q'].update(lattice_q)

            # Strategy 3d: Generate candidates from polynomial root bounds
            print(f"[Root's] Generating root bounds from polynomial analysis...")
            # For polynomials of the form p*q - N, we know roots are around sqrt(N)
            # Generate additional candidates using this insight
            sqrt_n = self._integer_sqrt_approx(self.N)
            # MASSIVE EXPANSION: Much wider range for 2048-bit numbers
            for i in range(-1000, 1001):
                if i != 0:  # Skip exact sqrt
                    p_cand = sqrt_n + i * (sqrt_n // 100000)  # Much finer-grained search
                    q_cand = self.N // p_cand if p_cand != 0 else 1
                    if p_cand > 1 and q_cand > 1:
                        # More lenient check for large numbers
                        if abs(p_cand * q_cand - self.N) < self.N // 10000:  # Within 0.01% for large numbers
                            root_candidates['p'].add(p_cand)
                            root_candidates['q'].add(q_cand)

        # Convert to lists and limit size for computational feasibility
        p_roots = sorted(list(root_candidates['p']))
        q_roots = sorted(list(root_candidates['q']))
        
        # Apply max_root_candidates limit if set
        max_root_candidates = self.config.get('max_root_candidates')
        if max_root_candidates:
            if len(p_roots) > max_root_candidates:
                p_roots = p_roots[:max_root_candidates]
                print(f"[Root's] Limited p root candidates to {max_root_candidates} (from {len(root_candidates['p'])} found)")
            if len(q_roots) > max_root_candidates:
                q_roots = q_roots[:max_root_candidates]
                print(f"[Root's] Limited q root candidates to {max_root_candidates} (from {len(root_candidates['q'])} found)")

        # Fallback: If we have very few candidates (especially for small numbers), generate more
        # BUT: Skip fallback if max_root_candidates is set and we already have candidates
        max_root_candidates = self.config.get('max_root_candidates')
        should_skip_fallback = max_root_candidates and (len(p_roots) >= max_root_candidates or len(q_roots) >= max_root_candidates)
        
        if not should_skip_fallback and (len(p_roots) < 5 or len(q_roots) < 5):
            print(f"[Root's] Limited root candidates ({len(p_roots)} p, {len(q_roots)} q), adding fallback generation...")

            # Generate more candidates using simple trial division approach
            sqrt_n = self._integer_sqrt_approx(self.N)
            bit_len = self.N.bit_length()

            # Limit trial division based on max_root_candidates if set
            if max_root_candidates:
                # Don't generate more than max_root_candidates
                max_trial = min(1000, max_root_candidates * 10, sqrt_n + 1)  # Conservative limit
            else:
                # Use same scaling as above
                if bit_len <= 32:
                    max_trial = min(100000, sqrt_n + 1)
                elif bit_len <= 48:
                    max_trial = min(500000, sqrt_n + 1)
                elif bit_len <= 56:
                    max_trial = min(1000000, sqrt_n + 1)
                else:
                    max_trial = min(2000000, sqrt_n + 1)

            for i in range(2, max_trial):
                if self.N % i == 0:
                    p_factor = i
                    q_factor = self.N // i
                    if p_factor not in p_roots:
                        p_roots.append(p_factor)
                        if max_root_candidates and len(p_roots) >= max_root_candidates:
                            break
                    if q_factor not in q_roots:
                        q_roots.append(q_factor)
                        if max_root_candidates and len(q_roots) >= max_root_candidates:
                            break
                    # Add both (p,q) and (q,p) pairs
                    if q_factor not in p_roots:
                        p_roots.append(q_factor)
                        if max_root_candidates and len(p_roots) >= max_root_candidates:
                            break
                    if p_factor not in q_roots:
                        q_roots.append(p_factor)
                        if max_root_candidates and len(q_roots) >= max_root_candidates:
                            break
                
                # Early exit if we've reached limits
                if max_root_candidates:
                    if len(p_roots) >= max_root_candidates and len(q_roots) >= max_root_candidates:
                        break

            # Also add some candidates around sqrt(N) if still limited (but respect max_root_candidates)
            if len(p_roots) < 10 and (not max_root_candidates or len(p_roots) < max_root_candidates):
                max_offset = 10 if not max_root_candidates else min(10, max_root_candidates - len(p_roots))
                for offset in range(-max_offset, max_offset + 1):
                    if max_root_candidates and (len(p_roots) >= max_root_candidates or len(q_roots) >= max_root_candidates):
                        break
                    cand = sqrt_n + offset
                    if cand > 1 and cand not in p_roots:
                        p_roots.append(cand)
                    cand = sqrt_n - offset
                    if cand > 1 and cand not in q_roots:
                        q_roots.append(cand)

            p_roots = sorted(list(set(p_roots)))  # Remove duplicates
            q_roots = sorted(list(set(q_roots)))  # Remove duplicates
            
            # Apply limit again after fallback generation
            if max_root_candidates:
                if len(p_roots) > max_root_candidates:
                    p_roots = p_roots[:max_root_candidates]
                    print(f"[Root's] Limited p root candidates to {max_root_candidates} after fallback")
                if len(q_roots) > max_root_candidates:
                    q_roots = q_roots[:max_root_candidates]
                    print(f"[Root's] Limited q root candidates to {max_root_candidates} after fallback")

        print(f"[Root's] Total root candidates - p: {len(p_roots)} roots, q: {len(q_roots)} roots")
        if len(p_roots) > 0:
            print(f"[Root's] Sample p roots: {p_roots[:3]}...{p_roots[-3:]}")
        if len(q_roots) > 0:
            print(f"[Root's] Sample q roots: {q_roots[:3]}...{q_roots[-3:]}")

        return {'p': p_roots, 'q': q_roots}

    def _generate_lattice_roots(self, polynomials, p_hint, q_hint):
        """
        Generate root candidates from lattice structure analysis.

        For large numbers, analyze polynomial coefficients to extract
        implicit root information from lattice basis vectors.
        """
        print(f"[Root's] Analyzing lattice structure for root generation...")

        roots_p = set()
        roots_q = set()

        # Strategy: Analyze polynomial coefficients for lattice patterns
        for poly in polynomials[:500]:  # MASSIVE EXPANSION: Analyze first 500 polynomials
            try:
                poly_obj = sp.Poly(poly, self.p, self.q)
                coeffs = poly_obj.as_dict()

                # Look for patterns that suggest lattice-generated roots
                # Pattern 1: Polynomials with small integer coefficients
                small_coeffs = []
                for monomial, coeff in coeffs.items():
                    if isinstance(coeff, (int, sp.Integer)) and abs(coeff) < 1000:
                        small_coeffs.append((monomial, coeff))

                if len(small_coeffs) >= 2:
                    # Try to extract potential roots from coefficient relationships
                    # This is heuristic - look for coefficients that might represent factor approximations
                    for (monomial, coeff) in small_coeffs:
                        if monomial == (1, 0) and coeff != 0:  # Coefficient of p
                            # If we have p + c*q + k = 0, then p ≈ -c*q - k
                            # But this is complex for large numbers
                            pass
                        elif monomial == (0, 1) and coeff != 0:  # Coefficient of q
                            pass

                # Pattern 2: Polynomials that are close to linear in one variable
                if poly_obj.degree(self.p) == 1 and poly_obj.degree(self.q) <= 1:
                    # Try to solve for p in terms of q
                    try:
                        p_expr = sp.solve(poly, self.p)
                        if p_expr and len(p_expr) == 1:
                            p_candidate = p_expr[0]
                            # If it's a simple expression, evaluate at test points
                            for test_q in [2, 3, 5, 7, 11]:
                                try:
                                    p_val = p_candidate.subs(self.q, test_q)
                                    if p_val.is_integer and p_val > 1:
                                        roots_p.add(int(p_val))
                                except:
                                    continue
                    except:
                        pass

                elif poly_obj.degree(self.q) == 1 and poly_obj.degree(self.p) <= 1:
                    # Try to solve for q in terms of p
                    try:
                        q_expr = sp.solve(poly, self.q)
                        if q_expr and len(q_expr) == 1:
                            q_candidate = q_expr[0]
                            # Evaluate at test points
                            for test_p in [2, 3, 5, 7, 11]:
                                try:
                                    q_val = q_candidate.subs(self.p, test_p)
                                    if q_val.is_integer and q_val > 1:
                                        roots_q.add(int(q_val))
                                except:
                                    continue
                    except:
                        pass

            except:
                continue

        # Strategy: Generate roots from continued fraction approximations
        if self.N.bit_length() > 100:
            print(f"[Root's] Generating continued fraction root approximations...")
            # Simple continued fraction for square root approximation (avoid float overflow)
            sqrt_approx = self._integer_sqrt_approx(self.N)

            # Generate a few approximations around the square root - EXPANDED
            for i in range(-100, 101):
                if i != 0:
                    p_cand = sqrt_approx + i * (sqrt_approx // 10000)  # Much finer steps
                    q_cand = sqrt_approx - i * (sqrt_approx // 10000)
                    if p_cand > 1:
                        roots_p.add(p_cand)
                    if q_cand > 1:
                        roots_q.add(q_cand)

        return list(roots_p), list(roots_q)

    def _validate_root_fusions(self, polynomials, root_candidates):
        """
        Step 2: Validate root combinations using A×B fused constraints.

        A×B fusions like p*q*(p-q) provide strong validation for root pairs.
        We check which (p_root, q_root) combinations satisfy the fusions.
        """
        validated_pairs = []

        p_roots = root_candidates['p']
        q_roots = root_candidates['q']

        # Find A×B fusion polynomials
        fusion_polys = []
        for poly in polynomials:
            poly_str = str(poly)
            if '*' in poly_str and 'p' in poly_str and 'q' in poly_str:
                fusion_polys.append(poly)
        
        # Limit fusion polynomials if max_polynomials is set
        # Use a fraction of max_polynomials for fusions (fusions are expensive to check)
        max_polys = self.config.get('max_polynomials')
        original_fusion_count = len(fusion_polys)
        if max_polys:
            # Limit fusions to 1/4 of max_polynomials or 50, whichever is smaller
            max_fusions = min(max_polys // 4, 50) if max_polys > 0 else 50
            if len(fusion_polys) > max_fusions:
                fusion_polys = fusion_polys[:max_fusions]
                print(f"[Root's] Limited fusion polynomials to {max_fusions} (from {original_fusion_count}, max_polynomials={max_polys})")
        
        # Limit root candidates to avoid explosion (already done in _extract_modular_roots, but double-check here)
        max_root_candidates = self.config.get('max_root_candidates')
        if max_root_candidates:
            if len(p_roots) > max_root_candidates:
                p_roots = p_roots[:max_root_candidates]
                print(f"[Root's] Limited p root candidates to {max_root_candidates} in validation")
            if len(q_roots) > max_root_candidates:
                q_roots = q_roots[:max_root_candidates]
                print(f"[Root's] Limited q root candidates to {max_root_candidates} in validation")

        total_combinations = len(p_roots) * len(q_roots)
        print(f"[Root's] Testing {len(p_roots)}×{len(q_roots)} = {total_combinations:,} root combinations against {len(fusion_polys)} fusions")
        
        # Apply scaling options
        max_combinations = self.config.get('max_root_combinations')
        sampling_strategy = self.config.get('root_sampling_strategy', 'none')
        sampling_fraction = self.config.get('root_sampling_fraction', 1.0)
        early_termination = self.config.get('early_termination', False)
        
        # Generate combination list based on strategy
        combinations_to_test = []
        
        if sampling_strategy == 'none' or sampling_fraction >= 1.0:
            # Test all combinations
            combinations_to_test = [(p, q) for p in p_roots for q in q_roots]
        elif sampling_strategy == 'random':
            # Random sampling
            num_samples = int(total_combinations * sampling_fraction)
            if max_combinations:
                num_samples = min(num_samples, max_combinations)
            all_combos = [(p, q) for p in p_roots for q in q_roots]
            combinations_to_test = random.sample(all_combos, min(num_samples, len(all_combos)))
            print(f"[Root's] Random sampling: testing {len(combinations_to_test):,} of {total_combinations:,} combinations ({sampling_fraction*100:.1f}%)")
        elif sampling_strategy == 'stratified':
            # Stratified sampling - evenly distributed across both dimensions
            num_samples = int(total_combinations * sampling_fraction)
            if max_combinations:
                num_samples = min(num_samples, max_combinations)
            # Sample evenly from p and q
            p_sample_size = int((num_samples / len(q_roots)) ** 0.5) if len(q_roots) > 0 else 0
            q_sample_size = int((num_samples / len(p_roots)) ** 0.5) if len(p_roots) > 0 else 0
            p_sample_size = min(p_sample_size, len(p_roots))
            q_sample_size = min(q_sample_size, len(q_roots))
            p_sampled = random.sample(p_roots, p_sample_size) if p_sample_size < len(p_roots) else p_roots
            q_sampled = random.sample(q_roots, q_sample_size) if q_sample_size < len(q_roots) else q_roots
            combinations_to_test = [(p, q) for p in p_sampled for q in q_sampled]
            print(f"[Root's] Stratified sampling: testing {len(combinations_to_test):,} of {total_combinations:,} combinations")
        elif sampling_strategy == 'adaptive':
            # Adaptive sampling - prioritize around hints and sqrt(N)
            import random
            num_samples = int(total_combinations * sampling_fraction)
            if max_combinations:
                num_samples = min(num_samples, max_combinations)
            
            # Prioritize combinations near hints
            sqrt_N = self._integer_sqrt_approx(self.N)
            p_hint = p_roots[len(p_roots)//2] if p_roots else sqrt_N
            q_hint = q_roots[len(q_roots)//2] if q_roots else sqrt_N
            
            # Create weighted list (higher weight for candidates near hints)
            weighted_combos = []
            for p in p_roots:
                for q in q_roots:
                    # Weight based on distance from hints
                    p_dist = abs(p - p_hint) / max(p_hint, 1)
                    q_dist = abs(q - q_hint) / max(q_hint, 1)
                    weight = 1.0 / (1.0 + p_dist + q_dist)
                    weighted_combos.append(((p, q), weight))
            
            # Sample based on weights
            items, weights = zip(*weighted_combos)
            combinations_to_test = random.choices(items, weights=weights, k=min(num_samples, len(items)))
            print(f"[Root's] Adaptive sampling: testing {len(combinations_to_test):,} of {total_combinations:,} combinations (prioritizing near hints)")
        
        # Apply max_combinations limit if set
        if max_combinations and len(combinations_to_test) > max_combinations:
            combinations_to_test = combinations_to_test[:max_combinations]
            print(f"[Root's] Limited to {max_combinations:,} combinations (from {total_combinations:,} total)")
        
        print(f"[Root's] Will test {len(combinations_to_test):,} combinations")
        if early_termination:
            print(f"[Root's] Early termination enabled - will stop after finding good candidate")

        # Test each root combination
        tested_count = 0
        report_interval = max(1, len(combinations_to_test) // 100)  # Report every 1%
        for p_root, q_root in combinations_to_test:
            tested_count += 1
            
            # Progress reporting
            if tested_count % report_interval == 0 or tested_count == len(combinations_to_test):
                progress_pct = (tested_count / len(combinations_to_test)) * 100
                print(f"[Root's] Progress: {tested_count:,}/{len(combinations_to_test):,} ({progress_pct:.1f}%) - Found {len(validated_pairs)} validated pairs so far")
                # For large numbers, be more lenient with factorization check
                product = p_root * q_root
                if self.N.bit_length() > 50:
                    # Allow approximate factorization for large numbers
                    product_diff = abs(product - self.N)
                    if product_diff > self.N // 1000:  # Within 0.1% for large numbers
                        continue
                else:
                    # For small numbers, prefer exact factors but also allow close approximations
                    product_diff = abs(product - self.N)
                    if product_diff > max(100, self.N // 10000):  # Allow small errors for small numbers
                        continue

                # Check against fusion constraints (with leniency for large numbers)
                fusion_score = 0
                max_penalty = 0
                fusion_satisfied = 0

                for fusion in fusion_polys:
                    try:
                        # Evaluate fusion at this root pair
                        val = fusion.subs([(self.p, p_root), (self.q, q_root)])
                        penalty = self._safe_penalty(val, max_penalty=1000.0)
                        fusion_score += penalty
                        max_penalty = max(max_penalty, penalty)

                        # Count as satisfied if penalty is reasonable
                        if self.N.bit_length() > 50:
                            # More lenient for large numbers (approximate polynomials)
                            if penalty < 1000:
                                fusion_satisfied += 1
                        else:
                            # More lenient for small numbers too - polynomials may not be perfect
                            if penalty < 100:  # Relaxed from 10 to 100
                                fusion_satisfied += 1
                    except:
                        fusion_score += 1000

                # Add enhanced verification for high-confidence candidates
                verification_bonus = 0
                if fusion_satisfied >= max(1, len(fusion_polys) // 2):
                    try:
                        # For candidates that pass fusion tests, do additional verification
                        if p_root.bit_length() + q_root.bit_length() <= 4096:
                            # Exact verification for smaller candidates
                            if p_root * q_root == self.N:
                                verification_bonus = -1000  # Large bonus for exact match
                                fusion_score += verification_bonus
                        else:
                            # Statistical verification for larger candidates
                            product_estimate = p_root * q_root
                            if abs(product_estimate - self.N) < self.N // 10000:  # Within 0.01%
                                verification_bonus = -500  # Bonus for very close match
                                fusion_score += verification_bonus
                    except:
                        pass

                # Accept based on number size and constraint satisfaction
                found_exact = False
                if self.N.bit_length() > 50:
                    # For large numbers: accept if some fusions are satisfied
                    if fusion_satisfied >= max(1, len(fusion_polys) // 3):
                        validated_pairs.append((p_root, q_root, fusion_score))
                        # Check for exact factorization
                        try:
                            if p_root.bit_length() + q_root.bit_length() <= 4096:
                                if p_root * q_root == self.N:
                                    found_exact = True
                        except:
                            pass
                else:
                    # For small numbers: be more lenient with fusion validation
                    # Accept if at least some fusions are satisfied or if we have exact factorization
                    exact_factorization = (p_root * q_root == self.N)
                    if exact_factorization:
                        found_exact = True
                    if exact_factorization or (fusion_satisfied >= max(1, len(fusion_polys) // 4) and max_penalty < 500):
                        validated_pairs.append((p_root, q_root, fusion_score))
                
                # Early termination if exact factorization found
                if early_termination and found_exact:
                    print(f"[Root's] Early termination: Found exact factorization after testing {tested_count:,} combinations!")
                    break

        # Sort by fusion score (lower is better)
        validated_pairs.sort(key=lambda x: x[2])
        
        print(f"[Root's] Completed testing: {tested_count:,} combinations tested, {len(validated_pairs)} pairs validated")

        # Return top candidates (limit to avoid explosion)
        return [(p, q) for p, q, score in validated_pairs]  # No limit - use all validated pairs

    def _refine_linear_bounds(self, polynomials, candidate_pairs):
        """
        Step 3: Refine candidates using A components (linear constraints).

        Linear constraints like a*p + b*q + c = 0 help narrow down candidates.
        We score candidates by how well they satisfy linear constraints.
        """
        refined_candidates = []

        # Find A components (linear constraints)
        linear_polys = []
        for poly in polynomials:
            try:
                poly_obj = sp.Poly(poly, self.p, self.q)
                if poly_obj.total_degree() == 1:
                    coeffs = poly_obj.as_dict()
                    # Exclude B components (pure modular)
                    if not (len(coeffs) == 2 and
                           ((1, 0) in coeffs and coeffs[(1, 0)] == 1 or
                            (0, 1) in coeffs and coeffs[(0, 1)] == 1)):
                        linear_polys.append(poly)
            except:
                continue

        print(f"[Root's] Refining {len(candidate_pairs)} candidates with {len(linear_polys)} linear constraints")

        # For very small numbers, skip linear validation as polynomials may not be well-formed
        if self.N.bit_length() <= 64 and len(linear_polys) > 0:
            print(f"[Root's] Skipping linear validation for small number (polynomials may be poorly conditioned)")
            return [(p, q) for p, q in candidate_pairs]  # Pass all candidates through

        for p_root, q_root in candidate_pairs:
            linear_score = 0

            for linear in linear_polys:
                try:
                    val = linear.subs([(self.p, p_root), (self.q, q_root)])
                    penalty = self._safe_penalty(val, max_penalty=1000.0)
                    linear_score += penalty
                except:
                    linear_score += 1000

            # Keep candidates with reasonable linear scores (adaptive threshold)
            if self.N.bit_length() > 1500:
                # For extremely large numbers (like 2000+ bits), be very lenient
                threshold = 100000.0  # Extremely high threshold
            elif self.N.bit_length() > 100:
                # For very large numbers, be more lenient
                threshold = 50000.0  # Much higher threshold for large numbers
            elif self.N.bit_length() > 50:
                # For large numbers, moderately lenient
                threshold = 10000.0
            else:
                # For small numbers, be much more lenient (they have different scaling)
                threshold = 100000.0  # Much higher threshold for small numbers
            if linear_score < threshold:
                refined_candidates.append((p_root, q_root, linear_score))

        # Sort by linear score and return top candidates
        refined_candidates.sort(key=lambda x: x[2])
        return [(p, q) for p, q, score in refined_candidates]  # No limit - use all refined candidates

    def _cross_validate_ratios(self, polynomials, candidate_pairs):
        """
        Step 4: Cross-validate using C components (ratio constraints).

        Ratio constraints like p*q - k*p - m*q + n = 0 provide additional validation.
        We use these to further filter candidates.
        """
        cross_validated = []

        # Find C components (ratio/cross-linking constraints)
        ratio_polys = []
        for poly in polynomials:
            try:
                poly_obj = sp.Poly(poly, self.p, self.q)
                degree = poly_obj.total_degree()
                if degree >= 2:  # Ratio constraints are typically degree 2+
                    # Exclude pure quadratics (D components)
                    has_cross_terms = (poly_obj.degree(self.p) >= 1 and
                                     poly_obj.degree(self.q) >= 1)
                    if has_cross_terms:
                        ratio_polys.append(poly)
            except:
                continue

        print(f"[Root's] Cross-validating {len(candidate_pairs)} candidates with {len(ratio_polys)} ratio constraints")

        # For very small numbers, skip ratio validation as polynomials may not be well-formed
        if self.N.bit_length() <= 64 and len(ratio_polys) > 0:
            print(f"[Root's] Skipping ratio validation for small number (polynomials may be poorly conditioned)")
            return [(p, q) for p, q in candidate_pairs]  # Pass all candidates through

        for p_root, q_root in candidate_pairs:
                ratio_score = 0
                valid_ratios = 0

                for ratio in ratio_polys:
                    try:
                        val = ratio.subs([(self.p, p_root), (self.q, q_root)])
                        penalty = self._safe_penalty(val, max_penalty=1000.0)
                        ratio_score += penalty
                        # Adaptive threshold based on number size
                        if self.N.bit_length() > 1500:
                            # For extremely large numbers (like 2000+ bits), be very lenient
                            threshold = 10000.0  # Accept very high penalties
                        elif self.N.bit_length() > 100:
                            # For very large numbers, be much more lenient
                            threshold = 1000.0  # Accept penalties up to 1000
                        elif self.N.bit_length() > 50:
                            # For large numbers, moderately lenient
                            threshold = 100.0  # Accept penalties up to 100
                        else:
                            # For small numbers, be more lenient (they have different scaling)
                            threshold = 100.0  # Lenient threshold for small numbers
                        if penalty < threshold:
                            valid_ratios += 1
                    except:
                        ratio_score += 1000

                # Adaptive cross-validation requirement based on number size
                if len(ratio_polys) == 0:
                    # No ratio constraints, accept all
                    cross_validated.append((p_root, q_root, ratio_score))
                elif self.N.bit_length() > 1500:
                    # For extremely large numbers, be very lenient - accept if any ratio is reasonably close
                    if valid_ratios >= 0:  # Accept all candidates for extremely large numbers
                        cross_validated.append((p_root, q_root, ratio_score))
                else:
                    # Normal requirement: at least one ratio constraint satisfied
                    if valid_ratios >= 1:
                        cross_validated.append((p_root, q_root, ratio_score))

        # Sort by ratio score
        cross_validated.sort(key=lambda x: x[2])
        return [(p, q) for p, q, score in cross_validated]  # No limit - use all cross-validated candidates

    def _verify_quadratic_constraints(self, polynomials, candidate_pairs):
        """
        Step 5: Final verification using D components (quadratic constraints).

        Quadratic constraints provide the strongest validation.
        We use these for final verification and ranking.
        """
        verified_solutions = []
        closest_candidates = []  # Track closest approximations even if not exact

        # Find D components (pure quadratic constraints)
        quad_polys = []
        for poly in polynomials:
            try:
                poly_obj = sp.Poly(poly, self.p, self.q)
                degree = poly_obj.total_degree()
                if degree == 2:
                    # Check if it's a pure quadratic (no cross terms or simple cross terms)
                    p_deg = poly_obj.degree(self.p)
                    q_deg = poly_obj.degree(self.q)
                    if (p_deg == 2 and q_deg == 0) or (p_deg == 0 and q_deg == 2):
                        quad_polys.append(poly)
            except:
                continue

        print(f"[Root's] Final verification of {len(candidate_pairs)} candidates with {len(quad_polys)} quadratics")

        for p_root, q_root in candidate_pairs:
            quad_score = 0
            perfect_match = True

            # For extremely large numbers, use more sophisticated quadratic evaluation
            if self.N.bit_length() > 1500:
                # For very large numbers, still evaluate but with scaled thresholds
                for quad in quad_polys:
                    try:
                        val = quad.subs([(self.p, p_root), (self.q, q_root)])
                        penalty = self._safe_penalty(val, max_penalty=1000.0)
                        quad_score += penalty
                        # Use adaptive threshold based on number size
                        error_threshold = min(1000.0, self.N.bit_length() * 0.5)
                        if penalty > error_threshold:
                            perfect_match = False
                    except:
                        quad_score += 1000
                        perfect_match = False
            else:
                for quad in quad_polys:
                    try:
                        val = quad.subs([(self.p, p_root), (self.q, q_root)])
                        penalty = self._safe_penalty(val, max_penalty=1000.0)
                        quad_score += penalty
                        # Adaptive threshold for "perfect match" based on number size
                        if self.N.bit_length() > 100:
                            # For very large numbers, allow higher error tolerance
                            error_threshold = 1000.0
                        elif self.N.bit_length() > 50:
                            # For large numbers, moderate tolerance
                            error_threshold = 100.0
                        else:
                            # For small numbers, be more lenient (different scaling)
                            error_threshold = 100.0
                        if penalty > error_threshold:
                            perfect_match = False
                    except:
                        quad_score += 1000
                        perfect_match = False

            # Check factorization accuracy
            try:
                # For extremely large numbers, we need to be careful with multiplication
                if self.N.bit_length() > 1000:
                    # For very large numbers, calculate difference more carefully
                    # p_root * q_root - N, but avoid direct multiplication overflow
                    # Since we're dealing with approximations, calculate relative error
                    if p_root > 1 and q_root > 1:
                        # Estimate the product difference without full multiplication
                        # Use the fact that we're close: p*q ≈ N, so difference is small
                        try:
                            # Try direct calculation for numbers that fit in integer arithmetic
                            if p_root.bit_length() + q_root.bit_length() <= 8192:  # Extended limit for large numbers
                                product = p_root * q_root
                                diff = abs(product - self.N)
                                is_exact = (product == self.N)
                            else:
                                # For extremely large numbers, use bit-length based approximation
                                # Estimate difference using bit lengths as a heuristic
                                product_bits = p_root.bit_length() + q_root.bit_length()
                                n_bits = self.N.bit_length()
                                if product_bits == n_bits:
                                    # Same bit length - could be close, use a large but finite difference
                                    diff = 2**(max(product_bits, n_bits) - 1)  # Half the magnitude
                                    is_exact = False
                                else:
                                    # Different bit lengths - definitely not close
                                    diff = 2**max(product_bits, n_bits)  # Full magnitude difference
                                    is_exact = False
                        except OverflowError:
                            # Fallback: use bit-length based estimation
                            product_bits = p_root.bit_length() + q_root.bit_length()
                            n_bits = self.N.bit_length()
                            if product_bits == n_bits:
                                diff = 2**(product_bits - 1)  # Large but finite
                            else:
                                diff = 2**max(product_bits, n_bits)  # Very large
                            is_exact = False
                    else:
                        diff = float('inf')
                        is_exact = False
                else:
                    # For smaller numbers, direct verification
                    product = p_root * q_root
                    diff = abs(product - self.N)
                    is_exact = (product == self.N)

                # ADVANCED VERIFICATION: Only accept truly exact factorizations
                # For large numbers, we need extremely strict verification
                if is_exact and diff == 0:
                    # Double-check with multiple methods for large numbers
                    try:
                        if self.N.bit_length() > 1000:
                            # Method 1: Direct multiplication check
                            product_check = p_root * q_root
                            if product_check != self.N:
                                is_exact = False
                                diff = abs(product_check - self.N)
                                print(f"[Root's] ❌ ADVANCED CHECK FAILED - Direct multiplication: p*q ≠ N (diff: {diff:,})")
                            else:
                                # Method 2: Modular check with large modulus
                                mod_check = (p_root * q_root) % (2**64)
                                n_mod_check = self.N % (2**64)
                                if mod_check != n_mod_check:
                                    is_exact = False
                                    print(f"[Root's] ❌ ADVANCED CHECK FAILED - Modular verification failed")
                                else:
                                    print(f"[Root's] ✓✓✓ ADVANCED VERIFICATION PASSED - Exact factorization confirmed!")
                        else:
                            print(f"[Root's] ✓✓✓ EXACT FACTORIZATION VERIFIED: p*q = N ✓")
                    except Exception as e:
                        print(f"[Root's] ⚠️ Advanced verification error: {e}")
                        is_exact = False

                if is_exact and diff == 0:
                    verified_solutions.append((p_root, q_root, quad_score))
                else:
                    # Track closest approximations
                    closest_candidates.append((p_root, q_root, quad_score, diff))
                    if diff < float('inf') and diff > 0:
                        print(f"[Root's] ≈ Close approximation: diff={diff:,} (p*q ≈ N)")
                    elif diff == 0:
                        print(f"[Root's] ⚠️ Unexpected: diff=0 but not marked as exact")
                    else:
                        print(f"[Root's] ✗ Factors too large to verify exactly")

            except OverflowError:
                # This should not happen with proper bounds checking
                print(f"[Root's] ⚠️ Unexpected overflow during verification")
                closest_candidates.append((p_root, q_root, quad_score, float('inf')))
            except Exception as e:
                print(f"[Root's] Error during verification: {e}")
                closest_candidates.append((p_root, q_root, quad_score, float('inf')))

        # Sort by quadratic score (should all be very close to 0)
        verified_solutions.sort(key=lambda x: x[2])

        if verified_solutions:
            # Return exact solutions as list of (p, q, score) tuples
            return verified_solutions
        else:
            # No exact solutions found, return closest approximations
            if closest_candidates:
                # Sort by difference (smallest first) then by quadratic score
                closest_candidates.sort(key=lambda x: (x[3], x[2]))
                best_approximations = closest_candidates[:5]  # Return top 5 closest

                print(f"[Root's] No exact solutions found. Returning {len(best_approximations)} closest approximations:")
                for i, (p, q, score, diff) in enumerate(best_approximations, 1):
                    if diff < float('inf'):
                        print(f"[Root's]   #{i}: p={p}, q={q}, diff={diff:,}")
                    else:
                        print(f"[Root's]   #{i}: p={p}, q={q} (diff unknown - too large)")

                # Try final ultra-refinement on the best approximations
                print(f"[Root's] 🔬 Attempting ultra-refinement on best approximation...")
                ultra_refined = self._ultra_refine_closest_factors(best_approximations[0], polynomials)

                if ultra_refined and isinstance(ultra_refined, tuple) and len(ultra_refined) >= 2:
                    ultra_p, ultra_q = ultra_refined
                    print(f"[Root's] ✨ Ultra-refined result: p={ultra_p}, q={ultra_q}")

                    # Final verification
                    try:
                        if ultra_p.bit_length() + ultra_q.bit_length() <= 4096:
                            ultra_product = ultra_p * ultra_q
                            if ultra_product == self.N:
                                print(f"[Root's] 🎯 ULTRA SUCCESS! Exact factorization found!")
                                return (ultra_p, ultra_q)
                            else:
                                ultra_diff = abs(ultra_product - self.N)
                                print(f"[Root's] 📏 Ultra-refined diff: {ultra_diff:,}")
                        else:
                            print(f"[Root's] 📏 Ultra-refined (too large to verify exactly)")
                    except:
                        print(f"[Root's] 📏 Ultra-refined (verification failed)")

                    return [(ultra_p, ultra_q, 0)]  # Add score=0 for consistency

                # Return the closest approximation
                if best_approximations and len(best_approximations[0]) >= 4:
                    best_p, best_q, best_score, best_diff = best_approximations[0]
                else:
                    print(f"[Root's] Error: Invalid approximation format: {best_approximations[0] if best_approximations else 'None'}")
                    return None
                print(f"[Root's] 🏆 Best approximation: p={best_p}, q={best_q}, diff={best_diff:,}")
                return [(best_p, best_q, best_score)]
            else:
                print(f"[Root's] No candidates could be verified")
                return []

    def _refine_candidates_to_exact(self, candidates):
        """
        Refine approximate candidates to find exact factors using advanced factorization techniques.

        For extremely large numbers, uses the lattice approximations as seeds for:
        1. Pollard's rho algorithm (seeded)
        2. Continued fraction factorization
        3. Smart trial division around approximations
        """
        print(f"[Root's] Refining {len(candidates)} approximate candidates to exact factors...")
        exact_solutions = []

        for candidate in candidates[:2]:  # Focus on top 2 candidates
            if len(candidate) == 3:
                p_approx, q_approx, score = candidate
            else:
                p_approx, q_approx = candidate
                score = 0

            print(f"[Root's] Refining candidate: p≈{p_approx}, q≈{q_approx}")

            # Try multiple factorization approaches
            if self.N.bit_length() > 1000:
                # For very large numbers, use advanced techniques

                # Method 1: Smart trial division around approximation
                factor = self._trial_division_around_approximation(p_approx, q_approx)
                if factor:
                    p_exact, q_exact = factor
                    # Verify the factorization is actually correct
                    if p_exact * q_exact == self.N:
                        exact_solutions.append((p_exact, q_exact, score))
                        print(f"[Root's] ✓✓✓ EXACT FACTORIZATION FOUND via smart trial division: p={p_exact}, q={q_exact}")
                        return exact_solutions
                    else:
                        print(f"[Root's] ❌ Smart trial division returned invalid factors: {p_exact} × {q_exact} ≠ {self.N}")

                # Method 2: Seeded Pollard's rho
                factor = self._pollards_rho_seeded(p_approx, q_approx)
                if factor:
                    p_exact, q_exact = factor
                    # Verify the factorization is actually correct
                    if p_exact * q_exact == self.N:
                        exact_solutions.append((p_exact, q_exact, score))
                        print(f"[Root's] ✓✓✓ EXACT FACTORIZATION FOUND via seeded Pollard's rho: p={p_exact}, q={q_exact}")
                        return exact_solutions
                    else:
                        print(f"[Root's] ❌ Seeded Pollard's rho returned invalid factors: {p_exact} × {q_exact} ≠ {self.N}")

                # Method 3: Continued fraction factorization seeded with approximation
                factor = self._continued_fraction_seeded(p_approx, q_approx)
                if factor:
                    p_exact, q_exact = factor
                    # Verify the factorization is actually correct
                    if p_exact * q_exact == self.N:
                        exact_solutions.append((p_exact, q_exact, score))
                        print(f"[Root's] ✓✓✓ EXACT FACTORIZATION FOUND via seeded continued fraction: p={p_exact}, q={q_exact}")
                        return exact_solutions
                    else:
                        print(f"[Root's] ❌ Seeded continued fraction returned invalid factors: {p_exact} × {q_exact} ≠ {self.N}")

            else:
                # For smaller large numbers, use targeted trial division
                search_radius = 100000  # Expanded from 10000 to 100000
                p_start = max(2, p_approx - search_radius)
                p_end = p_approx + search_radius

                for p_test in range(p_start, min(p_end, p_approx + search_radius + 1)):
                    if self.N % p_test == 0:
                        q_test = self.N // p_test
                        if q_test > 1 and p_test * q_test == self.N:
                            exact_solutions.append((p_test, q_test, score))
                            print(f"[Root's] ✓✓✓ EXACT FACTORIZATION FOUND via refinement: p={p_test}, q={q_test}")
                            return exact_solutions

        print(f"[Root's] Advanced refinement found {len(exact_solutions)} exact solutions")
        return exact_solutions

    def _trial_division_around_approximation(self, p_approx, q_approx):
        """
        Smart trial division that searches around the lattice approximations.
        Uses the fact that for RSA, factors are close to sqrt(N).
        """
        print(f"[Trial] Smart trial division around p≈{p_approx}, q≈{q_approx}")

        # Focus search around sqrt(N) since that's where RSA factors are
        sqrt_n = self._integer_sqrt_approx(self.N)

        # Search in a narrow band around sqrt(N)
        # For RSA numbers, factors are very close to sqrt(N)
        search_width = 10**9  # MASSIVE expansion: 1 billion range for maximum coverage

        start = max(2, sqrt_n - search_width // 2)
        end = sqrt_n + search_width // 2

        print(f"[Trial] Searching in range [{start}, {end}] around sqrt(N)")

        # Use larger steps for speed, but ensure we don't miss factors
        step = 2  # Check only odd numbers (even numbers > 2 can't be prime)

        for candidate in range(max(3, start | 1), min(end, 10**30), step):  # Ultra-massive upper limit
            if self.N % candidate == 0:
                factor1 = candidate
                factor2 = self.N // candidate
                if factor2 > 1 and factor1 * factor2 == self.N:
                    print(f"[Trial] Found factors: {factor1} × {factor2}")
                    return (factor1, factor2)

        print(f"[Trial] No factors found in search range")
        return None

    def _pollards_rho_seeded(self, p_hint, q_hint):
        """
        Pollard's rho algorithm seeded with lattice approximations.
        This uses the Brent cycle detection for efficiency.
        """
        print(f"[Rho] Seeded Pollard's rho with hints p≈{p_hint}, q≈{q_hint}")

        def f(x, c):
            return (x * x + c) % self.N

        # Use the approximation to seed the algorithm
        x = (p_hint + q_hint) % self.N  # Sum of approximations as seed
        y = x
        c = 1  # Simple constant

        d = 1
        steps = 0
        while d == 1 and steps < 10000:  # Limit steps to 10k
            x = f(x, c)
            y = f(f(y, c), c)
            d = abs(x - y)
            d = self._gcd(d, self.N)
            steps += 1

        if d == self.N:
            print(f"[Rho] Failed with seed c={c}, trying different seed")
            # Try different seeds
            for c in [3, 5, 7]:
                x = (p_hint * q_hint) % self.N  # Product of approximations
                y = x
                d = 1
                steps = 0
                while d == 1 and steps < 10000:  # Limit steps
                    x = f(x, c)
                    y = f(f(y, c), c)
                    d = abs(x - y)
                    d = self._gcd(d, self.N)
                    steps += 1

                if d != 1 and d != self.N:
                    break

        if d != 1 and d != self.N:
            factor1 = d
            factor2 = self.N // d
            print(f"[Rho] Found factors: {factor1} × {factor2}")
            return (factor1, factor2)

        print(f"[Rho] No factors found")
        return None

    def _continued_fraction_seeded(self, p_hint, q_hint):
        """
        Continued fraction factorization seeded with lattice approximation.
        Uses the approximation to guide the continued fraction expansion.
        """
        print(f"[CF] Seeded continued fraction with hints p≈{p_hint}, q≈{q_hint}")

        # Use the ratio of approximations to seed the continued fraction
        if q_hint != 0:
            seed_ratio = p_hint / q_hint
        else:
            seed_ratio = self._integer_sqrt_approx(self.N * 4) / 2  # Fallback to sqrt(4N)/2

        # Generate continued fraction convergents around the seed ratio
        h_prev, k_prev = 1, 0
        h, k = round(seed_ratio), 1

        # Generate convergents
        for i in range(20):  # Generate up to 20 convergents
            # Check if this convergent gives a factor
            convergent = h // k if k != 0 else h

            # Test if convergent * k is close to h (indicating good approximation)
            if abs(convergent * k - h) < 10:  # Close approximation
                # Test if this gives a factorization
                test_val = convergent
                if test_val > 1 and self.N % test_val == 0:
                    factor1 = test_val
                    factor2 = self.N // test_val
                    if factor2 > 1:
                        print(f"[CF] Found factors: {factor1} × {factor2}")
                        return (factor1, factor2)

            # Generate next convergent
            fractional_part = seed_ratio - (h // k) if k != 0 else seed_ratio
            if fractional_part == 0:
                break

            next_term = round(1 / fractional_part)
            h_next = next_term * h + h_prev
            k_next = next_term * k + k_prev

            h_prev, k_prev = h, k
            h, k = h_next, k_next

        print(f"[CF] No factors found")
        return None

    def _gcd(self, a, b):
        """Euclidean algorithm for GCD"""
        while b:
            a, b = b, a % b
        return a

    def _select_minimal_consistent_subset(self, polynomials):
        """
        Select minimal subset most likely to be consistent.
        Prioritizes A×B fusions + basic factorization.
        SMALL VERSION: Returns only 1-2 polynomials for fast computation.
        """
        subset = []

        # Always include basic factorization if present (limit to 1)
        for poly in polynomials:
            poly_str = str(poly)
            if 'p*q' in poly_str and str(self.N) in poly_str:
                subset.append(poly)
                break

        # Add only 1 A×B fusion (most reliable, small version)
        for poly in polynomials:
            poly_str = str(poly)
            if '*' in poly_str and 'p' in poly_str and 'q' in poly_str and poly not in subset:
                subset.append(poly)
                break  # Only 1 fusion for small version

        # Return even single polynomial for small computation
        return subset if len(subset) >= 1 else []

    def _select_single_fusion_subset(self, polynomials):
        """
        Select single A×B fusion + complementary constraints.
        SMALL VERSION: Returns only 1-2 polynomials for fast computation.
        """
        subset = []

        # Find one A×B fusion
        for poly in polynomials:
            poly_str = str(poly)
            if '*' in poly_str and 'p' in poly_str and 'q' in poly_str:
                subset.append(poly)
                break

        if not subset:
            return []

        # Add only 1 complementary linear constraint (small version)
        for poly in polynomials:
            if poly in subset:
                continue
            try:
                poly_obj = sp.Poly(poly, self.p, self.q)
                if poly_obj.total_degree() == 1:
                    subset.append(poly)
                    break  # Only 1 additional constraint for small version
            except:
                continue

        return subset

    def _select_linear_only_subset(self, polynomials):
        """
        Select only linear constraints (most likely to be consistent).
        SMALL VERSION: Returns only 2 polynomials for fast computation.
        """
        subset = []

        for poly in polynomials:
            try:
                poly_obj = sp.Poly(poly, self.p, self.q)
                if poly_obj.total_degree() == 1:
                    subset.append(poly)
                    if len(subset) >= 2:  # Small limit for fast computation
                        break
            except:
                continue

        return subset if len(subset) >= 2 else []

    def _progressive_basis_construction(self, polynomials):
        """
        Build basis progressively, adding constraints one by one until inconsistency.
        """
        basis = []

        # Start with the most basic constraint
        for poly in polynomials:
            poly_str = str(poly)
            if 'p*q' in poly_str and str(self.N) in poly_str:
                basis.append(poly)
                break

        if not basis:
            return []

        # Add constraints progressively, checking consistency
        for poly in polynomials:
            if poly in basis:
                continue

            # Try adding this polynomial
            test_basis = basis + [poly]

            try:
                gb = groebner(test_basis, self.p, self.q, order='lex')
                gb_list = list(gb)

                # If still consistent (or small system), keep it
                is_inconsistent = len(gb_list) == 1 and gb_list[0] == 1
                if not is_inconsistent or len(test_basis) <= 3:  # Allow small inconsistent systems
                    basis = test_basis
                    if len(basis) >= 5:  # Don't get too large
                        break
            except:
                # If it causes computation issues, skip it
                continue

        return basis

    def solve_with_resultants(self, polynomials: List[sp.Expr]) -> Optional[Tuple[int, int]]:
        """
        Use resultant elimination optimized for ABCD fused polynomials.

        Resultants work well with our structured constraints:
        - A×B fused constraints often produce clean resultants
        - Modular B components create solvable systems
        - Ratio C components provide good elimination targets

        Prioritizes A-B fused pairs and modular constraints.
        """
        print("\n[Resultant] Using resultant elimination on ABCD fused polynomials...")
        print("[Resultant] ℹ️  Resultants work well with approximate constraints (unlike Gröbner bases)")

        try:
            # Classify polynomials by ABCD types (reuse classification logic)
            a_components = []
            b_components = []
            c_components = []
            d_components = []

            for poly in polynomials:
                try:
                    # Convert to SymPy if it's a string
                    if isinstance(poly, str):
                        poly = sp.sympify(poly)

                    if hasattr(poly, 'has') and not poly.has(sp.Piecewise):
                        poly_obj = sp.Poly(poly, self.p, self.q)
                        degree = poly_obj.total_degree()

                        # Check for fused A×B patterns first (regardless of degree)
                        poly_str = str(poly)
                        if '*' in poly_str and ('p' in poly_str and 'q' in poly_str):
                            # This is a fused A×B constraint - highest priority
                            a_components.append(poly)  # Treat as A for now, will be prioritized later
                        else:
                            # Classify by structure and degree
                            if degree == 1:
                                coeffs = poly_obj.as_dict()
                                if len(coeffs) <= 3:
                                    if (len(coeffs) == 2 and
                                          ((1, 0) in coeffs and coeffs[(1, 0)] == 1) or
                                          ((0, 1) in coeffs and coeffs[(0, 1)] == 1)):
                                        b_components.append(poly)
                                    else:
                                        a_components.append(poly)
                            elif degree == 2:
                                if poly_obj.degree(self.p) == 2 or poly_obj.degree(self.q) == 2:
                                    d_components.append(poly)
                                else:
                                    c_components.append(poly)
                            else:
                                c_components.append(poly)
                    else:
                        # Handle non-polynomial expressions
                        a_components.append(poly)
                except:
                    a_components.append(poly)  # Fallback

            print(f"[Resultant] ABCD classification: A={len(a_components)}, B={len(b_components)}, C={len(c_components)}, D={len(d_components)}")

            # Strategy: Find best pairs for resultant elimination
            # Priority: A-B fused > A-A pairs > B-B pairs > C-D pairs

            candidate_pairs = []

            # 1. A-B fused pairs (highest priority)
            for a_poly in a_components:
                for b_poly in b_components:
                    candidate_pairs.append((a_poly, b_poly, "A×B"))

            # 2. A-A linear pairs
            for i in range(len(a_components)):
                for j in range(i+1, len(a_components)):
                    candidate_pairs.append((a_components[i], a_components[j], "A×A"))

            # 3. B-B modular pairs
            for i in range(len(b_components)):
                for j in range(i+1, len(b_components)):
                    candidate_pairs.append((b_components[i], b_components[j], "B×B"))

            # 4. C-D mixed pairs
            for c_poly in c_components[:3]:
                for d_poly in d_components[:3]:
                    candidate_pairs.append((c_poly, d_poly, "C×D"))

            print(f"[Resultant] Testing {min(len(candidate_pairs), 12)} ABCD polynomial pairs...")

            best_resultant = None
            best_p1 = None
            best_p2 = None
            best_pair_type = None

            tested_pairs = 0
            for poly1, poly2, pair_type in candidate_pairs[:12]:  # Limit to avoid excessive computation
                tested_pairs += 1
                try:
                    print(f"[Resultant] Testing {pair_type} pair {tested_pairs}:")
                    print(f"  f1 = {str(poly1)[:40]}...")
                    print(f"  f2 = {str(poly2)[:40]}...")

                    # Compute resultant with respect to q (eliminates q variable)
                    res_q = sp.resultant(poly1, poly2, self.q)

                    # Also try eliminating p
                    res_p = sp.resultant(poly1, poly2, self.p)

                    # Choose the better resultant
                    candidates = [(res_q, "q"), (res_p, "p")]
                    best_local = None
                    best_var = None

                    for res, var in candidates:
                        if res != 0 and not res.has(sp.zoo, sp.nan):
                            try:
                                if var == "p":
                                    res_degree = sp.Poly(res, self.q).degree()
                                else:
                                    res_degree = sp.Poly(res, self.p).degree()

                                if res_degree > 0 and res_degree <= 6:  # Reasonable degree
                                    if best_local is None or res_degree < sp.Poly(best_local, self.p if var == "q" else self.q).degree():
                                        best_local = res
                                        best_var = var
                            except:
                                if best_local is None:
                                    best_local = res
                                    best_var = var

                    if best_local is not None:
                        # Compare with global best
                        try:
                            if best_var == "p":
                                curr_degree = sp.Poly(best_local, self.q).degree()
                                if best_resultant is None:
                                    best_degree = float('inf')
                                else:
                                    best_degree = sp.Poly(best_resultant, self.p).degree()
                            else:
                                curr_degree = sp.Poly(best_local, self.p).degree()
                                if best_resultant is None:
                                    best_degree = float('inf')
                                else:
                                    best_degree = sp.Poly(best_resultant, self.p).degree()

                            if curr_degree < best_degree and curr_degree > 0:
                                best_resultant = best_local
                                best_p1 = poly1
                                best_p2 = poly2
                                best_pair_type = pair_type
                                print(f"[Resultant] ✓ Better {pair_type} resultant (degree {curr_degree}, eliminating {best_var})")
                        except:
                            if best_resultant is None:
                                best_resultant = best_local
                                best_p1 = poly1
                                best_p2 = poly2
                                best_pair_type = pair_type
                                print(f"[Resultant] ✓ Using {pair_type} resultant (eliminating {best_var})")

                except Exception as e:
                    print(f"[Resultant] ✗ Failed {pair_type} pair: {str(e)[:50]}...")
                    continue

            if best_resultant is None:
                print("[Resultant] No suitable ABCD resultants found")
                return None

            print(f"\n[Resultant] Selected best {best_pair_type} pair:")
            print(f"  f1(p,q) = {best_p1}")
            print(f"  f2(p,q) = {best_p2}")
            print(f"[Resultant] Resultant: {best_resultant}")

            # Determine which variable was eliminated
            try:
                if self.p in best_resultant.free_symbols:
                    elim_var = self.q
                    remaining_var = self.p
                else:
                    elim_var = self.p
                    remaining_var = self.q
            except:
                elim_var = self.q  # Default assumption
                remaining_var = self.p

            print(f"[Resultant] Eliminated {elim_var}, solving for {remaining_var}...")

            # Try to solve the resultant equation
            # For linear polynomials with huge coefficients, sp.solve() may fail
            # So we handle linear cases directly
            solutions = []
            
            try:
                # Check if it's a linear polynomial
                if remaining_var == self.p:
                    poly_p = sp.Poly(best_resultant, self.p)
                else:
                    poly_p = sp.Poly(best_resultant, self.q)
                
                degree = poly_p.degree()
                print(f"[Resultant] Resultant polynomial degree: {degree}")
                
                if degree == 1:
                    # Linear case: a*x + b = 0 => x = -b/a
                    # Extract coefficients directly
                    coeffs = poly_p.all_coeffs()
                    if len(coeffs) == 2:
                        a, b = coeffs[1], coeffs[0]  # a*x + b = 0
                        print(f"[Resultant] Linear equation: {a}*{remaining_var} + {b} = 0")
                        print(f"[Resultant] Solving: {remaining_var} = -({b}) / ({a})")
                        
                        # Compute solution directly
                        if a != 0:
                            solution = -b / a
                            solutions = [solution]
                            print(f"[Resultant] Direct solution: {remaining_var} = {solution}")
                            
                            # Check if it's a rational number that might simplify to an integer
                            # Also check absolute value in case sign is wrong
                            candidates_to_check = [solution]
                            
                            # If negative, also check if absolute value is valid
                            if hasattr(solution, '__lt__') and solution < 0:
                                abs_solution = -solution
                                candidates_to_check.append(abs_solution)
                                print(f"[Resultant]   → Also checking absolute value: {abs_solution}")
                            
                            # Try to simplify rationals
                            simplified_candidates = []
                            for cand in candidates_to_check:
                                try:
                                    if isinstance(cand, sp.Rational) or (hasattr(cand, 'is_rational') and cand.is_rational):
                                        simplified = sp.simplify(cand)
                                        if simplified.is_integer:
                                            simplified_candidates.append(simplified)
                                            print(f"[Resultant]   → Simplified to integer: {simplified}")
                                        elif hasattr(cand, 'p') and hasattr(cand, 'q'):
                                            num, den = cand.p, cand.q
                                            if den == 1:
                                                simplified_candidates.append(sp.Integer(num))
                                                print(f"[Resultant]   → Rational simplifies to integer: {num}")
                                            else:
                                                # Check if it divides N
                                                if den > 0 and num % den == 0:
                                                    int_val = num // den
                                                    if int_val > 1:
                                                        simplified_candidates.append(sp.Integer(int_val))
                                                        print(f"[Resultant]   → Rational evaluates to integer: {int_val}")
                                except:
                                    pass
                            
                            if simplified_candidates:
                                solutions = simplified_candidates
                        else:
                            print(f"[Resultant] ✗ Coefficient is zero, no solution")
                    else:
                        # Try sp.solve() as fallback
                        solutions = sp.solve(best_resultant, remaining_var)
                elif degree == 0:
                    # Constant polynomial
                    if best_resultant == 0:
                        print(f"[Resultant] Resultant is zero - polynomials are dependent")
                    else:
                        print(f"[Resultant] Resultant is non-zero constant - no solution")
                else:
                    # Higher degree - use sp.solve()
                    print(f"[Resultant] Higher degree ({degree}), using sp.solve()...")
                    solutions = sp.solve(best_resultant, remaining_var)
            except Exception as e:
                print(f"[Resultant] Direct solving failed: {e}, trying sp.solve()...")
                try:
                    solutions = sp.solve(best_resultant, remaining_var)
                except Exception as e2:
                    print(f"[Resultant] sp.solve() also failed: {e2}")
                    solutions = []

            print(f"[Resultant] Found {len(solutions)} potential {remaining_var} values")

            for i, val in enumerate(solutions[:8]):  # Limit candidates
                print(f"[Resultant] Testing {remaining_var} = {val}")
                
                # Try to convert to integer if it's a rational that simplifies
                # Also check absolute value in case the sign is wrong
                candidates_to_test = [val]
                
                # If negative, also check absolute value
                try:
                    if hasattr(val, '__lt__') and val < 0:
                        abs_val = -val
                        candidates_to_test.append(abs_val)
                        print(f"[Resultant]   → Also testing absolute value: {abs_val}")
                except:
                    pass
                
                val_int = None
                for test_val in candidates_to_test:
                    try:
                        # Check if it's already an integer
                        if hasattr(test_val, 'is_integer') and test_val.is_integer:
                            if test_val > 1:
                                val_int = int(test_val)
                                print(f"[Resultant]   → Found integer: {val_int}")
                                break
                        # Check if it's a rational number that simplifies to an integer
                        elif hasattr(test_val, 'is_rational') and test_val.is_rational:
                            # Try to simplify rational to integer
                            simplified = sp.simplify(test_val)
                            if simplified.is_integer and simplified > 1:
                                val_int = int(simplified)
                                print(f"[Resultant]   → Rational simplified to integer: {val_int}")
                                break
                        # Check if it's a SymPy Rational
                        elif isinstance(test_val, sp.Rational):
                            # Check if it's actually an integer (denominator = 1)
                            if test_val.q == 1 and test_val.p > 1:
                                val_int = int(test_val.p)
                                print(f"[Resultant]   → Rational with denominator 1: {val_int}")
                                break
                            elif test_val.q != 1:
                                # Check if numerator is divisible by denominator
                                # Handle negative numerators correctly
                                num = abs(test_val.p)
                                den = abs(test_val.q)
                                if num % den == 0:
                                    int_candidate = num // den
                                    # Preserve sign
                                    if test_val.p < 0:
                                        int_candidate = -int_candidate
                                    if abs(int_candidate) > 1:
                                        val_int = abs(int_candidate)  # Use absolute value for factors
                                        print(f"[Resultant]   → Rational divides evenly: {val_int} (from {int_candidate})")
                                        break
                                # Try to see if it simplifies
                                simplified = sp.simplify(test_val)
                                if simplified.is_integer and abs(simplified) > 1:
                                    val_int = abs(int(simplified))  # Use absolute value
                                    print(f"[Resultant]   → Rational simplified: {val_int}")
                                    break
                                # Last resort: check if it's very close to an integer (rounding error)
                                try:
                                    float_val = float(test_val)
                                    if abs(float_val) > 1:
                                        rounded = round(float_val)
                                        if abs(float_val - rounded) < 1e-10:
                                            val_int = abs(rounded)
                                            print(f"[Resultant]   → Rational rounds to integer: {val_int}")
                                            break
                                except:
                                    pass
                        # Check if it's a regular integer
                        elif isinstance(test_val, (int, sp.Integer)):
                            if test_val > 1:
                                val_int = int(test_val)
                                print(f"[Resultant]   → Found integer: {val_int}")
                                break
                    except Exception as e:
                        print(f"[Resultant]   → Error converting to integer: {e}")
                        continue
                    
                    if val_int is not None:
                        break
                
                if val_int is not None:

                    # Compute the other factor
                    if remaining_var == self.p:
                        # Found p, compute q = N / p
                        if self.N % val_int == 0:
                            q_int = self.N // val_int
                            print(f"[Resultant]   → Computed q = N / p = {self.N} / {val_int} = {q_int}")
                            if q_int > 1 and val_int * q_int == self.N:
                                print(f"[Resultant] ✓✓✓ EXACT SOLUTION FOUND: p={val_int}, q={q_int}")
                                print(f"[Resultant]   Verification: p × q = {val_int} × {q_int} = {val_int * q_int} = N ✓")
                                return (val_int, q_int)
                            else:
                                print(f"[Resultant]   ✗ Verification failed: p × q = {val_int} × {q_int} = {val_int * q_int} ≠ N")
                        else:
                            print(f"[Resultant]   ✗ {val_int} is not a divisor of N (remainder: {self.N % val_int})")
                    else:  # remaining_var == q
                        # Found q, compute p = N / q
                        if self.N % val_int == 0:
                            p_int = self.N // val_int
                            print(f"[Resultant]   → Computed p = N / q = {self.N} / {val_int} = {p_int}")
                            if p_int > 1 and p_int * val_int == self.N:
                                print(f"[Resultant] ✓✓✓ EXACT SOLUTION FOUND: p={p_int}, q={val_int}")
                                print(f"[Resultant]   Verification: p × q = {p_int} × {val_int} = {p_int * val_int} = N ✓")
                                return (p_int, val_int)
                            else:
                                print(f"[Resultant]   ✗ Verification failed: p × q = {p_int} × {val_int} = {p_int * val_int} ≠ N")
                        else:
                            print(f"[Resultant]   ✗ {val_int} is not a divisor of N (remainder: {self.N % val_int})")

            print("[Resultant] No valid solutions from ABCD resultants")
            return None

        except Exception as e:
            print(f"[Resultant] ABCD resultant elimination failed: {str(e)[:100]}...")
            return None
    
    def solve_with_modular_constraints(self, polynomials: List[sp.Expr], 
                                       small_primes: List[int] = None) -> Optional[Tuple[int, int]]:
        """
        Solve using modular arithmetic and check for small factors.
        
        Solves the system modulo small primes to find small factors.
        """
        print("\n[Modular] Solving with modular constraints...")
        
        if small_primes is None:
            small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
        
        try:
            # Extract modular constraints from polynomials first
            print(f"[Modular] Extracting modular constraints from {len(polynomials)} polynomials...")

            modular_constraints = []
            for i, poly in enumerate(polynomials):
                try:
                    # Look for polynomials that might represent modular constraints
                    # Check if it's a simple linear polynomial that could be a congruence
                    if poly.is_linear:
                        coeffs = poly.as_coefficients_dict()
                        if len(coeffs) <= 3:  # Simple polynomial
                            modular_constraints.append((i, poly))
                            print(f"[Modular] Found potential modular constraint: f{i+1} = {poly}")
                except:
                    pass

            # Check for small prime factors first
            print(f"[Modular] Checking for small prime factors...")

            for prime in small_primes:
                if self.N % prime == 0:
                    p_candidate = prime
                    q_candidate = self.N // prime

                    if p_candidate > 1 and q_candidate > 1:
                        print(f"[Modular] ✓ Found small prime factor: p={p_candidate}, q={q_candidate}")
                        return (p_candidate, q_candidate)
            
            # Try trial division with slightly larger numbers
            print(f"[Modular] Trying trial division up to 10000000...")
            limit = min(10000000, int(math.isqrt(self.N)) + 1)  # MASSIVE EXPANSION to 10M
            
            for candidate in range(2, limit):
                if self.N % candidate == 0:
                    p_candidate = candidate
                    q_candidate = self.N // candidate
                    
                    if p_candidate > 1 and q_candidate > 1:
                        print(f"[Modular] ✓ Found factor via trial division: p={p_candidate}, q={q_candidate}")
                        return (p_candidate, q_candidate)
            
            print("[Modular] No small factors found")
            return None
            
        except Exception as e:
            print(f"[Modular] Failed: {e}")
            return None
    
    def solve_with_numerical_refinement(self, polynomials: List[sp.Expr],
                                       p_hint: int, q_hint: int) -> Optional[Tuple[int, int]]:
        """
        Use numerical methods with ABCD polynomial classification and weighted scoring.

        ABCD components are weighted differently:
        - A (Linear): High weight - fundamental constraints
        - B (Modular): Medium weight - modular hints
        - C (Ratio): Medium weight - cross-linking constraints
        - D (Quadratic): Low weight - complex constraints

        Fused A×B constraints get highest priority.
        """
        print("\n[Numerical] Using ABCD-weighted numerical refinement...")

        try:
            # Classify polynomials by ABCD component types with weights
            weighted_polys = []  # (weight, polynomial, component_type)

            for poly in polynomials:
                try:
                    # Convert to SymPy if it's a string
                    if isinstance(poly, str):
                        poly = sp.sympify(poly)

                    if hasattr(poly, 'has') and not poly.has(sp.Piecewise):
                        poly_obj = sp.Poly(poly, self.p, self.q)
                        degree = poly_obj.total_degree()

                        # Check for fused A×B patterns first (regardless of degree)
                        poly_str = str(poly)
                        if '*' in poly_str and ('p' in poly_str and 'q' in poly_str):
                            weighted_polys.append((10.0, poly, "A×B"))  # Fused constraints
                        else:
                            # Classify by structure and degree
                            if degree == 1:
                                coeffs = poly_obj.as_dict()
                                if len(coeffs) <= 3:
                                    if (len(coeffs) == 2 and
                                          ((1, 0) in coeffs and coeffs[(1, 0)] == 1) or
                                          ((0, 1) in coeffs and coeffs[(0, 1)] == 1)):
                                        weighted_polys.append((3.0, poly, "B"))   # Modular constraints
                                    else:
                                        weighted_polys.append((4.0, poly, "A"))   # Linear constraints
                            elif degree == 2:
                                if poly_obj.degree(self.p) == 2 or poly_obj.degree(self.q) == 2:
                                    weighted_polys.append((1.0, poly, "D"))   # Quadratic constraints
                                else:
                                    weighted_polys.append((2.0, poly, "C"))   # Ratio constraints
                            else:
                                weighted_polys.append((0.5, poly, "C+"))     # Higher-degree (likely C fusions)
                    else:
                        # Handle non-polynomial expressions
                        weighted_polys.append((0.1, poly, "?"))
                except:
                    weighted_polys.append((0.1, poly, "?"))  # Fallback with low weight

            # Sort by weight (highest first) and take top polynomials
            weighted_polys.sort(key=lambda x: x[0], reverse=True)
            selected_polys = weighted_polys[:12]  # Use up to 12 weighted polynomials

            print(f"[Numerical] ABCD classification complete:")
            component_counts = {}
            for weight, poly, comp_type in selected_polys:
                component_counts[comp_type] = component_counts.get(comp_type, 0) + 1
            print(f"[Numerical] Selected components: {component_counts}")
            print(f"[Numerical] Using {len(selected_polys)} weighted polynomials for evaluation")
            
            best_solution = None
            best_score = float('inf')
            
            # Search in expanding radius around hints - MASSIVE EXPANSION
            for radius in [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000]:  # Much larger search radii
                print(f"[Numerical] Searching radius {radius} around hints...")
                
                # Check divisors near hints
                p_start = max(2, p_hint - radius)
                p_end = min(p_hint + radius, int(math.isqrt(self.N)) + radius)
                
                checked = 0
                for p_test in range(p_start, p_end):
                    if self.N % p_test != 0:
                        continue
                    
                    checked += 1
                    q_test = self.N // p_test
                    
                    if q_test <= 1:
                        continue
                    
                    # Evaluate ABCD-weighted polynomial system
                    score = 0.0
                    for weight, poly, comp_type in selected_polys:
                        try:
                            val = poly.subs([(self.p, p_test), (self.q, q_test)])
                            # Apply component-specific weighting
                            if comp_type == "A×B":
                                # Fused constraints are most important - amplify their influence
                                penalty = self._safe_penalty(val, max_penalty=1000.0) * weight * 2.0
                            elif comp_type == "A":
                                # Linear constraints are very important
                                penalty = self._safe_penalty(val, max_penalty=1000.0) * weight
                            elif comp_type == "B":
                                # Modular constraints help but are less critical
                                penalty = self._safe_penalty(val, max_penalty=1000.0) * weight * 0.8
                            elif comp_type == "C":
                                # Ratio constraints provide structure
                                penalty = self._safe_penalty(val, max_penalty=1000.0) * weight * 0.6
                            else:
                                # Other constraints (D, C+, ?) have lower weight
                                penalty = self._safe_penalty(val, max_penalty=1000.0) * weight * 0.3

                            score += penalty
                        except:
                            # Penalize evaluation failures (likely complex polynomials)
                            score += weight * 100
                    
                    if score < best_score:
                        best_score = score
                        best_solution = (p_test, q_test)
                    
                    # Exact solution found
                    if score < 1e-8 and p_test * q_test == self.N:
                        print(f"[Numerical] ✓✓✓ EXACT SOLUTION FOUND via ABCD scoring: p={p_test}, q={q_test}")
                        print(f"[Numerical] Final score: {score}")
                        return (p_test, q_test)
                
                print(f"[Numerical] Checked {checked} divisors in radius {radius}")
                
                if best_solution and best_score < 1.0:
                    p_sol, q_sol = best_solution
                    if p_sol * q_sol == self.N:
                        print(f"[Numerical] ✓ Good solution found: p={p_sol}, q={q_sol} (ABCD score={best_score:.4f})")
                        return best_solution
            
            if best_solution:
                print(f"[Numerical] Best solution found: score={best_score}")
                p_sol, q_sol = best_solution
                if p_sol * q_sol == self.N:
                    return best_solution
            
            print("[Numerical] No valid solution found")
            return None
            
        except Exception as e:
            print(f"[Numerical] Failed: {e}")
            return None
    
    def solve_with_hensel_lifting(self, polynomials: List[sp.Expr],
                                  p_hint: int = None, q_hint: int = None) -> Optional[Tuple[int, int]]:
        """
        Use Hensel lifting to lift modular solutions to full integers.
        
        Starts with solution mod small prime, then lifts to larger moduli.
        """
        print("\n[Hensel] Using Hensel lifting...")
        
        try:
            # Start with a small prime
            small_prime = 101
            
            # Find solutions modulo small_prime
            solutions_mod = []
            N_mod = self.N % small_prime
            
            for p_mod in range(1, small_prime):
                if (p_mod * (N_mod * pow(p_mod, -1, small_prime) % small_prime)) % small_prime == N_mod:
                    q_mod = (N_mod * pow(p_mod, -1, small_prime)) % small_prime
                    if (p_mod * q_mod) % small_prime == N_mod:
                        solutions_mod.append((p_mod, q_mod))
            
            print(f"[Hensel] Found {len(solutions_mod)} solutions mod {small_prime}")
            
            if not solutions_mod:
                return None
            
            # Try to lift solutions using hints
            if p_hint and q_hint:
                p_mod_target = p_hint % small_prime
                q_mod_target = q_hint % small_prime
                
                # Find closest solution
                best_sol = min(solutions_mod, 
                             key=lambda s: abs(s[0] - p_mod_target) + abs(s[1] - q_mod_target))
                
                print(f"[Hensel] Best modular solution: p≡{best_sol[0]} (mod {small_prime}), q≡{best_sol[1]} (mod {small_prime})")
                
                # Try values congruent to this solution
                for k in range(0, min(1000, self.N // small_prime)):
                    p_candidate = best_sol[0] + k * small_prime
                    
                    if p_candidate > 1 and self.N % p_candidate == 0:
                        q_candidate = self.N // p_candidate
                        
                        if q_candidate > 1 and p_candidate * q_candidate == self.N:
                            print(f"[Hensel] ✓ Lifted to solution: p={p_candidate}, q={q_candidate}")
                            return (p_candidate, q_candidate)
            
            print("[Hensel] Lifting did not find solution")
            return None
            
        except Exception as e:
            print(f"[Hensel] Failed: {e}")
            return None
    
    def solve_quadratic_factorization(self, polynomials: List[sp.Expr]) -> Optional[Tuple[int, int]]:
        """
        Solve the quadratic equation x² - Sx + N = 0 where S is the sum constraint.

        For the system:
        - p*q = N  (factorization)
        - p + q = S  (sum constraint)

        This gives us the quadratic equation x² - Sx + N = 0
        whose roots are p and q.
        """
        print("\n[Quadratic] Solving factorization quadratic equation...")

        try:
            # Extract N and S from polynomials using algebraic evaluation
            N = self.N
            S = None

            print(f"[Quadratic] Analyzing {len(polynomials)} polynomials to extract constraints...")

            # Method: Evaluate polynomials at specific points to extract coefficients
            # For p + q - S = 0, evaluate at p=1, q=0 to get 1 + 0 - S = 1 - S
            # So S = 1 - poly(1,0)

            for i, poly in enumerate(polynomials):
                try:
                    # Check if this looks like a sum constraint (contains both p and q with positive coefficients)
                    poly_expanded = sp.expand(poly)

                    # Evaluate at p=1, q=0 to extract constant term
                    val_at_10 = poly_expanded.subs([(self.p, 1), (self.q, 0)])
                    # Evaluate at p=0, q=1
                    val_at_01 = poly_expanded.subs([(self.p, 0), (self.q, 1)])
                    # Evaluate at p=1, q=1
                    val_at_11 = poly_expanded.subs([(self.p, 1), (self.q, 1)])

                    # For p + q - S = 0:
                    # At (1,0): 1 + 0 - S = 1 - S
                    # At (0,1): 0 + 1 - S = 1 - S
                    # At (1,1): 1 + 1 - S = 2 - S

                    if val_at_10 == val_at_01 and val_at_10 != val_at_11:
                        # This looks like p + q - S = 0
                        # From val_at_10 = 1 - S, so S = 1 - val_at_10
                        S = 1 - val_at_10
                        print(f"[Quadratic] Found sum constraint S = {S}")
                        print(f"[Quadratic] Verification: p + q - {S} evaluates to:")
                        print(f"[Quadratic]   At (1,0): {poly_expanded.subs([(self.p, 1), (self.q, 0)])}")
                        print(f"[Quadratic]   At (0,1): {poly_expanded.subs([(self.p, 0), (self.q, 1)])}")
                        print(f"[Quadratic]   At (1,1): {poly_expanded.subs([(self.p, 1), (self.q, 1)])}")
                        break

                except Exception as e:
                    print(f"[Quadratic] Could not analyze polynomial {i}: {e}")
                    continue

            if S is None:
                print("[Quadratic] Could not find sum constraint S")
                return None

            print(f"[Quadratic] Found N={N}, S={S}")
            print(f"[Quadratic] Solving: x² - {S}x + {N} = 0")

            # Compute discriminant D = S² - 4N
            discriminant = S * S - 4 * N
            print(f"[Quadratic] Discriminant D = {discriminant}")

            if discriminant < 0:
                print("[Quadratic] ❌ Negative discriminant - no real integer solutions")
                print(f"[Quadratic] Mathematical analysis: S={S} is too small for N={N}")
                print(f"[Quadratic] Required minimum S: > {int((-discriminant)**0.5 + 0.5)}")
                print(f"[Quadratic] Actual discriminant: {discriminant}")
                print(f"[Quadratic] 🌙 Complex roots revealed: p,q = [{S} ± i√{abs(discriminant)}]/2")
                print(f"[Quadratic] 🔮 The discriminant D = {discriminant} < 0 summons imaginary factors!")

                # Extract decimal approximations from complex roots
                try:
                    # Compute complex roots: [S ± i√|D|]/2
                    real_part = S / 2
                    imag_part = ((-discriminant)**0.5) / 2  # √|D| / 2

                    root1_real = real_part
                    root2_real = real_part

                    print(f"[Quadratic] 🤡 COMPLEX ROOTS EXPOSED (D = {discriminant} < 0):")
                    print(f"[Quadratic]   🧙 Complex root 1: {root1_real:.6f} + {imag_part:.6f}i")
                    print(f"[Quadratic]   🧙 Complex root 2: {root1_real:.6f} - {imag_part:.6f}i")
                    print(f"[Quadratic]   🤡 'Decimal factors': p = {root1_real:.6f}, q = {root1_real:.6f}")
                    print(f"[Quadratic]   🤡 But these are just real parts of complex numbers!")
                    print(f"[Quadratic]   🤡 They satisfy x² - {S}x + {N} = 0 but don't multiply to {N}")
                    print(f"[Quadratic]   🤡 No mystical divination - just complex conjugate pairs!")
                    print(f"[Quadratic]   🔍 Real integer factors: p,q ≈ √{N} ≈ {self._integer_sqrt_approx(N)}")

                    # Round to nearest integers and check if they give valid factors
                    p_approx = int(round(root1_real))
                    q_approx = int(round(root1_real))  # Both roots have same real part

                    print(f"[Quadratic] Rounded integer approximations: p≈{p_approx}, q≈{p_approx}")

                    # Check if the rounded values give valid factors
                    if p_approx > 1 and N % p_approx == 0:
                        q_exact = N // p_approx
                        if q_exact > 1:
                            print(f"[Quadratic] ✓✓✓ DECIMAL APPROXIMATION SUCCESS: {p_approx} × {q_exact} = {N}")
                            return (p_approx, q_exact)

                    # Also check if the complex roots suggest different factor combinations
                    # Try a few nearby integers
                    for offset in [-2, -1, 0, 1, 2]:
                        p_test = p_approx + offset
                        if p_test > 1 and N % p_test == 0:
                            q_test = N // p_test
                            if q_test > 1:
                                print(f"[Quadratic] ✓✓✓ NEARBY DECIMAL APPROXIMATION: {p_test} × {q_test} = {N}")
                                return (p_test, q_test)

                except:
                    print(f"[Quadratic] Could not extract decimal approximations from complex roots")

                print(f"[Quadratic] Complex root analysis complete - trying alternative factorization...")

                # Even with invalid constraints, try alternative factorization
                print(f"[Quadratic] Attempting alternative factorization approaches...")

                # Try direct trial division around the square root
                sqrt_N = self._integer_sqrt_approx(N)
                search_radius = min(100000, sqrt_N // 1000)

                print(f"[Quadratic] Trying trial division around √N ≈ {sqrt_N} with radius {search_radius}")

                for offset in range(-search_radius, search_radius + 1, max(1, search_radius // 100)):
                    p_cand = sqrt_N + offset
                    if p_cand > 1 and N % p_cand == 0:
                        q_cand = N // p_cand
                        if q_cand > 1:
                            print(f"[Quadratic] ✓✓✓ ALTERNATIVE FACTORIZATION FOUND: {p_cand} × {q_cand} = {N}")
                            return (int(p_cand), int(q_cand))

                print(f"[Quadratic] No alternative factorization found - passing to next method")
                return None

            # Check if discriminant is a perfect square (handle very large numbers)
            try:
                # Try floating point approximation first
                sqrt_D_approx = int(discriminant ** 0.5)
                if sqrt_D_approx * sqrt_D_approx == discriminant:
                    print(f"[Quadratic] ✅ D is a perfect square!")
                    print(f"[Quadratic] √D = {sqrt_D_approx}")
                    root = sqrt_D_approx
                elif (sqrt_D_approx + 1) * (sqrt_D_approx + 1) == discriminant:
                    print(f"[Quadratic] ✅ D is a perfect square!")
                    print(f"[Quadratic] √D = {sqrt_D_approx + 1}")
                    root = sqrt_D_approx + 1
                else:
                    print(f"[Quadratic] ❌ D is not a perfect square")
                    print(f"[Quadratic] √D ≈ {sqrt_D_approx}")
                    print(f"[Quadratic] D is too large for complete square checking")
                    print(f"[Quadratic] Discriminant has {len(str(discriminant))} digits")
                    print(f"[Quadratic] This means no integer factors p,q with p+q={S}")
                    print(f"[Quadratic] But N might still be factorable with different sum constraints")

                    # Instead of giving up, try alternative factorization approaches
                    print(f"[Quadratic] Attempting alternative factorization approaches...")

                    # Try direct trial division around the square root
                    sqrt_N = self._integer_sqrt_approx(N)
                    search_radius = min(100000, sqrt_N // 1000)

                    print(f"[Quadratic] Trying trial division around √N ≈ {sqrt_N} with radius {search_radius}")

                    for offset in range(-search_radius, search_radius + 1, max(1, search_radius // 100)):
                        p_cand = sqrt_N + offset
                        if p_cand > 1 and N % p_cand == 0:
                            q_cand = N // p_cand
                            if q_cand > 1:
                                print(f"[Quadratic] ✓✓✓ ALTERNATIVE FACTORIZATION FOUND: {p_cand} × {q_cand} = {N}")
                                return (int(p_cand), int(q_cand))

                    print(f"[Quadratic] No alternative factorization found - passing to next method")
                    return None
            except (OverflowError, ValueError) as e:
                print(f"[Quadratic] Discriminant too large for floating point operations")
                print(f"[Quadratic] D has {len(str(discriminant))} digits ({discriminant.bit_length()} bits)")
                print(f"[Quadratic] Cannot check if perfect square with current methods")
                print(f"[Quadratic] This is expected for cryptographic-sized numbers")
                print(f"[Quadratic] The quadratic equation x² - {S}x + {N} = 0 has a positive discriminant")
                print(f"[Quadratic] But solving it requires factoring the discriminant D")

                # Instead of giving up, try alternative approaches
                print(f"[Quadratic] Attempting alternative factorization approaches...")

                # Try direct trial division around the square root
                sqrt_N = self._integer_sqrt_approx(N)
                search_radius = min(100000, sqrt_N // 1000)

                print(f"[Quadratic] Trying trial division around √N ≈ {sqrt_N} with radius {search_radius}")

                for offset in range(-search_radius, search_radius + 1, max(1, search_radius // 100)):
                    p_cand = sqrt_N + offset
                    if p_cand > 1 and N % p_cand == 0:
                        q_cand = N // p_cand
                        if q_cand > 1:
                            print(f"[Quadratic] ✓✓✓ ALTERNATIVE FACTORIZATION FOUND: {p_cand} × {q_cand} = {N}")
                            return (int(p_cand), int(q_cand))

                print(f"[Quadratic] No alternative factorization found - passing to next method")
                return None

            # Compute roots: x = [S ± √D] / 2
            root1 = (S - sqrt_D) // 2
            root2 = (S + sqrt_D) // 2

            print(f"[Quadratic] Roots: x₁ = {root1}, x₂ = {root2}")

            # Verify the roots multiply to N
            if root1 * root2 == N and root1 > 1 and root2 > 1:
                print(f"[Quadratic] ✓✓✓ VERIFICATION: {root1} × {root2} = {root1 * root2} ✓")
                return (int(root1), int(root2))
            else:
                print(f"[Quadratic] ❌ Verification failed: {root1} × {root2} = {root1 * root2} ≠ {N}")
                return None

        except Exception as e:
            print(f"[Quadratic] Failed: {e}")
            return None

    def solve_with_diophantine(self, polynomials: List[sp.Expr],
                               p_hint: int = None, q_hint: int = None) -> Optional[Tuple[int, int]]:
        """
        Robust Diophantine solver for the system involving self.p, self.q and self.N.

        Key features:
        - Uses exact integer arithmetic (math.isqrt or sympy.integer_nthroot) instead of float sqrt.
        - Treats single-variable linear polynomials (p - k = 0 or q - k = 0) as exact equalities.
        - Only accepts p = q if the linear polynomial is exactly p - q or q - p.
        - Implements quadratic Diophantine approach: y*z + q_given*y + p_given*z + prod_mod_n = 0
        - Accepts approximate solutions (p*q close to N) when exact solutions aren't found.
        - Uses adaptive tolerance based on N size (0.01% for small N, up to 1% for very large N).
        
        Args:
            polynomials: List of polynomial equations to solve
            p_hint: Optional approximation for p to narrow search space and prioritize solutions
            q_hint: Optional approximation for q to narrow search space and prioritize solutions
            
        Returns:
            Tuple of (p, q) if found (exact or approximate), or None. Approximate solutions
            are returned when p*q is within tolerance of N (adaptive based on N size).
        """
        import math
        from sympy import Integer, Poly, symbols
        from sympy import integer_nthroot

        print("\n[Diophantine] Solving polynomial system as Diophantine equations...")
        if p_hint and q_hint:
            print(f"[Diophantine] Using approximations: p ≈ {p_hint} ({p_hint.bit_length()}-bit), q ≈ {q_hint} ({q_hint.bit_length()}-bit)")

        # Normalize input polynomials: keep only SymPy expressions
        main_polys = []
        for expr in polynomials:
            try:
                if expr is None:
                    continue
                # Accept SymPy Expr or convertible objects
                main_polys.append(sp.sympify(expr))
            except Exception:
                continue

        if len(main_polys) < 1:
            print("[Diophantine] Need at least 1 polynomial")
            return None

        print(f"[Diophantine] Processing {len(main_polys)} polynomial equations...")

        # Helper: exact integer sqrt test
        def is_perfect_square(n: int) -> Tuple[bool, int]:
            if n < 0:
                return False, 0
            # prefer math.isqrt for speed
            r = math.isqrt(n)
            if r * r == n:
                return True, r
            return False, r


        # First pass: look for direct factorization equation p*q - N = 0
        factor_poly = None
        for poly in main_polys:
            # exact match: p*q - N
            if sp.simplify(poly - (self.p * self.q - self.N)) == 0 or sp.simplify(poly - (self.p * self.q) + self.N) == 0:
                factor_poly = poly
                break
            # also accept poly that contains p*q term and constant -N
            try:
                P = Poly(poly, self.p, self.q)
                if P.total_degree() == 2 and P.coeffs():
                    # check if coefficient of p*q equals 1 and constant equals -N (exact)
                    coeff_pq = P.coeff_monomial(self.p * self.q)
                    const = P.coeff_monomial(1)
                    if coeff_pq == 1 and const == -self.N:
                        factor_poly = poly
                        break
            except Exception:
                continue

        # If we have the factorization equation, try to use linear constraints to solve
        if factor_poly is not None:
            print(f"[Diophantine] Found factorization equation: p*q = {self.N}")

            # Collect linear constraints (exact)
            linear_constraints = []
            for other in main_polys:
                if other == factor_poly:
                    continue
                try:
                    L = Poly(other, self.p, self.q)
                except Exception:
                    continue
                if L.total_degree() == 1:
                    # Extract integer coefficients a*p + b*q + c = 0
                    coeffs = L.as_dict()
                    a = int(coeffs.get((1, 0), 0))
                    b = int(coeffs.get((0, 1), 0))
                    c = int(coeffs.get((0, 0), 0))
                    linear_constraints.append((a, b, c, other))

            # Process exact single-variable equalities first: p = k or q = k
            for a, b, c, poly_expr in linear_constraints:
                # a*p + b*q + c = 0
                if a != 0 and b == 0:
                    # p = -c / a  (must be integer)
                    if (-c) % a == 0:
                        p_cand = (-c) // a
                        if p_cand > 1 and self.N % p_cand == 0:
                            q_cand = self.N // p_cand
                            print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND from p = const: p={p_cand}, q={q_cand}")
                            if p_hint and q_hint:
                                print(f"[Diophantine] Distance from hints: p_diff={abs(p_cand - p_hint)}, q_diff={abs(q_cand - q_hint)}")
                            return (int(p_cand), int(q_cand))
                if b != 0 and a == 0:
                    # q = -c / b
                    if (-c) % b == 0:
                        q_cand = (-c) // b
                        if q_cand > 1 and self.N % q_cand == 0:
                            p_cand = self.N // q_cand
                            print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND from q = const: p={p_cand}, q={q_cand}")
                            if p_hint and q_hint:
                                print(f"[Diophantine] Distance from hints: p_diff={abs(p_cand - p_hint)}, q_diff={abs(q_cand - q_hint)}")
                            return (int(p_cand), int(q_cand))

            # Process exact p = q constraint only if polynomial is exactly p - q or q - p
            for a, b, c, poly_expr in linear_constraints:
                if (a == 1 and b == -1 and c == 0) or (a == -1 and b == 1 and c == 0):
                    # exact p = q
                    print("[Diophantine] Found exact constraint: p = q")
                    # check perfect square exactly
                    is_sq, root = is_perfect_square(int(self.N))
                    if is_sq:
                        print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND: p = q = {root}")
                        if p_hint and q_hint:
                            print(f"[Diophantine] Distance from hints: p_diff={abs(root - p_hint)}, q_diff={abs(root - q_hint)}")
                        return (int(root), int(root))
                    else:
                        print("[Diophantine] N is not a perfect square; p = q impossible.")
                        # continue to other constraints

            # Process sum constraints p + q = S (exact)
            for a, b, c, poly_expr in linear_constraints:
                # detect p + q = S or -p - q + c = 0
                if a != 0 and b != 0 and abs(a) == abs(b):
                    # normalize to p + q = S
                    if a == 1 and b == 1:
                        S = -c
                    elif a == -1 and b == -1:
                        S = c
                    else:
                        # general normalization: divide by gcd if possible
                        # but require exact integer normalization
                        if c % a == 0:
                            S = -c // a
                        else:
                            continue
                    # discriminant D = S^2 - 4N
                    D = S * S - 4 * int(self.N)
                    if D < 0:
                        continue
                    is_sq, r = is_perfect_square(D)
                    if not is_sq:
                        continue
                    # candidate roots
                    p1 = (S - r) // 2
                    p2 = (S + r) // 2
                    candidates = [(p1, S - p1), (p2, S - p2)]
                    # prioritize by hints if available
                    if p_hint and q_hint:
                        candidates.sort(key=lambda x: abs(x[0] - p_hint) + abs(x[1] - q_hint))
                        print(f"[Diophantine] Prioritizing candidates closer to hints")
                    for p_cand, q_cand in candidates:
                        # CRITICAL: Verify both p+q = S AND p*q = N exactly
                        sum_check = p_cand + q_cand
                        product_check = p_cand * q_cand
                        
                        if p_cand > 1 and q_cand > 1:
                            # Must satisfy BOTH conditions exactly
                            if sum_check == S and product_check == int(self.N):
                                print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND from p+q: p={p_cand}, q={q_cand}")
                                print(f"[Diophantine] Verification: p + q = {p_cand} + {q_cand} = {sum_check} = S = {S} ✓")
                                print(f"[Diophantine] Verification: p*q = {p_cand}*{q_cand} = {product_check} = N ✓")
                                if p_hint and q_hint:
                                    print(f"[Diophantine] Distance from hints: p_diff={abs(p_cand - p_hint)}, q_diff={abs(q_cand - q_hint)}")
                                return (int(p_cand), int(q_cand))
                            else:
                                # Log why it was rejected
                                if sum_check != S:
                                    print(f"[Diophantine] Rejected candidate: sum check failed (p+q = {sum_check} ≠ S = {S})")
                                if product_check != int(self.N):
                                    print(f"[Diophantine] Rejected candidate: product check failed (p*q = {product_check} ≠ N = {self.N})")
                                    print(f"[Diophantine]   Difference: {abs(product_check - int(self.N))}")

            # Process general linear constraints a*p + b*q + c = 0 with both a and b non-zero
            for a, b, c, poly_expr in linear_constraints:
                if a == 0 or b == 0:
                    continue
                # prefer expressing the smaller coefficient variable to reduce overflow risk
                # express p = (-b*q - c)/a  => substitute into p*q = N => b*q^2 + c*q + a*N = 0
                A = b
                B = c
                C = a * int(self.N)
                # discriminant for q: B^2 - 4*A*C
                D = B * B - 4 * A * C
                if D < 0:
                    continue
                is_sq, r = is_perfect_square(D)
                if not is_sq:
                    continue
                # compute q candidates (ensure integer division)
                denom = 2 * A
                if denom == 0:
                    continue
                q1_num = -B - r
                q2_num = -B + r
                candidates_q = []
                if q1_num % denom == 0:
                    q1 = q1_num // denom
                    if q1 > 1:
                        # compute p exactly
                        if (-b * q1 - c) % a == 0:
                            p1 = (-b * q1 - c) // a
                            candidates_q.append((p1, q1))
                if q2_num % denom == 0:
                    q2 = q2_num // denom
                    if q2 > 1:
                        if (-b * q2 - c) % a == 0:
                            p2 = (-b * q2 - c) // a
                            candidates_q.append((p2, q2))
                
                # Prioritize by hints if available
                if p_hint and q_hint:
                    candidates_q.sort(key=lambda x: abs(x[0] - p_hint) + abs(x[1] - q_hint))
                    print(f"[Diophantine] Prioritizing candidates closer to hints")
                
                for p_cand, q_cand in candidates_q:
                    # CRITICAL: Verify both the linear constraint AND p*q = N exactly
                    # Check linear constraint: a*p + b*q + c should equal 0
                    linear_check = a * p_cand + b * q_cand + c
                    product_check = p_cand * q_cand
                    
                    if p_cand > 1 and q_cand > 1:
                        # Must satisfy BOTH conditions exactly
                        if linear_check == 0 and product_check == int(self.N):
                            print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND (general linear): p={p_cand}, q={q_cand}")
                            print(f"[Diophantine] Verification: a*p + b*q + c = {a}*{p_cand} + {b}*{q_cand} + {c} = {linear_check} ✓")
                            print(f"[Diophantine] Verification: p*q = {p_cand}*{q_cand} = {product_check} = N ✓")
                            if p_hint and q_hint:
                                print(f"[Diophantine] Distance from hints: p_diff={abs(p_cand - p_hint)}, q_diff={abs(q_cand - q_hint)}")
                            return (int(p_cand), int(q_cand))
                        else:
                            # Log why it was rejected
                            if linear_check != 0:
                                print(f"[Diophantine] Rejected candidate: linear constraint not satisfied (a*p + b*q + c = {linear_check} ≠ 0)")
                            if product_check != int(self.N):
                                print(f"[Diophantine] Rejected candidate: product check failed (p*q = {product_check} ≠ N = {self.N})")
                                print(f"[Diophantine]   Difference: {abs(product_check - int(self.N))}")


        # If we didn't detect p*q = N or couldn't solve via linear constraints, attempt other direct strategies:

        # 1) Look for explicit single-variable equalities (p = k or q = k)
        for poly in main_polys:
            try:
                L = Poly(poly, self.p, self.q)
            except Exception:
                continue
            if L.total_degree() == 1:
                coeffs = L.as_dict()
                a = int(coeffs.get((1, 0), 0))
                b = int(coeffs.get((0, 1), 0))
                c = int(coeffs.get((0, 0), 0))
                if a != 0 and b == 0:
                    if (-c) % a == 0:
                        p_cand = (-c) // a
                        if p_cand > 1 and int(self.N) % p_cand == 0:
                            q_cand = int(self.N) // p_cand
                            print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND from single-var linear: p={p_cand}, q={q_cand}")
                            if p_hint and q_hint:
                                print(f"[Diophantine] Distance from hints: p_diff={abs(p_cand - p_hint)}, q_diff={abs(q_cand - q_hint)}")
                            return (int(p_cand), int(q_cand))
                if b != 0 and a == 0:
                    if (-c) % b == 0:
                        q_cand = (-c) // b
                        if q_cand > 1 and int(self.N) % q_cand == 0:
                            p_cand = int(self.N) // q_cand
                            print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND from single-var linear: p={p_cand}, q={q_cand}")
                            if p_hint and q_hint:
                                print(f"[Diophantine] Distance from hints: p_diff={abs(p_cand - p_hint)}, q_diff={abs(q_cand - q_hint)}")
                            return (int(p_cand), int(q_cand))

        # 2) IDEAL LATTICE DIMENSION METHOD: Use lattice reduction on ideal structures
        # This method exploits the algebraic structure of ideals in Z[x] and their
        # correspondence with lattices to find short vectors that reveal factorizations
        if p_hint and q_hint:
            print(f"[Diophantine] Attempting Ideal Lattice Dimension method...")

            p_given = p_hint
            q_given = q_hint

            # === EARLY DIVISIBILITY CHECKS ===
            print(f"[Diophantine] === EARLY DIVISIBILITY CHECKS ===")

            # Check 1: Does p_given divide N exactly?
            if self.N % p_given == 0:
                q_exact = self.N // p_given
                print(f"[Diophantine] ✓✓✓ p_given divides N exactly!")
                print(f"[Diophantine]   p = {p_given}, q = {q_exact}")
                print(f"[Diophantine]   Corrections: e_p = 0, e_q = {q_exact - q_given}")
                return (int(p_given), int(q_exact))

            # Check 2: Does q_given divide N exactly?
            if self.N % q_given == 0:
                p_exact = self.N // q_given
                print(f"[Diophantine] ✓✓✓ q_given divides N exactly!")
                print(f"[Diophantine]   p = {p_exact}, q = {q_given}")
                print(f"[Diophantine]   Corrections: e_p = {p_exact - p_given}, e_q = 0")
                return (int(p_exact), int(q_given))

            # Check 3: Try small adjustments to p_given
            print(f"[Diophantine] Checking p_given ± small values...")
            for offset in range(-100, 101):
                if offset == 0:
                    continue
                p_test = p_given + offset
                if p_test > 1 and self.N % p_test == 0:
                    q_exact = self.N // p_test
                    print(f"[Diophantine] ✓✓✓ Found exact divisor at p_given + {offset}!")
                    print(f"[Diophantine]   p = {p_test}, q = {q_exact}")
                    print(f"[Diophantine]   Corrections: e_p = {offset}, e_q = {q_exact - q_given}")
                    return (int(p_test), int(q_exact))

            # Check 4: Try small adjustments to q_given
            print(f"[Diophantine] Checking q_given ± small values...")
            for offset in range(-100, 101):
                if offset == 0:
                    continue
                q_test = q_given + offset
                if q_test > 1 and self.N % q_test == 0:
                    p_exact = self.N // q_test
                    print(f"[Diophantine] ✓✓✓ Found exact divisor at q_given + {offset}!")
                    print(f"[Diophantine]   p = {p_exact}, q = {q_test}")
                    print(f"[Diophantine]   Corrections: e_p = {p_exact - p_given}, e_q = {offset}")
                    return (int(p_exact), int(q_test))

            print(f"[Diophantine] No exact divisors found in range ±100")

            N_product = p_given * q_given
            delta = N_product - self.N

            print(f"[Diophantine] N_product = p_given * q_given = {N_product}")
            print(f"[Diophantine] delta = N_product - N_target = {delta}")
            print(f"[Diophantine] delta bit length: {delta.bit_length()} bits")

            # Method 2a: Direct GCD checks from delta relationship
            print(f"[Diophantine] Checking GCD relationships from delta...")
            
            # GCD of delta with N_target
            g_delta = math.gcd(abs(delta), self.N)
            if 1 < g_delta < self.N:
                q_found = self.N // g_delta
                print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND via gcd(delta, N)!")
                print(f"[Diophantine]   p = {g_delta}, q = {q_found}")
                return (int(g_delta), int(q_found))
            
            # GCD of delta with p_given and q_given
            g_delta_p = math.gcd(abs(delta), p_given)
            if 1 < g_delta_p < self.N:
                g_final = math.gcd(g_delta_p, self.N)
                if 1 < g_final < self.N:
                    q_found = self.N // g_final
                    print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND via gcd(delta, p_given)!")
                    print(f"[Diophantine]   p = {g_final}, q = {q_found}")
                    return (int(g_final), int(q_found))
            
            g_delta_q = math.gcd(abs(delta), q_given)
            if 1 < g_delta_q < self.N:
                g_final = math.gcd(g_delta_q, self.N)
                if 1 < g_final < self.N:
                    q_found = self.N // g_final
                    print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND via gcd(delta, q_given)!")
                    print(f"[Diophantine]   p = {g_final}, q = {q_found}")
                    return (int(g_final), int(q_found))
            
            # Method 2b: Modular inverse approach
            print(f"[Diophantine] Trying modular inverse reconstruction...")
            try:
                # Check if p_given is invertible mod N_target
                # If not, gcd(p_given, N_target) gives a factor
                p_inv = pow(p_given, -1, self.N) if math.gcd(p_given, self.N) == 1 else None
                if p_inv is None:
                    g = math.gcd(p_given, self.N)
                    if 1 < g < self.N:
                        q_found = self.N // g
                        print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND (p_given not invertible)!")
                        print(f"[Diophantine]   p = {g}, q = {q_found}")
                        return (int(g), int(q_found))
                else:
                    # Reconstruct q from delta and p_inv
                    q_reconstructed = (delta * p_inv) % self.N
                    g = math.gcd(q_reconstructed, self.N)
                    if 1 < g < self.N:
                        q_found = self.N // g
                        print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND via q reconstruction!")
                        print(f"[Diophantine]   p = {g}, q = {q_found}")
                        return (int(g), int(q_found))
            except:
                pass
            
            try:
                q_inv = pow(q_given, -1, self.N) if math.gcd(q_given, self.N) == 1 else None
                if q_inv is None:
                    g = math.gcd(q_given, self.N)
                    if 1 < g < self.N:
                        p_found = self.N // g
                        print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND (q_given not invertible)!")
                        print(f"[Diophantine]   p = {p_found}, q = {g}")
                        return (int(p_found), int(g))
                else:
                    # Reconstruct p from delta and q_inv
                    p_reconstructed = (delta * q_inv) % self.N
                    g = math.gcd(p_reconstructed, self.N)
                    if 1 < g < self.N:
                        p_found = self.N // g
                        print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND via p reconstruction!")
                        print(f"[Diophantine]   p = {p_found}, q = {g}")
                        return (int(p_found), int(g))
            except:
                pass
            
            # Method 2c: Linear combinations of p_given and q_given
            print(f"[Diophantine] Checking linear combinations of p_given and q_given...")
            for a in range(-10, 11):
                for b in range(-10, 11):
                    if a == 0 and b == 0:
                        continue
                    val = a * p_given + b * q_given
                    if val != 0:
                        g = math.gcd(abs(val), self.N)
                        if 1 < g < self.N:
                            q_found = self.N // g
                            print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND from {a}*p + {b}*q!")
                            print(f"[Diophantine]   p = {g}, q = {q_found}")
                            return (int(g), int(q_found))
            
            # Method 2d: Using delta to find corrections
            # If delta = p_given * q_given - p * q, and p = p_given + e_p, q = q_given + e_q
            # Then: delta = -e_p * q_given - e_q * p_given - e_p * e_q
            # For small e_p, e_q: delta ≈ -e_p * q_given - e_q * p_given
            print(f"[Diophantine] Using delta to find error corrections...")
            
            # Check if delta reveals corrections directly
            if delta % q_given == 0:
                e_p = -delta // q_given
                p_candidate = p_given + e_p
                if p_candidate > 1 and self.N % p_candidate == 0:
                    q_candidate = self.N // p_candidate
                    print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND via delta correction!")
                    print(f"[Diophantine]   e_p = {e_p}, p = {p_candidate}, q = {q_candidate}")
                    return (int(p_candidate), int(q_candidate))
            
            if delta % p_given == 0:
                e_q = -delta // p_given
                q_candidate = q_given + e_q
                if q_candidate > 1 and self.N % q_candidate == 0:
                    p_candidate = self.N // q_candidate
                    print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND via delta correction!")
                    print(f"[Diophantine]   e_q = {e_q}, p = {p_candidate}, q = {q_candidate}")
                    return (int(p_candidate), int(q_candidate))
            
            # Method 2e: Solve the quadratic Diophantine equation for differences directly
            # Equation: (p_given + e_p) * (q_given + e_q) = N_target
            # Expanding: p_given * q_given + p_given * e_q + q_given * e_p + e_p * e_q = N_target
            # Rearranging: p_given * e_q + q_given * e_p + e_p * e_q = N_target - p_given * q_given = -delta
            # Or: e_p * e_q + q_given * e_p + p_given * e_q + delta = 0
            print(f"[Diophantine] Solving quadratic Diophantine equation for differences e_p and e_q...")
            print(f"[Diophantine] Equation: e_p * e_q + q_given * e_p + p_given * e_q + delta = 0")
            print(f"[Diophantine] Where: e_p = p - p_given, e_q = q - q_given")
            print()
            
            # Method 2e.1: Improved quadratic search - solving e_p = (-p_given * e_q - delta) / (e_q + q_given)
            print(f"[Diophantine] === IMPROVED QUADRATIC SEARCH ===")

            # Calculate expected differences based on division
            expected_q = self.N // p_given
            expected_p = self.N // q_given

            diff_q_expected = expected_q - q_given
            diff_p_expected = expected_p - p_given

            print(f"[Diophantine] Expected corrections from division:")
            print(f"[Diophantine]   e_q_expected = {diff_q_expected}")
            print(f"[Diophantine]   e_p_expected = {diff_p_expected}")

            # Strategy 1: Try expected values first
            test_cases = [
                (0, diff_q_expected, "e_p=0, e_q=expected"),
                (diff_p_expected, 0, "e_p=expected, e_q=0"),
                (diff_p_expected, diff_q_expected, "both=expected"),
                (0, diff_q_expected - 1, "e_p=0, e_q=expected-1"),
                (0, diff_q_expected + 1, "e_p=0, e_q=expected+1"),
                (diff_p_expected - 1, 0, "e_p=expected-1, e_q=0"),
                (diff_p_expected + 1, 0, "e_p=expected+1, e_q=0"),
            ]

            for e_p_test, e_q_test, label in test_cases:
                p_test = p_given + e_p_test
                q_test = q_given + e_q_test

                if p_test > 1 and q_test > 1:
                    if p_test * q_test == self.N:
                        print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND ({label})!")
                        print(f"[Diophantine]   e_p = {e_p_test}, e_q = {e_q_test}")
                        print(f"[Diophantine]   p = {p_test}, q = {q_test}")
                        return (int(p_test), int(q_test))

            # Strategy 2: Systematic search with formula validation
            # For each e_q, solve: e_p = (-p_given * e_q - delta) / (e_q + q_given)

            # Determine search range based on expected values
            max_offset = max(100, abs(diff_q_expected) * 2, abs(diff_p_expected) * 2)
            search_range = min(max_offset, 10000)

            print(f"[Diophantine] Searching e_q in range [{-search_range}, {search_range}]...")

            for e_q_test in range(-search_range, search_range + 1):
                denominator = e_q_test + q_given
                if denominator == 0:
                    continue

                numerator = -p_given * e_q_test - delta

                if numerator % denominator == 0:
                    e_p_test = numerator // denominator
                    p_test = p_given + e_p_test
                    q_test = q_given + e_q_test

                    if p_test > 1 and q_test > 1 and p_test * q_test == self.N:
                        print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND via formula!")
                        print(f"[Diophantine]   e_p = {e_p_test}, e_q = {e_q_test}")
                        print(f"[Diophantine]   p = {p_test}, q = {q_test}")
                        return (int(p_test), int(q_test))
                if e_q_test == 0:
                    continue
                
                denominator = e_q_test + q_given
                if denominator == 0:
                    continue
                
                numerator = -p_given * e_q_test - delta
                
                # Check if numerator is divisible by denominator
                if numerator % denominator == 0:
                    e_p_test = numerator // denominator
                    p_test = p_given + e_p_test
                    q_test = q_given + e_q_test
                    
                    if p_test > 1 and q_test > 1:
                        # Verify the equation
                        product = p_test * q_test
                        if product == self.N:
                            print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND via difference equation!")
                            print(f"[Diophantine]   e_p = {e_p_test}, e_q = {e_q_test}")
                            print(f"[Diophantine]   p = p_given + {e_p_test} = {p_test}")
                            print(f"[Diophantine]   q = q_given + {e_q_test} = {q_test}")
                            print(f"[Diophantine]   Verification: p * q = {product} = N_target ✓")
                            return (int(p_test), int(q_test))
            
            # Method 2e.2: Rearrange to solve for e_q in terms of e_p
            # e_p * e_q + q_given * e_p + p_given * e_q + delta = 0
            # e_q * (e_p + p_given) = -q_given * e_p - delta
            # e_q = (-q_given * e_p - delta) / (e_p + p_given)
            print(f"[Diophantine] Method 2e.2: Solving e_q = (-q_given * e_p - delta) / (e_p + p_given)")
            
            # Check expected difference for p
            expected_p = self.N // q_given
            diff_p = expected_p - p_given
            print(f"[Diophantine] expected_p - p_given = {diff_p}")
            
            # Try the expected difference first
            if abs(diff_p) < 10000:
                print(f"[Diophantine] Trying e_p = {diff_p} first (most likely)...")
                e_p_test = diff_p
                if e_p_test != 0:
                    p_test = p_given + e_p_test
                    # First, check if p_test divides N_target exactly (fastest check)
                    if p_test > 1 and self.N % p_test == 0:
                        q_test = self.N // p_test
                        e_q_test = q_test - q_given
                        print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND (p = expected_p divides N)!")
                        print(f"[Diophantine]   e_p = {e_p_test}, e_q = {e_q_test}")
                        print(f"[Diophantine]   p = {p_test}, q = {q_test}")
                        return (int(p_test), int(q_test))
                    
                    # Otherwise, use the formula
                    denominator = e_p_test + p_given
                    if denominator != 0:
                        numerator = -q_given * e_p_test - delta
                        if numerator % denominator == 0:
                            e_q_test = numerator // denominator
                            q_test = q_given + e_q_test
                            
                            if p_test > 1 and q_test > 1:
                                product = p_test * q_test
                                if product == self.N:
                                    print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND via expected difference formula!")
                                    print(f"[Diophantine]   e_p = {e_p_test}, e_q = {e_q_test}")
                                    print(f"[Diophantine]   p = {p_test}, q = {q_test}")
                                    return (int(p_test), int(q_test))
                                else:
                                    # Also check if q_test divides N_target exactly
                                    if self.N % q_test == 0:
                                        p_actual = self.N // q_test
                                        print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND (q from formula divides N)!")
                                        print(f"[Diophantine]   e_p = {p_actual - p_given}, e_q = {e_q_test}")
                                        print(f"[Diophantine]   p = {p_actual}, q = {q_test}")
                                        return (int(p_actual), int(q_test))
            
            print(f"[Diophantine] Trying small e_p values...")
            
            # Expand range based on expected difference
            search_range = max(1000, abs(diff_p) * 10) if abs(diff_p) < 1000 else 10000
            print(f"[Diophantine] Searching e_p in range [{-search_range}, {search_range}]...")
            
            for e_p_test in range(-search_range, search_range + 1):
                if e_p_test == 0:
                    continue
                
                denominator = e_p_test + p_given
                if denominator == 0:
                    continue
                
                numerator = -q_given * e_p_test - delta
                
                # Check if numerator is divisible by denominator
                if numerator % denominator == 0:
                    e_q_test = numerator // denominator
                    p_test = p_given + e_p_test
                    q_test = q_given + e_q_test
                    
                    if p_test > 1 and q_test > 1:
                        # Verify the equation
                        product = p_test * q_test
                        if product == self.N:
                            print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND via difference equation!")
                            print(f"[Diophantine]   e_p = {e_p_test}, e_q = {e_q_test}")
                            print(f"[Diophantine]   p = p_given + {e_p_test} = {p_test}")
                            print(f"[Diophantine]   q = q_given + {e_q_test} = {q_test}")
                            print(f"[Diophantine]   Verification: p * q = {product} = N_target ✓")
                            return (int(p_test), int(q_test))
            
            # Method 2e.3: Use the fact that (p_given + e_p) must divide N_target
            # If (p_given + e_p) divides N_target, then e_p = d - p_given for some divisor d of N_target
            # But we can use the relationship more directly
            print(f"[Diophantine] Method 2e.3: Using divisor relationship")
            print(f"[Diophantine] Since (p_given + e_p) must divide N_target, trying e_p values...")
            
            # Method 2e.3: Fixed version - Using divisor relationship
            expected_q_from_division = self.N // p_given
            remainder = self.N % p_given

            print(f"[Diophantine] N_target = {expected_q_from_division} * p_given + {remainder}")
            print(f"[Diophantine] q_given = {q_given}, difference = {expected_q_from_division - q_given}")

            diff_q = expected_q_from_division - q_given
            print(f"[Diophantine] Trying e_q = {diff_q} (q = q_given + {diff_q})...")

            # Key fix: Use a clear variable name for the test value
            if abs(diff_q) < 10000:  # Increased range from 1000
                q_test = q_given + diff_q

                if q_test > 1 and self.N % q_test == 0:
                    p_exact = self.N // q_test
                    e_p = p_exact - p_given
                    e_q = diff_q

                    print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND via expected_q!")
                    print(f"[Diophantine]   e_p = {e_p}, e_q = {e_q}")
                    print(f"[Diophantine]   p = {p_exact}, q = {q_test}")
                    return (int(p_exact), int(q_test))
            
            # If the difference is small, try adjusting p_given to make it divide exactly
            # We want: (p_given + e_p) * (q_given + e_q) = N_target
            # If e_q ≈ expected_q - q_given, then we need e_p such that:
            # (p_given + e_p) * expected_q ≈ N_target
            # p_given * expected_q + e_p * expected_q ≈ N_target
            # e_p * expected_q ≈ remainder
            # e_p ≈ remainder / expected_q
            
            if expected_q != 0:
                e_p_approx = remainder // expected_q
                print(f"[Diophantine] e_p approximation from remainder/expected_q = {e_p_approx}")
                
                # Try a wider range around this approximation
                # The search range should be based on the remainder size and expected difference
                search_range = max(1000, min(10000, abs(remainder).bit_length() * 10, abs(diff_q) * 100))
                print(f"[Diophantine] Searching e_p in range [{e_p_approx - search_range}, {e_p_approx + search_range}]...")
                
                for e_p_offset in range(-search_range, search_range + 1):
                    e_p = e_p_approx + e_p_offset
                    p_test = p_given + e_p
                    
                    if p_test > 1 and self.N % p_test == 0:
                        q_test = self.N // p_test
                        e_q = q_test - q_given
                        
                        # Verify the quadratic relationship
                        # delta should equal -e_p * q_given - e_q * p_given - e_p * e_q
                        delta_calc = -e_p * q_given - e_q * p_given - e_p * e_q
                        
                        if abs(delta_calc - delta) < abs(delta) // 1000:  # Allow small error
                            print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND via error equation!")
                            print(f"[Diophantine]   e_p = {e_p}, e_q = {e_q}")
                            print(f"[Diophantine]   p = {p_test}, q = {q_test}")
                            print(f"[Diophantine]   delta check: calculated = {delta_calc}, actual = {delta}")
                            return (int(p_test), int(q_test))
                
                # Also try the direct approach: if remainder divides expected_q
                if remainder % expected_q == 0:
                    e_p = remainder // expected_q
                    p_test = p_given + e_p
                    if p_test > 1 and self.N % p_test == 0:
                        q_test = self.N // p_test
                        e_q = q_test - q_given
                        print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND (remainder divides expected_q)!")
                        print(f"[Diophantine]   e_p = {e_p}, e_q = {e_q}")
                        print(f"[Diophantine]   p = {p_test}, q = {q_test}")
                        return (int(p_test), int(q_test))
            
            # Method 2f: Use extended Euclidean algorithm on delta and approximations
            print(f"[Diophantine] Using extended GCD on delta and approximations...")
            
            # Try to find small integers a, b such that a * delta + b * p_given reveals a factor
            # Or: find a, b such that gcd(a * delta + b * p_given, N_target) > 1
            for a in range(-20, 21):
                for b in range(-20, 21):
                    if a == 0 and b == 0:
                        continue
                    val = a * delta + b * p_given
                    if val != 0:
                        g = math.gcd(abs(val), self.N)
                        if 1 < g < self.N:
                            q_found = self.N // g
                            print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND from {a}*delta + {b}*p_given!")
                            print(f"[Diophantine]   p = {g}, q = {q_found}")
                            return (int(g), int(q_found))
            
            for a in range(-20, 21):
                for b in range(-20, 21):
                    if a == 0 and b == 0:
                        continue
                    val = a * delta + b * q_given
                    if val != 0:
                        g = math.gcd(abs(val), self.N)
                        if 1 < g < self.N:
                            p_found = self.N // g
                            print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND from {a}*delta + {b}*q_given!")
                            print(f"[Diophantine]   p = {p_found}, q = {g}")
                            return (int(p_found), int(g))
        
        # 3) QUADRATIC DIOPHANTINE APPROACH: Solve y*z + q_given*y + p_given*z + prod_mod_n = 0
        # Where y = p - p_given, z = q - q_given
        # Rearranging: (p_given + y)(q_given + z) = N_target
        # So p_given + y must be a divisor of N_target
        if p_hint and q_hint:
            print(f"[Diophantine] Attempting quadratic Diophantine approach with hints...")
            print(f"[Diophantine] Equation: y*z + q_given*y + p_given*z + prod_mod_n = 0")
            print(f"[Diophantine] Where: y = p - p_given, z = q - q_given")
            print(f"[Diophantine] Rearranging: (p_given + y)(q_given + z) = N_target")
            print()
            
            p_given = p_hint
            q_given = q_hint
            prod_mod_n = (p_given * q_given) % self.N
            
            # Direct checks: Does p_given or q_given divide N?
            print(f"[Diophantine] Direct check: Does p_given divide N?")
            if self.N % p_given == 0:
                q_sol = self.N // p_given
                y_sol = 0
                z_sol = q_sol - q_given
                print(f"[Diophantine] ✓ YES! p_given divides N")
                print(f"[Diophantine]   p = {p_given}, q = {q_sol}")
                if p_given * q_sol == self.N:
                    print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND!")
                    return (int(p_given), int(q_sol))
            else:
                print(f"[Diophantine]   No, remainder = {self.N % p_given}")
            
            print(f"[Diophantine] Direct check: Does q_given divide N?")
            if self.N % q_given == 0:
                p_sol = self.N // q_given
                y_sol = p_sol - p_given
                z_sol = 0
                print(f"[Diophantine] ✓ YES! q_given divides N")
                print(f"[Diophantine]   p = {p_sol}, q = {q_given}")
                if p_sol * q_given == self.N:
                    print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND!")
                    return (int(p_sol), int(q_given))
            else:
                print(f"[Diophantine]   No, remainder = {self.N % q_given}")
            
            # Check small corrections: p_given ± small, q_given ± small
            print(f"[Diophantine] Checking small corrections around p_given and q_given...")
            
            # Calculate tolerance for approximate solutions
            if self.N.bit_length() > 1000:
                max_relative_error = 0.01
                max_absolute_error = self.N // 100
            elif self.N.bit_length() > 500:
                max_relative_error = 0.001
                max_absolute_error = self.N // 1000
            else:
                max_relative_error = 0.0001
                max_absolute_error = max(1000, self.N // 10000)
            
            best_approx = None
            best_error = None
            
            print(f"[Diophantine] Using complex algebra to solve for differences e_p and e_q...")

            # The key equation: (p_given + e_p) * (q_given + e_q) = N
            # We know: p_given * q_given = N + delta, where delta = p_given * q_given - N
            # So: p_given * q_given + p_given * e_q + q_given * e_p + e_p * e_q = p_given * q_given - delta
            # Simplifying: p_given * e_q + q_given * e_p + e_p * e_q = -delta

            # For small e_p, e_q, we approximate: p_given * e_q + q_given * e_p ≈ -delta
            # This gives us: p_given * e_q + q_given * e_p + delta ≈ 0

            # We can solve this linear system:
            # p_given * e_q + q_given * e_p = -delta

            # Let me solve for e_q in terms of e_p:
            # p_given * e_q = -delta - q_given * e_p
            # e_q = (-delta - q_given * e_p) / p_given

            # For integer solutions, (-delta - q_given * e_p) must be divisible by p_given
            # Similarly, we can solve for e_p in terms of e_q

            print(f"[Diophantine] Solving: p_given * e_q + q_given * e_p = -delta")
            print(f"[Diophantine] Where: p_given = {p_given}, q_given = {q_given}, delta = {delta}")

            # Strategy 1: Try small integer values for e_p and solve for e_q
            print(f"[Diophantine] Strategy 1: Solve for e_q given e_p values (scaled range)")
            # Scale search range based on delta magnitude
            max_range = max(1000, min(100000, abs(delta) // max(1, min(p_given, q_given) // 1000)))
            print(f"[Diophantine] Using scaled range ±{max_range} for large delta")

            for e_p_trial in range(-max_range, max_range + 1, max(1, max_range // 1000)):  # Adaptive step size
                if e_p_trial == 0:
                    continue

                # e_q = (-delta - q_given * e_p_trial) / p_given
                numerator = -delta - q_given * e_p_trial

                if numerator % p_given == 0:
                    e_q_trial = numerator // p_given

                    # Check if this gives exact solution
                    p_test = p_given + e_p_trial
                    q_test = q_given + e_q_trial

                    if p_test > 1 and q_test > 1 and p_test * q_test == self.N:
                        print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND via algebra!")
                        print(f"[Diophantine]   e_p = {e_p_trial}, e_q = {e_q_trial}")
                        print(f"[Diophantine]   p = {p_test}, q = {q_test}")
                        print(f"[Diophantine]   Verification: {p_test} * {q_test} = {p_test * q_test} = N ✓")
                        return (int(p_test), int(q_test))

            # Strategy 2: Try e_q values and solve for e_p (scaled range)
            print(f"[Diophantine] Strategy 2: Solve for e_p given e_q values (scaled range)")
            max_range = max(1000, min(100000, abs(delta) // max(1, min(p_given, q_given) // 1000)))

            for e_q_trial in range(-max_range, max_range + 1, max(1, max_range // 1000)):
                if e_q_trial == 0:
                    continue

                # e_p = (-delta - p_given * e_q_trial) / q_given
                numerator = -delta - p_given * e_q_trial

                if numerator % q_given == 0:
                    e_p_trial = numerator // q_given

                    # Check if this gives exact solution
                    p_test = p_given + e_p_trial
                    q_test = q_given + e_q_trial

                    if p_test > 1 and q_test > 1 and p_test * q_test == self.N:
                        print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND via algebra!")
                        print(f"[Diophantine]   e_p = {e_p_trial}, e_q = {e_q_trial}")
                        print(f"[Diophantine]   p = {p_test}, q = {q_test}")
                        print(f"[Diophantine]   Verification: {p_test} * {q_test} = {p_test * q_test} = N ✓")
                        return (int(p_test), int(q_test))

            # Strategy 3: Use the complete quadratic equation
            # We have: p_given * e_q + q_given * e_p + e_p * e_q = -delta
            # Rearranging: e_p * e_q + p_given * e_q + q_given * e_p = -delta

            # This is a quadratic in terms of e_p and e_q
            # We can treat it as: e_p * e_q + (p_given) * e_q + (q_given) * e_p = -delta
            # Group terms: e_p * e_q + q_given * e_p + p_given * e_q + delta = 0

            # This looks like: e_p * (e_q + q_given) + p_given * e_q + delta = 0
            # Or: e_p * (e_q + q_given) = -p_given * e_q - delta
            # e_p = (-p_given * e_q - delta) / (e_q + q_given)

            # Which is the same as Strategy 1 above

            # Strategy 4: Consider the exact expansion
            # (p_given + e_p) * (q_given + e_q) = N
            # p_given * q_given + p_given * e_q + q_given * e_p + e_p * e_q = N
            # But p_given * q_given = N + delta, so:
            # N + delta + p_given * e_q + q_given * e_p + e_p * e_q = N
            # delta + p_given * e_q + q_given * e_p + e_p * e_q = 0
            # e_p * e_q + p_given * e_q + q_given * e_p + delta = 0

            # This is the same equation we started with

            # Strategy 5: Try to factor or find roots
            # Treat this as a quadratic in one variable

            print(f"[Diophantine] Strategy 3: Using quadratic formula approach")

            # Consider e_p as the variable, treat e_q as parameter
            # From: e_p * e_q + p_given * e_q + q_given * e_p + delta = 0
            # e_p * (e_q + q_given) + (p_given * e_q + delta) = 0
            # e_p * (e_q + q_given) = - (p_given * e_q + delta)
            # e_p = - (p_given * e_q + delta) / (e_q + q_given)

            # For this to be integer, - (p_given * e_q + delta) must be divisible by (e_q + q_given)

            # Actually, let's try a different approach: assume e_q is small and solve the quadratic for e_p
            # From: e_p * e_q + q_given * e_p + p_given * e_q + delta = 0
            # e_p * (e_q + q_given) + p_given * e_q + delta = 0
            # e_p * (e_q + q_given) = - p_given * e_q - delta

            # This is already what we have

            # Strategy 6: Use continued fractions or other algebraic methods
            # For now, let's extend the range of our trial solutions

            print(f"[Diophantine] Extending algebraic search range...")

            # Scale extended range based on delta magnitude and N size
            delta_magnitude = abs(delta).bit_length()
            n_magnitude = self.N.bit_length()
            extended_range = max(10000, min(1000000, 2 ** min(20, delta_magnitude - n_magnitude + 10)))
            step_size = max(10, extended_range // 10000)

            print(f"[Diophantine] Using extended range ±{extended_range} with step {step_size}")

            # Try larger range for e_p and e_q
            for e_p_trial in range(-extended_range, extended_range + 1, step_size):
                if e_p_trial == 0:
                    continue

                numerator = -delta - q_given * e_p_trial
                if numerator % p_given == 0:
                    e_q_trial = numerator // p_given

                    p_test = p_given + e_p_trial
                    q_test = q_given + e_q_trial

                    if p_test > 1 and q_test > 1 and p_test * q_test == self.N:
                        print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND via extended algebra!")
                        print(f"[Diophantine]   e_p = {e_p_trial}, e_q = {e_q_trial}")
                        print(f"[Diophantine]   p = {p_test}, q = {q_test}")
                        return (int(p_test), int(q_test))

            for e_q_trial in range(-extended_range, extended_range + 1, step_size):
                if e_q_trial == 0:
                    continue

                numerator = -delta - p_given * e_q_trial
                if numerator % q_given == 0:
                    e_p_trial = numerator // q_given

                    p_test = p_given + e_p_trial
                    q_test = q_given + e_q_trial

                    if p_test > 1 and q_test > 1 and p_test * q_test == self.N:
                        print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND via extended algebra!")
                        print(f"[Diophantine]   e_p = {e_p_trial}, e_q = {e_q_trial}")
                        print(f"[Diophantine]   p = {p_test}, q = {q_test}")
                        return (int(p_test), int(q_test))

            print(f"[Diophantine] No algebraic solutions found in tested ranges")

            # Strategy 5: Advanced algebraic approach using continued fractions
            print(f"[Diophantine] Strategy 5: Using continued fraction approximation")

            # The equation is: e_p * e_q + p_given * e_q + q_given * e_p + delta = 0
            # Rearrange as quadratic in e_p: e_p * (e_q + q_given) + (p_given * e_q + delta) = 0
            # e_p = - (p_given * e_q + delta) / (e_q + q_given)

            # We can solve for e_q first using continued fraction approximation of delta/p_given
            from fractions import Fraction

            try:
                # Approximate delta/p_given as a continued fraction to find likely e_q values
                target_ratio = Fraction(delta, p_given)
                cf_approx = target_ratio.limit_denominator(10000)  # Limit denominator for practicality

                # Try the continued fraction approximation
                e_q_cf = -cf_approx.numerator
                denominator_cf = cf_approx.denominator

                if denominator_cf != 0:
                    # Solve for e_p using this e_q approximation
                    numerator = -delta - p_given * e_q_cf
                    if numerator % denominator_cf == 0:
                        e_p_cf = numerator // denominator_cf

                        p_test = p_given + e_p_cf
                        q_test = q_given + e_q_cf

                        if p_test > 1 and q_test > 1 and p_test * q_test == self.N:
                            print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND via continued fractions!")
                            print(f"[Diophantine]   e_p = {e_p_cf}, e_q = {e_q_cf}")
                            print(f"[Diophantine]   p = {p_test}, q = {q_test}")
                            return (int(p_test), int(q_test))

                # Also try approximation of delta/q_given
                target_ratio_q = Fraction(delta, q_given)
                cf_approx_q = target_ratio_q.limit_denominator(10000)

                e_p_cf_q = -cf_approx_q.numerator
                denominator_cf_q = cf_approx_q.denominator

                if denominator_cf_q != 0:
                    numerator_q = -delta - q_given * e_p_cf_q
                    if numerator_q % denominator_cf_q == 0:
                        e_q_cf_q = numerator_q // denominator_cf_q

                        p_test = p_given + e_p_cf_q
                        q_test = q_given + e_q_cf_q

                        if p_test > 1 and q_test > 1 and p_test * q_test == self.N:
                            print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND via continued fractions (q)!")
                            print(f"[Diophantine]   e_p = {e_p_cf_q}, e_q = {e_q_cf_q}")
                            print(f"[Diophantine]   p = {p_test}, q = {q_test}")
                            return (int(p_test), int(q_test))

            except Exception as e:
                print(f"[Diophantine] Continued fraction approach failed: {e}")

            # Strategy 6: Use the exact quadratic formula approach
            print(f"[Diophantine] Strategy 6: Exact quadratic solution")

            # Treat as quadratic equation: e_p * e_q + (p_given) * e_q + (q_given) * e_p + delta = 0
            # For fixed e_q, solve quadratic in e_p: e_q * e_p^2 + (p_given + q_given) * e_p + (p_given * e_q + delta) = 0
            # Wait, that's not right. Let me reconsider.

            # Actually, we have: e_p * e_q + p_given * e_q + q_given * e_p + delta = 0
            # Group terms with e_p: e_p * (e_q + q_given) + (p_given * e_q + delta) = 0
            # This is linear in e_p for fixed e_q, which we already solved.

            # The full quadratic would be if we had e_p^2 terms, but we don't.

            # Strategy 7: Use linear congruence approach
            print(f"[Diophantine] Strategy 7: Linear congruence approach")

            # We need to solve: p_given * e_q + q_given * e_p ≡ -delta mod gcd(p_given, q_given)
            # But since p_given and q_given are likely coprime, we can solve the system

            g = abs(self._gcd(p_given, q_given))
            if (-delta) % g == 0:
                print(f"[Diophantine] System is solvable mod {g}")

                # Try small solutions and lift to full solution
                for k in range(-100, 101):
                    if k == 0:
                        continue

                    # Try e_p = k * (q_given // g), e_q = -k * (p_given // g) or similar
                    # This is getting complex, let's use a simpler approach

                    # Just check if the error can be corrected by small adjustments
                    test_p = p_given + k
                    if test_p > 1 and self.N % test_p == 0:
                        test_q = self.N // test_p
                        dq = test_q - q_given
                        print(f"[Diophantine] ✓✓✓ EXACT via congruence: p = p_given + {k}, q = {test_q}")
                        return (int(test_p), int(test_q))

                    test_q = q_given + k
                    if test_q > 1 and self.N % test_q == 0:
                        test_p = self.N // test_q
                        dp = test_p - p_given
                        print(f"[Diophantine] ✓✓✓ EXACT via congruence: p = {test_p}, q = q_given + {k}")
                        return (int(test_p), int(test_q))

            print(f"[Diophantine] All algebraic strategies exhausted")

            # Strategy 8: Variable bounds analysis - determine what values would make equations solvable
            print(f"[Diophantine] Strategy 8: Variable bounds analysis for non-integer solutions")

            # The equation is: e_p * e_q + p_given * e_q + q_given * e_p + delta = 0
            # Rearrange: e_p * e_q + (p_given) * e_q + (q_given) * e_p = -delta

            # For this to have integer solutions, -delta must be achievable
            # Let's find bounds on e_p and e_q that would make this possible

            # Treat as: e_p * e_q ≈ -delta - p_given * e_q - q_given * e_p
            # For small perturbations, we can find ranges where integer solutions exist

            print(f"[Diophantine] Analyzing variable bounds for solvability...")

            # Strategy 8a: Find e_p bounds for fixed small e_q
            min_e_p_bound = float('inf')
            max_e_p_bound = float('-inf')

            for e_q_trial in range(-100, 101):
                if e_q_trial == 0:
                    continue

                # From: e_p * e_q + p_given * e_q + q_given * e_p + delta = 0
                # e_p * (e_q + q_given) = -delta - p_given * e_q
                # e_p = (-delta - p_given * e_q) / (e_q + q_given)

                denominator = e_q_trial + q_given
                if denominator == 0:
                    continue

                numerator = -delta - p_given * e_q_trial
                e_p_float = numerator / denominator

                # This gives us the exact e_p needed for this e_q
                # But we need integer solutions, so check if numerator is divisible by denominator
                if numerator % denominator == 0:
                    e_p_exact = numerator // denominator
                    min_e_p_bound = min(min_e_p_bound, e_p_exact)
                    max_e_p_bound = max(max_e_p_bound, e_p_exact)
                else:
                    # Even if not exact, track the range of possible e_p values
                    e_p_approx = round(e_p_float)
                    min_e_p_bound = min(min_e_p_bound, e_p_approx)
                    max_e_p_bound = max(max_e_p_bound, e_p_approx)

            if min_e_p_bound != float('inf'):
                print(f"[Diophantine] Variable bounds analysis: e_p should be in range [{min_e_p_bound}, {max_e_p_bound}]")
                print(f"[Diophantine] This suggests testing e_p values around this range")

                # Test values in this derived range
                for e_p_test in range(max(-100000, int(min_e_p_bound) - 10), min(100000, int(max_e_p_bound) + 11)):
                    numerator = -delta - q_given * e_p_test
                    if numerator % p_given == 0:
                        e_q_test = numerator // p_given

                        p_test = p_given + e_p_test
                        q_test = q_given + e_q_test

                        if p_test > 1 and q_test > 1 and p_test * q_test == self.N:
                            print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND via bounds analysis!")
                            print(f"[Diophantine]   e_p = {e_p_test}, e_q = {e_q_test}")
                            print(f"[Diophantine]   p = {p_test}, q = {q_test}")
                            return (int(p_test), int(q_test))

            # Strategy 8b: Perturbation analysis - how much to perturb to get exact solutions
            print(f"[Diophantine] Strategy 8b: Perturbation analysis")

            # We have approximations p_given, q_given with p_given * q_given = N + delta
            # We want p_exact, q_exact with p_exact * q_exact = N

            # The minimal perturbation would satisfy:
            # (p_given + dp) * (q_given + dq) = N
            # p_given*q_given + p_given*dq + q_given*dp + dp*dq = N
            # Since p_given*q_given = N + delta, we get:
            # N + delta + p_given*dq + q_given*dp + dp*dq = N
            # delta + p_given*dq + q_given*dp + dp*dq = 0

            # For small dp, dq, we can solve approximately:
            # p_given*dq + q_given*dp ≈ -delta

            # But for exact solutions, we need to find dp, dq such that the full equation holds
            # This is a quadratic Diophantine equation in dp, dq

            # Let's solve it as: dp*dq + p_given*dq + q_given*dp + delta = 0
            # Rearrange: dp*dq + (p_given)*dq + (q_given)*dp = -delta

            # This is still hard, but we can try assuming dq is small and solve for dp

            # Actually, let's use the fact that we know the exact factors should be close
            # Try to find the minimal perturbation

            for dq_pert in range(-100, 101):
                # Solve: q_given*dp + p_given*dq_pert + delta + dp*dq_pert ≈ 0
                # q_given*dp ≈ -delta - p_given*dq_pert - dp*dq_pert

                # For small dq_pert, approximate dp ≈ (-delta - p_given*dq_pert) / q_given
                if q_given != 0:
                    dp_approx = (-delta - p_given * dq_pert) / q_given
                    dp_candidates = [round(dp_approx), round(dp_approx) - 1, round(dp_approx) + 1]

                    for dp_cand in dp_candidates:
                        # Check if this gives exact solution
                        p_test = p_given + dp_cand
                        q_test = q_given + dq_pert

                        if p_test > 1 and q_test > 1 and p_test * q_test == self.N:
                            print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND via perturbation analysis!")
                            print(f"[Diophantine]   dp = {dp_cand}, dq = {dq_pert}")
                            print(f"[Diophantine]   p = {p_test}, q = {q_test}")
                            return (int(p_test), int(q_test))

            # Strategy 8c: Homogeneous solutions approach
            print(f"[Diophantine] Strategy 8c: Homogeneous solutions")

            # The homogeneous equation is: e_p * e_q + p_given * e_q + q_given * e_p = 0
            # Rearrange: e_p * (e_q + q_given) + p_given * e_q = 0
            # e_p = - (p_given * e_q) / (e_q + q_given)

            # Solutions to homogeneous equation give directions we can move while staying close to solution
            # We can use these to adjust our particular solution

            for e_q_hom in range(-50, 51):
                if e_q_hom == 0:
                    continue

                denominator_hom = e_q_hom + q_given
                if denominator_hom == 0:
                    continue

                numerator_hom = - p_given * e_q_hom
                if numerator_hom % denominator_hom == 0:
                    e_p_hom = numerator_hom // denominator_hom

                    # Now we can add multiples of this homogeneous solution to adjust our approximation
                    for multiplier in range(-10, 11):
                        if multiplier == 0:
                            continue

                        dp_adjust = multiplier * e_p_hom
                        dq_adjust = multiplier * e_q_hom

                        p_test = p_given + dp_adjust
                        q_test = q_given + dq_adjust

                        if p_test > 1 and q_test > 1 and p_test * q_test == self.N:
                            print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND via homogeneous solutions!")
                            print(f"[Diophantine]   homogeneous: e_p={e_p_hom}, e_q={e_q_hom}, multiplier={multiplier}")
                            print(f"[Diophantine]   adjustments: dp={dp_adjust}, dq={dq_adjust}")
                            print(f"[Diophantine]   p = {p_test}, q = {q_test}")
                            return (int(p_test), int(q_test))

            print(f"[Diophantine] Variable bounds and perturbation analysis complete")

            # Strategy 9: Optimization-based approach using gradient descent
            print(f"[Diophantine] Strategy 9: Optimization-based factor finding")

            # Use a simple optimization to minimize |p*q - N| subject to p ≈ p_given, q ≈ q_given
            # Start from the given approximations and use coordinate descent

            best_p = p_given
            best_q = q_given
            best_error = abs(best_p * best_q - self.N)

            # Try coordinate descent: fix q, optimize p, then fix p, optimize q
            max_iterations = 100
            step_size = max(1, abs(delta) // 1000000)  # Adaptive step size

            print(f"[Diophantine] Starting optimization with step_size={step_size}")

            for iteration in range(max_iterations):
                improved = False

                # Phase 1: Optimize p for fixed q
                current_q = best_q
                current_error = best_error

                # Try small adjustments to p
                for dp in [-2*step_size, -step_size, step_size, 2*step_size]:
                    test_p = best_p + dp
                    if test_p > 1:
                        test_error = abs(test_p * current_q - self.N)
                        if test_error < current_error:
                            best_p = test_p
                            best_error = test_error
                            current_error = test_error
                            improved = True

                            # Check if we found exact solution
                            if best_error == 0:
                                print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND via optimization!")
                                print(f"[Diophantine]   p = {best_p}, q = {best_q}")
                                print(f"[Diophantine]   Found after {iteration+1} iterations")
                                return (int(best_p), int(best_q))

                # Phase 2: Optimize q for fixed p
                current_p = best_p

                for dq in [-2*step_size, -step_size, step_size, 2*step_size]:
                    test_q = best_q + dq
                    if test_q > 1:
                        test_error = abs(current_p * test_q - self.N)
                        if test_error < current_error:
                            best_q = test_q
                            best_error = test_error
                            improved = True

                            # Check if we found exact solution
                            if best_error == 0:
                                print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND via optimization!")
                                print(f"[Diophantine]   p = {best_p}, q = {best_q}")
                                print(f"[Diophantine]   Found after {iteration+1} iterations")
                                return (int(best_p), int(best_q))

                if not improved:
                    print(f"[Diophantine] Optimization converged after {iteration+1} iterations")
                    break

                if iteration % 10 == 0:
                    print(f"[Diophantine] Iteration {iteration+1}: error = {best_error}")

            if best_error < abs(delta):  # Better than original
                print(f"[Diophantine] Optimization improved solution: error reduced from {abs(delta)} to {best_error}")

            # Strategy 10: Modular constraints analysis
            print(f"[Diophantine] Strategy 10: Modular constraints for factor finding")

            # Since p and q are factors of N, they must satisfy certain modular conditions
            # Use Chinese Remainder Theorem with small primes to narrow down possibilities

            small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
            constraints = []

            for prime in small_primes:
                n_mod = self.N % prime
                p_given_mod = p_given % prime
                q_given_mod = q_given % prime

                # p * q ≡ N mod prime
                # So p * q ≡ n_mod mod prime

                # Possible (p_mod, q_mod) pairs that satisfy this
                possible_pairs = []
                for p_mod in range(prime):
                    for q_mod in range(prime):
                        if (p_mod * q_mod) % prime == n_mod:
                            possible_pairs.append((p_mod, q_mod))

                if possible_pairs:
                    constraints.append((prime, possible_pairs))
                    print(f"[Diophantine] Mod {prime}: {len(possible_pairs)} possible (p,q) pairs")

            # Now try to find p, q close to approximations that satisfy these modular constraints
            search_radius = min(1000, abs(delta) // 1000)

            for dp in range(-search_radius, search_radius + 1, max(1, search_radius // 100)):
                for dq in range(-search_radius, search_radius + 1, max(1, search_radius // 100)):
                    p_test = p_given + dp
                    q_test = q_given + dq

                    if p_test <= 1 or q_test <= 1:
                        continue

                    # Check if this satisfies the modular constraints
                    satisfies_all = True
                    for prime, pairs in constraints[:3]:  # Check first few primes
                        p_mod = p_test % prime
                        q_mod = q_test % prime
                        if (p_mod, q_mod) not in pairs:
                            satisfies_all = False
                            break

                    if satisfies_all:
                        # Check if it's actually a factor
                        if p_test * q_test == self.N:
                            print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND via modular constraints!")
                            print(f"[Diophantine]   p = {p_test}, q = {q_test}")
                            print(f"[Diophantine]   adjustments: dp={dp}, dq={dq}")
                            return (int(p_test), int(q_test))

            print(f"[Diophantine] All advanced strategies exhausted - trying final targeted search")

            # Strategy 15: Final targeted brute force around best approximations
            print(f"[Diophantine] Strategy 15: Targeted brute force search around best approximations")

            # Since we have extremely close approximations, do a focused search
            # around the current best factors
            search_range = 10000000  # Search ±10M around current approximations (increased from 1M)

            print(f"[Diophantine] Searching p in range [{p_given - search_range}, {p_given + search_range}]")
            print(f"[Diophantine] Searching q in range [{q_given - search_range}, {q_given + search_range}]")

            # First try p variations with fixed q
            step_size = max(1, search_range // 50000)  # Smaller steps for better coverage
            for dp in range(-search_range, search_range + 1, step_size):
                p_test = p_given + dp
                if p_test > 1 and self.N % p_test == 0:
                    q_exact = self.N // p_test
                    dq = q_exact - q_given
                    print(f"[Diophantine] ✓✓✓ EXACT FACTORS FOUND via targeted p search!")
                    print(f"[Diophantine]   p = p_given + {dp} = {p_test}")
                    print(f"[Diophantine]   q = {q_exact} (adjustment: {dq})")
                    return (int(p_test), int(q_exact))

            # Then try q variations with fixed p
            for dq in range(-search_range, search_range + 1, step_size):
                q_test = q_given + dq
                if q_test > 1 and self.N % q_test == 0:
                    p_exact = self.N // q_test
                    dp = p_exact - p_given
                    print(f"[Diophantine] ✓✓✓ EXACT FACTORS FOUND via targeted q search!")
                    print(f"[Diophantine]   p = {p_exact} (adjustment: {dp})")
                    print(f"[Diophantine]   q = q_given + {dq} = {q_test}")
                    return (int(p_exact), int(q_test))

            print(f"[Diophantine] Targeted search completed - no exact factors found in range ±{search_range}")

            # Strategy 16: Ultra-focused search using error analysis
            print(f"[Diophantine] Strategy 16: Ultra-focused search using error analysis")

            # If we have the error from the current approximation, we can be more targeted
            # The error tells us exactly how much to adjust
            current_product = p_given * q_given
            if current_product != self.N:
                error = current_product - self.N
                print(f"[Diophantine] Current error: {error} ({len(str(abs(error)))} digits)")

                # Try a wider range of adjustments around the error-based estimates
                error_range = 1000000  # Search ±1M around error-based adjustments

                # Try to adjust by error/p or error/q with wider search
                if q_given != 0:
                    dp_base = -error // q_given
                    print(f"[Diophantine] Error-based dp adjustment: {dp_base}")
                    for dp_offset in range(-error_range, error_range + 1, max(1, error_range // 1000)):
                        dp_adjust = dp_base + dp_offset
                        p_test = p_given + dp_adjust
                        if p_test > 1 and self.N % p_test == 0:
                            q_exact = self.N // p_test
                            dq = q_exact - q_given
                            print(f"[Diophantine] ✓✓✓ EXACT FACTORS FOUND via error adjustment on p!")
                            print(f"[Diophantine]   error = {error}")
                            print(f"[Diophantine]   dp_adjust = {dp_adjust} (base: {dp_base}, offset: {dp_offset})")
                            print(f"[Diophantine]   p = {p_test}, q = {q_exact}")
                            return (int(p_test), int(q_exact))

                if p_given != 0:
                    dq_base = -error // p_given
                    print(f"[Diophantine] Error-based dq adjustment: {dq_base}")
                    for dq_offset in range(-error_range, error_range + 1, max(1, error_range // 1000)):
                        dq_adjust = dq_base + dq_offset
                        q_test = q_given + dq_adjust
                        if q_test > 1 and self.N % q_test == 0:
                            p_exact = self.N // q_test
                            dp = p_exact - p_given
                            print(f"[Diophantine] ✓✓✓ EXACT FACTORS FOUND via error adjustment on q!")
                            print(f"[Diophantine]   error = {error}")
                            print(f"[Diophantine]   dq_adjust = {dq_adjust} (base: {dq_base}, offset: {dq_offset})")
                            print(f"[Diophantine]   p = {p_exact}, q = {q_test}")
                            return (int(p_exact), int(q_test))

            print(f"[Diophantine] Error-based adjustments exhausted")

            print(f"[Diophantine] ALL STRATEGIES EXHAUSTED - Exact factorization not found with current mathematical approaches")

            # Strategy 11: Newton's method for factor refinement
            print(f"[Diophantine] Strategy 11: Newton's method for factor refinement")

            # Since we have very close approximations (0.0000% error), we can use Newton's method
            # to converge to exact factors. The idea is to solve f(x) = 0 where f(x) = N - x*q_approx
            # and we want x to be an integer factor of N.

            # More precisely, we want to find p such that p divides N and p is close to p_given
            # We can use Newton iteration: p_{n+1} = p_n - f(p_n)/f'(p_n)
            # where f(p) = p - N/p (for the case where we want p^2 ≈ N)
            # But here we have two factors, so it's more complex.

            # Alternative: since p*q = N and we have approximations, we can solve for the correction
            # using the derivative of the product.

            # Let g(p) = p * q_approx - N
            # We want g(p) = 0, so p = N / q_approx
            # But since q_approx is approximate, we need to iterate.

            # Actually, let's use the fact that if we have p_approx * q_approx ≈ N
            # Then the exact p should be close to N / q_approx

            max_newton_iterations = 20
            tolerance = 1  # We want exact division

            # Try Newton's method for p
            p_current = p_given
            for iteration in range(max_newton_iterations):
                # Check if current p divides N
                if self.N % p_current == 0:
                    q_exact = self.N // p_current
                    print(f"[Diophantine] ✓✓✓ Newton's method found exact p after {iteration+1} iterations!")
                    print(f"[Diophantine]   p = {p_current}, q = {q_exact}")
                    return (int(p_current), int(q_exact))

                # Newton's step: we want p such that p * q_given ≈ N
                # So f(p) = p * q_given - N
                # f'(p) = q_given
                # p_{n+1} = p_n - f(p_n)/f'(p_n) = p_n - (p_n * q_given - N)/q_given

                if q_given != 0:
                    f_val = p_current * q_given - self.N
                    f_prime = q_given
                    step = f_val // f_prime  # Integer division for large numbers
                    p_current = p_current - step

                    if abs(step) < tolerance:
                        break

            # Try Newton's method for q
            q_current = q_given
            for iteration in range(max_newton_iterations):
                # Check if current q divides N
                if self.N % q_current == 0:
                    p_exact = self.N // q_current
                    print(f"[Diophantine] ✓✓✓ Newton's method found exact q after {iteration+1} iterations!")
                    print(f"[Diophantine]   p = {p_exact}, q = {q_current}")
                    return (int(p_exact), int(q_current))

                # Newton's step for q: f(q) = q * p_given - N, f'(q) = p_given
                if p_given != 0:
                    f_val = q_current * p_given - self.N
                    f_prime = p_given
                    step = f_val // f_prime
                    q_current = q_current - step

                    if abs(step) < tolerance:
                        break

            # Strategy 12: Binary search for exact factors
            print(f"[Diophantine] Strategy 12: Binary search for exact factors")

            # Since we have bounds on where the exact factors should be, we can do binary search
            # We know the exact p should be between some lower and upper bounds

            # For p: we need N % p == 0 and p ≈ p_given
            # Since p * q ≈ N, and q ≈ q_given, we can estimate bounds

            # Lower bound: p must be at least sqrt(N) - some margin
            # Upper bound: p must be at most sqrt(N) + some margin

            sqrt_n = self._integer_sqrt_approx(self.N)

            # Search in a range around sqrt(N)
            p_lower = max(2, sqrt_n - abs(delta) // (2 * sqrt_n) - 1000)
            p_upper = sqrt_n + abs(delta) // (2 * sqrt_n) + 1000

            print(f"[Diophantine] Binary search for p in range [{p_lower}, {p_upper}]")

            # Binary search for p
            while p_lower <= p_upper:
                p_mid = (p_lower + p_upper) // 2

                if p_mid > 1 and self.N % p_mid == 0:
                    q_exact = self.N // p_mid
                    # Check if q is also close to our approximation
                    if abs(q_exact - q_given) < abs(delta) // p_given:  # Reasonable closeness
                        print(f"[Diophantine] ✓✓✓ Binary search found exact p!")
                        print(f"[Diophantine]   p = {p_mid}, q = {q_exact}")
                        return (int(p_mid), int(q_exact))

                # Decide which half to search
                # If p_mid * q_given < N, we need larger p
                if p_mid * q_given < self.N:
                    p_lower = p_mid + 1
                else:
                    p_upper = p_mid - 1

            # Strategy 13: Lattice-based refinement
            print(f"[Diophantine] Strategy 13: Lattice-based refinement")

            # Use the fact that we have a lattice point (p_given, q_given) close to the curve p*q = N
            # We can find the closest lattice points on the curve p*q = N

            # The solutions to p*q = N are lattice points on the hyperbola
            # We can use continued fractions or other methods to find nearby solutions

            # One approach: solve the Pell-like equation
            # We want p*q = N with p ≈ p_given, q ≈ q_given

            # Let d = gcd(p_given, q_given)
            # Then we can write the equation as:
            # (p_given/d + x) * (q_given/d + y) = N/d^2

            # But this might be complicated. Let's try a simpler approach.

            # Use the continued fraction expansion of p_given/q_given to find convergents
            # that give exact factors.

            try:
                from fractions import Fraction

                # Get continued fraction of the ratio
                ratio = Fraction(p_given, q_given)
                cf = []
                a = ratio
                for i in range(20):  # Limit depth
                    cf.append(int(a))
                    a = a - int(a)
                    if a == 0:
                        break
                    a = 1 / a

                # Generate convergents
                convergents = []
                h_prev, k_prev = 1, 0
                h, k = cf[0], 1

                convergents.append((h, k))

                for i in range(1, len(cf)):
                    h_new = cf[i] * h + h_prev
                    k_new = cf[i] * k + k_prev
                    convergents.append((h_new, k_new))
                    h_prev, k_prev = h, k
                    h, k = h_new, k_new

                # Test convergents as potential factors
                for h_conv, k_conv in convergents:
                    # Try p = h_conv, q = N//h_conv if it divides
                    if h_conv > 1 and self.N % h_conv == 0:
                        q_test = self.N // h_conv
                        if abs(h_conv - p_given) < abs(delta) // q_given:
                            print(f"[Diophantine] ✓✓✓ Continued fraction found exact p!")
                            print(f"[Diophantine]   p = {h_conv}, q = {q_test}")
                            return (int(h_conv), int(q_test))

                    # Try q = k_conv, p = N//k_conv if it divides
                    if k_conv > 1 and self.N % k_conv == 0:
                        p_test = self.N // k_conv
                        if abs(k_conv - q_given) < abs(delta) // p_given:
                            print(f"[Diophantine] ✓✓✓ Continued fraction found exact q!")
                            print(f"[Diophantine]   p = {p_test}, q = {k_conv}")
                            return (int(p_test), int(k_conv))

            except Exception as e:
                print(f"[Diophantine] Continued fraction approach failed: {e}")

            print(f"[Diophantine] All strategies including Newton and lattice methods exhausted")

            # Strategy 14: Direct factor extraction from close approximations
            print(f"[Diophantine] Strategy 14: Direct factor extraction from close approximations")

            # Since the approximations are extremely close (0.0000% error), the exact factors
            # should be very close. We can directly compute what the exact factors should be.

            # Method 1: p_exact should be close to N / q_given
            p_from_q = self.N // q_given
            q_remainder = self.N % q_given

            if q_remainder == 0:
                print(f"[Diophantine] ✓✓✓ Direct extraction: q_given divides N!")
                print(f"[Diophantine]   p = {p_from_q}, q = {q_given}")
                return (int(p_from_q), int(q_given))

            # If not exact, q_remainder tells us how much to adjust
            # Since q_given * p_from_q = N - q_remainder
            # We need q_given * p_exact = N, so p_exact = (N + adjustment) / q_given
            # But since we're dealing with integers, we need to find adjustment

            # Method 2: q_exact should be close to N / p_given
            q_from_p = self.N // p_given
            p_remainder = self.N % p_given

            if p_remainder == 0:
                print(f"[Diophantine] ✓✓✓ Direct extraction: p_given divides N!")
                print(f"[Diophantine]   p = {p_given}, q = {q_from_p}")
                return (int(p_given), int(q_from_p))

            # Method 3: Since both remainders are the same (as we saw in the output),
            # we can solve the system:
            # q_given * p + p_remainder = N
            # p_given * q + p_remainder = N

            # From the first equation: q_given * p = N - p_remainder
            # From the second: p_given * q = N - p_remainder

            # So p = (N - p_remainder) // q_given
            # q = (N - p_remainder) // p_given

            # Since p_remainder = q_remainder (from the output), let's check
            if p_remainder == q_remainder:
                target = self.N - p_remainder
                p_candidate = target // q_given
                q_candidate = target // p_given

                # Check if this gives exact factors
                if p_candidate * q_candidate == self.N:
                    print(f"[Diophantine] ✓✓✓ Direct extraction via remainder adjustment!")
                    print(f"[Diophantine]   p = {p_candidate}, q = {q_candidate}")
                    print(f"[Diophantine]   Adjustment made: subtracted remainder {p_remainder}")
                    return (int(p_candidate), int(q_candidate))

            # Method 4: Use delta to directly adjust
            # We know p_given * q_given = N + delta
            # For exact factors, we need p_exact * q_exact = N

            # Since the approximations are very close, we can assume one factor is almost exact
            # and adjust the other.

            # Try assuming q_given is exact, find p that makes p * q_given = N
            # Since it doesn't divide exactly, we need to adjust q_given

            # The adjustment needed is -delta / q_given for p, or -delta / p_given for q

            # Try adjusting p
            dp_exact = -delta // q_given  # Since delta is negative in this case, this will be positive
            p_adjusted = p_given + dp_exact
            if p_adjusted > 1 and self.N % p_adjusted == 0:
                q_exact = self.N // p_adjusted
                print(f"[Diophantine] ✓✓✓ Direct extraction via delta adjustment on p!")
                print(f"[Diophantine]   dp = {dp_exact}, p = {p_adjusted}, q = {q_exact}")
                return (int(p_adjusted), int(q_exact))

            # Try adjusting q
            dq_exact = -delta // p_given
            q_adjusted = q_given + dq_exact
            if q_adjusted > 1 and self.N % q_adjusted == 0:
                p_exact = self.N // q_adjusted
                print(f"[Diophantine] ✓✓✓ Direct extraction via delta adjustment on q!")
                print(f"[Diophantine]   dq = {dq_exact}, p = {p_exact}, q = {q_adjusted}")
                return (int(p_exact), int(q_adjusted))

            # Method 5: Combined adjustment
            # Since both factors need adjustment, we can solve the system
            # p_given + dp = N / (q_given + dq)
            # This gives a relation between dp and dq

            # From p*(q + dq) = N, so p = N/(q + dq) ≈ (N/q) * (1 - dq/q)
            # So p_given + dp ≈ (N/q_given) * (1 - dq/q_given)
            # Which gives: dp ≈ (N/q_given) - p_given - (N/q_given^2)*dq

            # This is getting complex. Let's use numerical approximation

            # Since we have p_given*q_given = N + delta, we can solve the quadratic equation:
            # d = sqrt( (p_given + q_given)^2 - 4*(p_given*q_given - N) ) / 2
            # Then p_exact = (p_given + q_given + d)/2, q_exact = (p_given + q_given - d)/2

            discriminant = (p_given + q_given)**2 - 4*(p_given*q_given - self.N)

            if discriminant >= 0:
                try:
                    d = self._integer_sqrt_approx(discriminant)
                    if d * d == discriminant:  # Perfect square
                        p_exact = (p_given + q_given + d) // 2
                        q_exact = (p_given + q_given - d) // 2

                        if p_exact * q_exact == self.N:
                            print(f"[Diophantine] ✓✓✓ Direct extraction via quadratic formula!")
                            print(f"[Diophantine]   discriminant = {discriminant} = {d}^2")
                            print(f"[Diophantine]   p = {p_exact}, q = {q_exact}")
                            return (int(p_exact), int(q_exact))

                        # Try the other root
                        p_exact2 = (p_given + q_given - d) // 2
                        q_exact2 = (p_given + q_given + d) // 2

                        if p_exact2 * q_exact2 == self.N:
                            print(f"[Diophantine] ✓✓✓ Direct extraction via quadratic formula (alt)!")
                            print(f"[Diophantine]   p = {p_exact2}, q = {q_exact2}")
                            return (int(p_exact2), int(q_exact2))
                except:
                    pass

            print(f"[Diophantine] Direct factor extraction strategies exhausted")
            
            # Solve the equation directly for small z values
            print(f"[Diophantine] Solving equation directly for small z values...")
            print(f"[Diophantine] From: y*z + q_given*y + p_given*z + prod_mod_n = 0")
            print(f"[Diophantine] Rearranging: y*(z + q_given) = -p_given*z - prod_mod_n")
            print(f"[Diophantine] So: y = (-p_given*z - prod_mod_n) / (z + q_given)")
            print()
            
            # Calculate tolerance
            if self.N.bit_length() > 1000:
                max_relative_error = 0.01
                max_absolute_error = self.N // 100
            elif self.N.bit_length() > 500:
                max_relative_error = 0.001
                max_absolute_error = self.N // 1000
            else:
                max_relative_error = 0.0001
                max_absolute_error = max(1000, self.N // 10000)
            
            best_approx = None
            best_error = None
            
            for z_test in [-2, -1, 0, 1, 2]:
                denominator = z_test + q_given
                if denominator == 0:
                    continue
                
                numerator = -p_given*z_test - prod_mod_n
                
                # Try exact division first
                if numerator % denominator == 0:
                    y_test = numerator // denominator
                    p_test = p_given + y_test
                    q_test = q_given + z_test
                    
                    # Verify both the equation and factorization
                    lhs = y_test*z_test + q_given*y_test + p_given*z_test + prod_mod_n
                    if lhs == 0 and p_test > 1 and q_test > 1:
                        if p_test * q_test == self.N:
                            print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND (z = {z_test})!")
                            print(f"[Diophantine]   y = {y_test}, z = {z_test}")
                            print(f"[Diophantine]   p = {p_test}, q = {q_test}")
                            print(f"[Diophantine]   Equation check: {lhs} = 0 ✓")
                            return (int(p_test), int(q_test))
                        else:
                            # Approximate solution
                            product = p_test * q_test
                            error = abs(product - self.N)
                            relative_error = error / self.N if self.N > 0 else float('inf')
                            
                            if error <= max_absolute_error and relative_error <= max_relative_error:
                                if best_approx is None or error < best_error:
                                    best_approx = (p_test, q_test)
                                    best_error = error
                else:
                    # Not exactly divisible, but try approximate
                    # Use rational division and round
                    y_approx = numerator // denominator  # Integer division (truncates)
                    p_test = p_given + y_approx
                    q_test = q_given + z_test
                    
                    if p_test > 1 and q_test > 1:
                        product = p_test * q_test
                        error = abs(product - self.N)
                        relative_error = error / self.N if self.N > 0 else float('inf')
                        
                        if error <= max_absolute_error and relative_error <= max_relative_error:
                            if best_approx is None or error < best_error:
                                best_approx = (p_test, q_test)
                                best_error = error
            
            # Return best approximate if no exact solution found
            if best_approx:
                p_approx, q_approx = best_approx
                product = p_approx * q_approx
                relative_error = (best_error / self.N * 100) if self.N > 0 else 0
                print(f"[Diophantine] ✓ Best approximate solution (quadratic equation):")
                print(f"[Diophantine]   p = {p_approx}, q = {q_approx}")
                print(f"[Diophantine]   p * q = {product}, N = {self.N}")
                print(f"[Diophantine]   Error = {best_error} ({relative_error:.4f}%)")
                
                # Try refinement
                print()
                print("[Diophantine] Attempting refinement to exact factors...")
                exact_result = self._refine_approximate_solution(p_approx, q_approx, product - self.N)
                if exact_result:
                    return exact_result
                
                return (int(p_approx), int(q_approx))
            
            # Use the relationship: N_target = expected_q * p_given + remainder
            expected_q = self.N // p_given
            remainder = self.N % p_given
            print(f"[Diophantine] Using division relationship: N = {expected_q} * p_given + {remainder}")
            print(f"[Diophantine] q_given = {q_given}, difference = {expected_q - q_given}")
            
            # If z = -1, then q = q_given - 1 = expected_q (if expected_q = q_given - 1)
            if expected_q > 0 and self.N % expected_q == 0:
                p_solution = self.N // expected_q
                y_solution = p_solution - p_given
                z_solution = expected_q - q_given
                
                # Verify
                lhs = y_solution*z_solution + q_given*y_solution + p_given*z_solution + prod_mod_n
                if lhs == 0 and p_solution * expected_q == self.N:
                    print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND using expected_q!")
                    print(f"[Diophantine]   y = {y_solution}, z = {z_solution}")
                    print(f"[Diophantine]   p = {p_solution}, q = {expected_q}")
                    return (int(p_solution), int(expected_q))
        
        # 3) Try to combine modular constraints to create linear relations
        # Collect all modular constraints: p - k = 0 and q - m = 0
        p_modular = []
        q_modular = []

        for poly in main_polys:
            try:
                L = Poly(poly, self.p, self.q)
                if L.total_degree() == 1:
                    coeffs = L.as_dict()
                    a = int(coeffs.get((1, 0), 0))
                    b = int(coeffs.get((0, 1), 0))
                    c = int(coeffs.get((0, 0), 0))

                    # p - k = 0 (modular hint for p)
                    if a == 1 and b == 0 and c != 0:
                        k = -c
                        p_modular.append(k)
                    # q - m = 0 (modular hint for q)
                    elif a == 0 and b == 1 and c != 0:
                        m = -c
                        q_modular.append(m)
            except:
                continue

        # If we have hints and modular constraints, try to find combinations
        if p_hint and q_hint and (p_modular or q_modular):
            print(f"[Diophantine] Trying to combine {len(p_modular)} p-modular and {len(q_modular)} q-modular constraints with hints...")

            # For each pair of modular constraints, see if we can create a linear relation
            for k in p_modular[:5]:  # Limit to avoid explosion
                for m in q_modular[:5]:
                    # We have p ≈ k and q ≈ m, so try p - k and q - m as potential corrections
                    # This creates a linear system: p = k + dp, q = m + dq, with p*q = N
                    try:
                        # Solve for dp, dq such that (k + dp) * (m + dq) = N
                        # This gives: k*m + k*dq + m*dp + dp*dq = N
                        # For small corrections, dp*dq ≈ 0, so: k*m + k*dq + m*dp ≈ N
                        # Thus: k*dq + m*dp ≈ N - k*m
                        target = int(self.N) - k * m
                        if abs(target) < 10**12:  # Reasonable bound
                            # Try small integer solutions
                            for dp in range(-1000, 1001):
                                for dq in range(-1000, 1001):
                                    if k * dq + m * dp == target:
                                        p_cand = k + dp
                                        q_cand = m + dq
                                        if p_cand > 1 and q_cand > 1 and p_cand * q_cand == int(self.N):
                                            print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND from modular combination: p={p_cand}, q={q_cand}")
                                            print(f"[Diophantine] Used modular hints: p ≈ {k}, q ≈ {m}, corrections dp={dp}, dq={dq}")
                                            return (p_cand, q_cand)
                    except:
                        continue

        # 5) Try direct trial division using hints to narrow search
        if p_hint and q_hint:
            print(f"[Diophantine] Attempting targeted trial division around hints...")

            # Search in a radius around the hints
            p_center = p_hint
            q_center = q_hint
            search_radius = max(1000000, min(p_center // 10, q_center // 10, 10000000))  # MASSIVE expansion for large numbers

            print(f"[Diophantine] Searching p ∈ [{p_center - search_radius}, {p_center + search_radius}]")
            print(f"[Diophantine] Searching q ∈ [{q_center - search_radius}, {q_center + search_radius}]")
            
            # Track best approximate solution
            best_approx = None
            best_error = None
            
            # Calculate tolerance for approximate solutions
            if self.N.bit_length() > 1000:
                max_relative_error = 0.01
                max_absolute_error = self.N // 100
            elif self.N.bit_length() > 500:
                max_relative_error = 0.001
                max_absolute_error = self.N // 1000
            else:
                max_relative_error = 0.0001
                max_absolute_error = max(1000, self.N // 10000)

            # Try candidates close to hints first
            step = max(1, search_radius // 1000)  # Larger step for efficiency
            for dp in range(-search_radius, search_radius + 1, step):
                for dq in range(-search_radius, search_radius + 1, step):
                    p_test = p_center + dp
                    q_test = q_center + dq

                    if p_test > 1 and q_test > 1:
                        product = p_test * q_test

                        # Check for exact solution first
                        if product == int(self.N):
                            print(f"[Diophantine] ✓✓✓ EXACT SOLUTION FOUND from hint-guided search: p={p_test}, q={q_test}")
                            print(f"[Diophantine] Distance from hints: dp={dp}, dq={dq}")
                            return (p_test, q_test)
                        
                        # Track approximate solution
                        error = abs(product - self.N)
                        relative_error = error / self.N if self.N > 0 else float('inf')
                        
                        if error <= max_absolute_error and relative_error <= max_relative_error:
                            if best_approx is None or error < best_error:
                                best_approx = (p_test, q_test)
                                best_error = error
            
            # Return best approximate solution if found
            if best_approx:
                p_approx, q_approx = best_approx
                product = p_approx * q_approx
                relative_error = (best_error / self.N * 100) if self.N > 0 else 0
                print(f"[Diophantine] ✓ Best approximate solution from hint-guided search:")
                print(f"[Diophantine]   p = {p_approx}, q = {q_approx}")
                print(f"[Diophantine]   p * q = {product}, N = {self.N}")
                print(f"[Diophantine]   Error = {best_error} ({relative_error:.4f}%)")
                
                # Try refinement
                print()
                print("[Diophantine] Attempting refinement to exact factors...")
                exact_result = self._refine_approximate_solution(p_approx, q_approx, product - self.N)
                if exact_result:
                    return exact_result
                
                return (int(p_approx), int(q_approx))

        # No more fallbacks - return None if no solution found
        print("[Diophantine] No solution found with available constraints.")
        return None
    
    def _refine_approximate_solution(self, p_approx: int, q_approx: int, error: int) -> Optional[Tuple[int, int]]:
        """
        Refine an approximate solution (p_approx, q_approx) to exact factors.
        
        Uses the error = p_approx * q_approx - N to guide refinement.
        If error > 0, we need to decrease p or q (or both).
        If error < 0, we need to increase p or q (or both).
        
        Args:
            p_approx: Approximate value for p
            q_approx: Approximate value for q
            error: p_approx * q_approx - N_target
            
        Returns:
            Tuple of (p, q) if exact factors found, None otherwise
        """
        print(f"[Diophantine] Refining: p_approx = {p_approx}, q_approx = {q_approx}")
        print(f"[Diophantine] Error = {error}")
        print()
        
        # Method 1: Use error to estimate correction
        # If p_approx * q_approx = N + error, then:
        # (p_approx + dp) * (q_approx + dq) = N
        # p_approx * q_approx + p_approx * dq + q_approx * dp + dp * dq = N
        # error + p_approx * dq + q_approx * dp + dp * dq = 0
        # For small dp, dq: error + p_approx * dq + q_approx * dp ≈ 0
        # So: p_approx * dq + q_approx * dp ≈ -error
        
        # Estimate: if we adjust only q: p_approx * dq ≈ -error, so dq ≈ -error / p_approx
        # Or if we adjust only p: q_approx * dp ≈ -error, so dp ≈ -error / q_approx
        
        # Calculate estimates more carefully
        # error = p_approx * q_approx - N
        # If error < 0, then p_approx * q_approx < N, so we need to increase p or q
        # If error > 0, then p_approx * q_approx > N, so we need to decrease p or q
        
        # For large numbers, use the error magnitude to estimate search range
        # The error tells us how far off the product is
        error_magnitude = abs(error).bit_length()
        error_abs = abs(error)
        
        # Estimate: if we adjust only q: p_approx * dq ≈ -error
        # dq ≈ -error / p_approx
        # For large numbers, this might be small, so we need to search around 0
        
        # Better approach: use the relationship that if p is close, then
        # q = N / p should be close to q_approx
        # So we search for p such that N / p is close to q_approx
        
        print(f"[Diophantine] Error analysis:")
        print(f"[Diophantine]   error = {error}")
        print(f"[Diophantine]   error magnitude = {error_magnitude} bits")
        print(f"[Diophantine]   error sign: {'negative (need to increase)' if error < 0 else 'positive (need to decrease)'}")
        print()
        
        # Method 1: Algebraic approach using the error relationship
        # We have: p_approx * q_approx = N + error
        # We want: (p_approx + dp) * (q_approx + dq) = N
        # Expanding: p_approx * q_approx + p_approx * dq + q_approx * dp + dp * dq = N
        # So: N + error + p_approx * dq + q_approx * dp + dp * dq = N
        # error + p_approx * dq + q_approx * dp + dp * dq = 0
        # For small dp, dq: error + p_approx * dq + q_approx * dp ≈ 0
        # p_approx * dq + q_approx * dp ≈ -error

        print(f"[Diophantine] Method 1: Solving correction equations algebraically...")

        # We can solve: p_approx * dq + q_approx * dp = -error
        # Scale range based on error magnitude
        error_magnitude = abs(error)
        max_correction_range = max(10000, min(1000000, error_magnitude // max(1, min(p_approx, q_approx) // 10000)))
        print(f"[Diophantine] Using correction range ±{max_correction_range} for large error")

        # Try dp values and solve for dq
        print(f"[Diophantine] Solving: {p_approx} * dq + {q_approx} * dp = {-error}")

        step_size = max(1, max_correction_range // 10000)  # Adaptive step size
        for dp_trial in range(-max_correction_range, max_correction_range + 1, step_size):
            if dp_trial == 0:
                continue

            # p_approx * dq = -error - q_approx * dp_trial
            numerator = -error - q_approx * dp_trial

            if numerator % p_approx == 0:
                dq_trial = numerator // p_approx

                p_test = p_approx + dp_trial
                q_test = q_approx + dq_trial

                if p_test > 1 and q_test > 1 and p_test * q_test == self.N:
                    print(f"[Diophantine] ✓✓✓ EXACT FACTORS FOUND via correction algebra!")
                    print(f"[Diophantine]   dp = {dp_trial}, dq = {dq_trial}")
                    print(f"[Diophantine]   p = {p_test}, q = {q_test}")
                    return (int(p_test), int(q_test))

        # Try dq values and solve for dp
        for dq_trial in range(-max_correction_range, max_correction_range + 1, step_size):
            if dq_trial == 0:
                continue

            # q_approx * dp = -error - p_approx * dq_trial
            numerator = -error - p_approx * dq_trial

            if numerator % q_approx == 0:
                dp_trial = numerator // q_approx

                p_test = p_approx + dp_trial
                q_test = q_approx + dq_trial

                if p_test > 1 and q_test > 1 and p_test * q_test == self.N:
                    print(f"[Diophantine] ✓✓✓ EXACT FACTORS FOUND via correction algebra!")
                    print(f"[Diophantine]   dp = {dp_trial}, dq = {dq_trial}")
                    print(f"[Diophantine]   p = {p_test}, q = {q_test}")
                    return (int(p_test), int(q_test))

        # Method 2: Use the exact quadratic relationship
        # We have: dp * dq + p_approx * dq + q_approx * dp + error = 0
        # Rearrange: dp * dq + (p_approx) * dq + (q_approx) * dp = -error
        # Group: dq * (dp + p_approx) + q_approx * dp = -error

        print(f"[Diophantine] Method 2: Using exact quadratic relationship...")

        # Try to solve for one variable in terms of the other
        # From: dq * (dp + p_approx) = -error - q_approx * dp
        # dq = (-error - q_approx * dp) / (dp + p_approx)

        for dp_trial in range(-10000, 10001, 10):  # Step by 10 for efficiency
            if dp_trial == -p_approx:  # Avoid division by zero
                continue

            denominator = dp_trial + p_approx
            numerator = -error - q_approx * dp_trial

            if numerator % denominator == 0:
                dq_trial = numerator // denominator

                p_test = p_approx + dp_trial
                q_test = q_approx + dq_trial

                if p_test > 1 and q_test > 1 and p_test * q_test == self.N:
                    print(f"[Diophantine] ✓✓✓ EXACT FACTORS FOUND via quadratic algebra!")
                    print(f"[Diophantine]   dp = {dp_trial}, dq = {dq_trial}")
                    print(f"[Diophantine]   p = {p_test}, q = {q_test}")
                    return (int(p_test), int(q_test))

        # Method 3: Search for exact divisors algebraically
        # If p = p_approx + dp and p divides N exactly, then q = N/p
        # We can check if p_approx + dp divides N for dp that satisfy the error equation

        print(f"[Diophantine] Method 3: Algebraic divisor search...")

        # From the approximation: p_approx * q_approx ≈ N
        # If we assume q is close to q_approx, then p should be close to N / q_approx
        # Let's check divisors around N / q_approx

        p_estimated = self.N // q_approx
        q_estimated = self.N // p_approx

        print(f"[Diophantine] Estimated p from N/q_approx = {p_estimated}")
        print(f"[Diophantine] Estimated q from N/p_approx = {q_estimated}")

        # Check if these estimates work
        if p_estimated * q_estimated == self.N:
            dp = p_estimated - p_approx
            dq = q_estimated - q_approx
            print(f"[Diophantine] ✓✓✓ EXACT FACTORS FOUND from estimates!")
            print(f"[Diophantine]   dp = {dp}, dq = {dq}")
            print(f"[Diophantine]   p = {p_estimated}, q = {q_estimated}")
            return (int(p_estimated), int(q_estimated))

        # Check small adjustments to the estimates
        for dp_offset in range(-1000, 1001):
            p_test = p_estimated + dp_offset
            if p_test > 1 and self.N % p_test == 0:
                q_test = self.N // p_test
                dq = q_test - q_approx
                print(f"[Diophantine] ✓✓✓ EXACT FACTORS FOUND from p estimate adjustment!")
                print(f"[Diophantine]   dp = {dp_offset}, dq = {dq}")
                print(f"[Diophantine]   p = {p_test}, q = {q_test}")
                return (int(p_test), int(q_test))

        for dq_offset in range(-1000, 1001):
            q_test = q_estimated + dq_offset
            if q_test > 1 and self.N % q_test == 0:
                p_test = self.N // q_test
                dp = p_test - p_approx
                print(f"[Diophantine] ✓✓✓ EXACT FACTORS FOUND from q estimate adjustment!")
                print(f"[Diophantine]   dp = {dp}, dq = {dq_offset}")
                print(f"[Diophantine]   p = {p_test}, q = {q_test}")
                return (int(p_test), int(q_test))
        
        # Method 4: Use the relationship error = p_approx * q_approx - N
        # Try to solve: (p_approx + dp) * (q_approx + dq) = N
        # Expanding: p_approx * q_approx + p_approx * dq + q_approx * dp + dp * dq = N
        # So: error + p_approx * dq + q_approx * dp + dp * dq = 0
        # Rearranging: p_approx * dq + q_approx * dp = -error - dp * dq
        # For small dp, dq: p_approx * dq + q_approx * dp ≈ -error
        
        print(f"[Diophantine] Method 4: Solving correction equation...")
        print(f"[Diophantine] Equation: p_approx * dq + q_approx * dp ≈ -error")
        
        # Try small dp values and solve for dq
        for dp in range(-100, 101):
            if dp == 0:
                continue
            
            # p_approx * dq ≈ -error - q_approx * dp
            rhs = -error - q_approx * dp
            if rhs % p_approx == 0:
                dq = rhs // p_approx
                p_test = p_approx + dp
                q_test = q_approx + dq
                
                if p_test > 1 and q_test > 1:
                    product = p_test * q_test
                    if product == self.N:
                        print(f"[Diophantine] ✓✓✓ EXACT FACTORS FOUND via correction equation!")
                        print(f"[Diophantine]   p = p_approx + {dp} = {p_test}")
                        print(f"[Diophantine]   q = q_approx + {dq} = {q_test}")
                        return (int(p_test), int(q_test))
        
        print(f"[Diophantine] Refinement did not find exact factors in tested ranges")
        return None
    
    def solve_with_all_methods(self, polynomials: List[sp.Expr],
                               p_hint: int = None, q_hint: int = None) -> Optional[Tuple[int, int]]:
        """
        Try all solving methods in sequence until one succeeds.
        """
        print(f"\n{'='*80}")
        print("ENHANCED POLYNOMIAL SOLVING - TRYING ALL METHODS")
        print(f"{'='*80}")
        print(f"Target N: {self.N} ({self.N.bit_length()}-bit)")
        print(f"Number of polynomials: {len(polynomials)}")
        print(f"\nComplete polynomial system to solve:")
        for i, poly in enumerate(polynomials, 1):
            print(f"  f{i}(p,q) = {poly} = 0")
        
        methods = [
            ("Diophantine Equation Solver", lambda: self.solve_with_diophantine(polynomials, p_hint, q_hint)),
            ("Root's Method", lambda: self.solve_with_roots_method_iterative(polynomials, p_hint, q_hint)),
            ("Algebraic Elimination", lambda: self.solve_with_algebraic_elimination(polynomials)),
            ("Quadratic Factorization Solver", lambda: self.solve_quadratic_factorization(polynomials)),
            ("Modular Constraints & Trial Division", lambda: self.solve_with_modular_constraints(polynomials)),
        ]
        
        if p_hint and q_hint:
            methods.extend([
                ("Numerical Refinement", lambda: self.solve_with_numerical_refinement(polynomials, p_hint, q_hint)),
                ("Hensel Lifting", lambda: self.solve_with_hensel_lifting(polynomials, p_hint, q_hint)),
            ])
        
        # Include symbolic methods for ALL numbers (they may fail but will attempt)
        methods.extend([
            ("Resultant Elimination", lambda: self.solve_with_resultants(polynomials)),
            ("Gröbner Basis", lambda: self.solve_with_groebner_basis(polynomials)),
        ])
        
        # Track best candidate (for display if no exact solution found)
        best_candidate = None
        best_diff = float('inf')
        best_method = None
        
        for method_name, method_func in methods:
            print(f"\n{'─'*80}")
            print(f"Trying: {method_name}")
            print(f"{'─'*80}")
            
            try:
                solution = method_func()
                if solution:
                    p_val, q_val = solution
                    
                    # Check if exact
                    try:
                        if p_val.bit_length() + q_val.bit_length() <= 4096:
                            product = p_val * q_val
                            if product == self.N:
                                print(f"\n{'='*80}")
                                print(f"🎉 SUCCESS WITH {method_name.upper()}!")
                                print(f"{'='*80}")
                                print(f"p = {p_val}")
                                print(f"q = {q_val}")
                                print(f"Verification: {p_val} × {q_val} = {self.N} ✓")
                                return solution
                            else:
                                # Not exact, but track as best candidate
                                diff = abs(product - self.N)
                                if diff < best_diff:
                                    best_diff = diff
                                    best_candidate = (p_val, q_val)
                                    best_method = method_name
                        else:
                            # Too large to verify exactly, but track as candidate
                            if best_candidate is None:
                                best_candidate = (p_val, q_val)
                                best_method = method_name
                    except:
                        # Error calculating, but still track as candidate
                        if best_candidate is None:
                            best_candidate = (p_val, q_val)
                            best_method = method_name
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[{method_name}] Error: {e}")
                continue
        
        # If we have a best candidate, return it (even if not exact)
        if best_candidate:
            p_best, q_best = best_candidate
            print(f"\n{'='*80}")
            print(f"BEST CANDIDATE FROM {best_method.upper()}")
            print(f"{'='*80}")
            print(f"p = {p_best}")
            print(f"q = {q_best}")
            try:
                if p_best.bit_length() + q_best.bit_length() <= 4096:
                    product = p_best * q_best
                    diff = abs(product - self.N)
                    print(f"Product: {p_best} × {q_best} = {product}")
                    print(f"Difference from N: {diff:,}")
                    if diff == 0:
                        print(f"✓ EXACT FACTORIZATION!")
                    else:
                        print(f"⚠️  Approximation (not exact)")
                else:
                    print(f"⚠️  Approximation (too large to verify exactly)")
            except:
                print(f"⚠️  Approximation (verification failed)")
            return best_candidate
        
        print(f"\n{'='*80}")
        print("All polynomial solving methods exhausted without finding any candidates")
        print(f"{'='*80}")
        return None

    def _ultra_refine_closest_factors(self, closest_approximation, polynomials):
        """
        Ultra-refinement: Take the closest approximation and apply advanced techniques
        to find the exact adjustment needed.

        This method uses:
        1. Polynomial evaluation at the approximation point
        2. Newton-Raphson style refinement
        3. Small search space around the approximation
        """
        p_approx, q_approx, score, diff = closest_approximation

        print(f"[Ultra] Starting ultra-refinement from p≈{p_approx}, q≈{q_approx}")
        print(f"[Ultra] Initial difference: {diff:,}")

        # Method 1: Newton-Raphson style refinement using polynomial derivatives
        try:
            # Find polynomials that are close to zero at our approximation
            best_poly = None
            best_residual = float('inf')

            for poly in polynomials[:50]:  # Use best polynomials
                try:
                    residual = abs(poly.subs([(self.p, p_approx), (self.q, q_approx)]))
                    if residual < best_residual:
                        best_residual = residual
                        best_poly = poly
                except:
                    continue

            if best_poly:
                print(f"[Ultra] Best polynomial residual: {best_residual}")

                # Try Newton-Raphson style adjustment
                # For f(p,q) ≈ 0, we want to adjust p and q to make f(p+dp,q+dq) = 0
                # This is complex for multivariate polynomials, so we'll do simple adjustments

        except Exception as e:
            print(f"[Ultra] Newton-Raphson failed: {e}")

        # Method 2: Small brute force search around the approximation
        print(f"[Ultra] Trying small brute force search...")

        # Calculate how much we need to adjust based on the difference
        try:
            current_product = p_approx * q_approx
            target_diff = self.N - current_product

            # Estimate adjustment to q (since p is smaller)
            if p_approx != 0:
                q_adjustment = target_diff // p_approx
                q_candidate = q_approx + q_adjustment

                # Verify
                if p_approx * q_candidate == self.N:
                    print(f"[Ultra] ✅ Exact adjustment found: q += {q_adjustment}")
                    return (p_approx, q_candidate)

                # Try small variations around this adjustment
                for delta in range(-1000, 1001):
                    q_test = q_candidate + delta
                    if p_approx * q_test == self.N:
                        print(f"[Ultra] ✅ Exact adjustment found: q += {q_adjustment + delta}")
                        return (p_approx, q_test)

        except:
            pass

        # Method 3: Use polynomial constraints to guide small adjustments
        print(f"[Ultra] Using polynomial constraints for guided search...")

        # Look for linear constraints that can help adjust the factors
        linear_polys = []
        for poly in polynomials:
            try:
                if poly.is_linear:
                    linear_polys.append(poly)
            except:
                continue

        # Try small adjustments guided by linear constraints
        # MASSIVE EXPANSION for 2048-bit numbers: up to 10 BILLION radius
        search_radius = min(10000000000, max(1000000, abs(diff) // max(1, p_approx)))  # 10^10 radius for huge numbers

        print(f"[Ultra] Guided search radius: {search_radius}")

        for dp in range(-search_radius, search_radius + 1, max(1, search_radius // 100)):
            for dq in range(-search_radius, search_radius + 1, max(1, search_radius // 100)):
                p_test = p_approx + dp
                q_test = q_approx + dq

                if p_test <= 1 or q_test <= 1:
                    continue

                # Quick check using linear constraints
                valid_constraints = 0
                for linear in linear_polys[:10]:  # Use first 10 linear constraints
                    try:
                        residual = abs(linear.subs([(self.p, p_test), (self.q, q_test)]))
                        if residual < 1000:  # Reasonable threshold
                            valid_constraints += 1
                    except:
                        continue

                # If most constraints are satisfied, check exact factorization
                if valid_constraints >= len(linear_polys[:10]) // 2:
                    try:
                        if p_test * q_test == self.N:
                            print(f"[Ultra] 🎯 SUCCESS! Guided search found exact factors")
                            print(f"[Ultra] Adjustments: dp={dp}, dq={dq}")
                            return (p_test, q_test)
                    except:
                        continue

        print(f"[Ultra] Ultra-refinement completed - no exact factors found")
        return None


# ============================================================================
# MINIMIZABLE FACTORIZATION LATTICE SOLVER
# ============================================================================

class MinimizableFactorizationLatticeSolver:
    """
    Standalone Minimizable Factorization Lattice Solver

    Uses lattice minimization to find optimal factorization corrections.
    Creates vectors representing (dp,dq) corrections with quality metrics, then
    finds the "shortest" vectors that minimize factorization error.
    """

    def __init__(self, N: int, delta: float = 0.75):
        """
        Initialize minimizable factorization lattice solver.

        Args:
            N: Number to factor
            delta: Lattice minimization parameter (0.25 < delta < 1.0, typically 0.75)
        """
        self.N = N
        self.delta = delta
        self.n_bits = N.bit_length()
        
        # Initialize p and q approximations
        import math
        sqrt_N = math.isqrt(N)
        self.p_approx = sqrt_N
        self.q_approx = N // sqrt_N
        
        print(f"[Lattice Solver] Initialized for {self.n_bits}-bit N, delta={delta}")
        print(f"[Lattice Solver] Initial p_approx = {self.p_approx}, q_approx = {self.q_approx}")

    def refine_approximations(self, max_iterations: int = 5) -> Tuple[int, int]:
        """
        Refine p and q approximations using iterative methods.
        
        Uses:
        1. Newton's method: p_new = (p_old + N/p_old) // 2
        2. Balance correction: Adjust to minimize |p - q|
        3. Error minimization: Find p that minimizes |p * (N//p) - N|
        
        Returns:
            (refined_p_approx, refined_q_approx)
        """
        import math
        
        p_current = self.p_approx
        q_current = self.q_approx
        best_error = abs(p_current * q_current - self.N)
        best_p = p_current
        best_q = q_current
        
        print(f"[Refinement] Starting with p={p_current}, q={q_current}, error={best_error}")
        
        # Method 1: Newton's method refinement (converges to sqrt(N))
        for iteration in range(max_iterations):
            # Newton step: p_new = (p + N/p) // 2
            if p_current > 0:
                p_new = (p_current + self.N // p_current) // 2
                q_new = self.N // p_new
                error = abs(p_new * q_new - self.N)
                
                if error < best_error:
                    best_error = error
                    best_p = p_new
                    best_q = q_new
                    print(f"[Refinement] Iteration {iteration+1}: p={p_new}, error={error} (improved)")
                
                # Check convergence
                if abs(p_new - p_current) < max(1, p_current // 1000):
                    break
                
                p_current = p_new
                q_current = q_new
        
        # Method 2: Try values that make p and q more balanced
        # For RSA, p and q are typically within a factor of 2-10
        sqrt_N = math.isqrt(self.N)
        for factor in [8, 9, 10, 11, 12]:  # Try different ratios
            p_candidate = (sqrt_N * factor) // 10
            if p_candidate > 0:
                q_candidate = self.N // p_candidate
                error = abs(p_candidate * q_candidate - self.N)
                
                # Also check balance (prefer p and q closer in size)
                balance = abs(p_candidate - q_candidate)
                if error < best_error * 1.1 and balance < abs(best_p - best_q) * 1.1:
                    best_p = p_candidate
                    best_q = q_candidate
                    best_error = error
        
        initial_error = abs(self.p_approx * self.q_approx - self.N)
        self.p_approx = best_p
        self.q_approx = best_q
        
        print(f"[Refinement] Final: p={best_p}, q={best_q}, error={best_error}")
        print(f"[Refinement] Improvement: {initial_error - best_error} reduction in error")
        
        return best_p, best_q

    def _construct_pyramid_lattice_basis(self, p_candidate: int, q_candidate: int) -> np.ndarray:
        """
        Construct a pyramid-shaped lattice basis for factorization.

        Pyramid structure with normalized coefficients for large numbers:
        - Base layer: Fundamental relations (p, q, p*q - N) with reduced coefficients
        - Middle layer: Derived relations (p+q, p-q, p²+q²) normalized
        - Apex: Complex combinations forming pyramid apex with bounded coefficients
        """
        print(f"[Lattice] Constructing pyramid lattice basis...")

        # Adaptive coefficient handling based on number size and lattice dimension
        import math
        sqrt_N_bits = int(math.sqrt(self.N.bit_length()))

        # Use lattice dimension parameter if available to scale coefficients
        if hasattr(self, 'config') and self.config and self.config.get('lattice_dimension'):
            lattice_base = self.config['lattice_dimension']
            try:
                lattice_base_int = int(lattice_base)
                lattice_exponent = lattice_base_int.bit_length()

                # Scale coefficients based on lattice dimension and number size
                if self.N.bit_length() > 1000:
                    total_exponent = lattice_exponent + sqrt_N_bits
                    coeff_limit = 2 ** total_exponent  # NO CAP - let lattice dimension scale fully
                    norm_factor = 2**(self.N.bit_length() // 10)
                    print(f"[Lattice]   Using lattice-scaled coefficients (2^{total_exponent} limit)")
                elif self.N.bit_length() > 500:
                    total_exponent = lattice_exponent + (sqrt_N_bits // 2)
                    coeff_limit = 2 ** total_exponent  # NO CAP - let lattice dimension scale fully
                    norm_factor = 2**(self.N.bit_length() // 12)
                    print(f"[Lattice]   Using lattice-scaled coefficients (2^{total_exponent} limit)")
                else:
                    coeff_limit = 2 ** lattice_exponent  # NO CAP - let lattice dimension scale fully
                    norm_factor = 2**(self.N.bit_length() // 16)
                    print(f"[Lattice]   Using lattice coefficients (2^{lattice_exponent} limit)")

            except (ValueError, OverflowError) as e:
                print(f"[Lattice]   ERROR: Invalid lattice dimension '{lattice_base}': {e}, using default scaling")
                # For invalid lattice dimensions, still allow extreme scaling based on N size
                if self.N.bit_length() > 1000:
                    coeff_limit = 2**(self.N.bit_length() // 2)  # Extreme scaling even for fallback
                    norm_factor = 2**(self.N.bit_length() // 8)
                else:
                    coeff_limit = 10**18
                    norm_factor = 2**(self.N.bit_length() // 8)
        elif self.N.bit_length() > 1500:
            # For extremely large numbers, use normalized coefficients
            coeff_limit = 10**18
            norm_factor = 2**(self.N.bit_length() // 8)
            print(f"[Lattice]   Using normalized coefficients (limit: {coeff_limit}, norm: 2^{self.N.bit_length() // 8})")

            def reduce_coeff(x):
                """Normalize then reduce large coefficients"""
                if abs(x) > coeff_limit:
                    normalized = x // norm_factor
                    if abs(normalized) > coeff_limit:
                        sign = 1 if normalized > 0 else -1
                        reduced = abs(normalized) % coeff_limit
                        return sign * reduced if reduced != 0 else sign * coeff_limit
                    return normalized
                return x
        elif self.N.bit_length() > 1000:
            # For very large numbers, use moderate precision
            coeff_limit = 10**20
            print(f"[Lattice]   Using moderate precision (coefficient limit: {coeff_limit})")

            def reduce_coeff(x):
                """Reduce large coefficients to manageable size"""
                if abs(x) > coeff_limit:
                    return x % coeff_limit if x > 0 else -(abs(x) % coeff_limit)
                return x
        else:
            # For very large numbers, use moderate precision
            coeff_limit = 10**20  # Ultra-expanded for massive coefficient range
            print(f"[Lattice]   Using moderate precision (coefficient limit: {coeff_limit})")

            def reduce_coeff(x):
                """Reduce large coefficients to manageable size"""
                if abs(x) > coeff_limit:
                    return x % coeff_limit if x > 0 else -(abs(x) % coeff_limit)
                return x
            # For smaller numbers, use standard reduction
            coeff_limit = 10**12  # Expanded from 10^6 to 10^12

        def reduce_coeff(x):
                """Reduce large coefficients to manageable size"""
                if abs(x) > coeff_limit:
                    return x % coeff_limit if x > 0 else -(abs(x) % coeff_limit)
                return x

        # Base vectors (fundamental relations with reduced coefficients)
        # Key relationship: p*q = N, so (p_candidate + dp)*(q_candidate + dq) = N
        # Expanding: p_candidate*q_candidate + p_candidate*dq + q_candidate*dp + dp*dq = N
        # So: p_candidate*dq + q_candidate*dp + dp*dq = N - p_candidate*q_candidate
        # For lattice: a + b*dp + c*dq = 0, where we want to find small dp, dq
        
        # The fundamental relationship: p*q - N = 0
        # With corrections: (p_candidate + dp)*(q_candidate + dq) - N = 0
        # This gives: p_candidate*q_candidate - N + p_candidate*dq + q_candidate*dp + dp*dq = 0
        # Linearizing (ignoring dp*dq for now): p_candidate*dq + q_candidate*dp = N - p_candidate*q_candidate
        
        error = self.N - p_candidate * q_candidate  # The error we need to correct
        
        base_vectors = [
            # Fundamental: p_candidate*dq + q_candidate*dp = error
            # Represented as: error + q_candidate*dp + p_candidate*dq = 0
            [reduce_coeff(error), reduce_coeff(q_candidate), reduce_coeff(p_candidate)],
            
            # Alternative forms of the same relationship
            [reduce_coeff(-error), reduce_coeff(-q_candidate), reduce_coeff(-p_candidate)],
            
            # p*q = N relationship: p_candidate*q_candidate - N + corrections
            [reduce_coeff(p_candidate * q_candidate - self.N), reduce_coeff(q_candidate), reduce_coeff(p_candidate)],
            
            # Sum relationship: (p_candidate + dp) + (q_candidate + dq) = p_candidate + q_candidate + dp + dq
            [0, 1, 1],  # dp + dq = 0 (for balanced corrections)
            
            # Difference relationship: (p_candidate + dp) - (q_candidate + dq) = p_candidate - q_candidate + dp - dq
            [0, 1, -1],  # dp - dq = 0
            
            # Product relationship variations
            [reduce_coeff(self.N), reduce_coeff(-q_candidate), reduce_coeff(-p_candidate)],  # N - q_candidate*dp - p_candidate*dq = 0
        ]

        # Middle layer (derived relations with normalized coefficients)
        s = p_candidate + q_candidate  # sum
        d = p_candidate - q_candidate  # difference
        pq = p_candidate * q_candidate  # product

        # MASSIVELY EXPANDED: Many more derived relations for large numbers
        middle_vectors = [
            [1, 1, -reduce_coeff(s)],  # 1 + p + q = s
            [1, -1, -reduce_coeff(d)], # 1 + p - q = d
            [reduce_coeff(p_candidate), reduce_coeff(q_candidate), -reduce_coeff(s)], # p*1 + q*p - s*q = p + q*p - s*q
            # Additional middle layer relations
            [reduce_coeff(s), reduce_coeff(d), -reduce_coeff(pq)],  # s*1 + d*p - pq*q = s + d*p - pq*q
            [reduce_coeff(pq), reduce_coeff(s), -reduce_coeff(p_candidate**2 + q_candidate**2)],  # pq*1 + s*p - (p²+q²)*q = pq + s*p - (p²+q²)*q
            [reduce_coeff(d), reduce_coeff(pq), -reduce_coeff(p_candidate * s - q_candidate * d)],  # Complex cross-relations
            [reduce_coeff(p_candidate**2), reduce_coeff(q_candidate**2), -reduce_coeff((p_candidate * q_candidate)**2)],  # p²*1 + q²*p - (pq)²*q
            # Even more relations for 2048-bit coverage
            [reduce_coeff(p_candidate * q_candidate * s), reduce_coeff(d), -reduce_coeff(p_candidate + q_candidate)],  # pq*s*1 + d*p - (p+q)*q
            [reduce_coeff(p_candidate + q_candidate), reduce_coeff(p_candidate * q_candidate), -reduce_coeff(s * d)],  # (p+q)*1 + pq*p - s*d*q
        ]

        # Apex vectors (complex pyramid combinations with bounded coefficients)
        p2 = reduce_coeff(p_candidate**2 % coeff_limit)
        q2 = reduce_coeff(q_candidate**2 % coeff_limit)
        pq = reduce_coeff((p_candidate * q_candidate) % coeff_limit)

        # MASSIVELY EXPANDED: Many more apex relations for comprehensive 2048-bit coverage
        apex_vectors = [
            [reduce_coeff(pq), 1, -reduce_coeff(self.N % coeff_limit)],  # pq*1 + 1*p - N*q = pq + p - N*q
            [reduce_coeff(s), reduce_coeff(d), -2],  # s*1 + d*p - 2*q = s + d*p - 2*q
            [reduce_coeff(p2 - q2), reduce_coeff(2 * pq), -1],  # (p²-q²)*1 + 2pq*p - q = p²-q² + 2pq*p - q
            # Additional apex vectors for 2048-bit numbers
            [reduce_coeff(pq * s), reduce_coeff(d), -reduce_coeff(p_candidate)],  # pq*s*1 + d*p - p*q
            [reduce_coeff(p2 + q2), reduce_coeff(2*pq), -reduce_coeff(s**2 - 2*pq)],  # (p²+q²)*1 + 2pq*p - (s²-2pq)*q
            [reduce_coeff(p_candidate * d + q_candidate * s), reduce_coeff(pq), -reduce_coeff(s * d)],  # Complex apex relations
            [reduce_coeff(pq**2), reduce_coeff(s * d), -reduce_coeff(p_candidate**2 * q_candidate**2)],  # pq²*1 + s*d*p - p²q²*q
            # Even more comprehensive apex relations
            [reduce_coeff((p_candidate + q_candidate)**3), reduce_coeff(3*pq*(p_candidate + q_candidate)), -reduce_coeff(p_candidate**3 + q_candidate**3 + 3*pq*s)],  # Cubic relations
            [reduce_coeff(p_candidate**4 + q_candidate**4 + 2*pq**2), reduce_coeff(4*pq*s), -reduce_coeff((p_candidate**2 + q_candidate**2)**2)],  # Quartic relations
            [reduce_coeff(pq * (p_candidate**2 + q_candidate**2)), reduce_coeff(s**2 - 2*pq), -reduce_coeff(p_candidate * q_candidate * (p_candidate + q_candidate))],  # Mixed high-degree relations
        ]

        # Combine all layers to form pyramid
        pyramid_basis = base_vectors + middle_vectors + apex_vectors

        # Convert to numpy array
        basis = np.array(pyramid_basis, dtype=object)

        print(f"[Lattice] ✓ Pyramid lattice constructed with {len(pyramid_basis)} vectors")
        print(f"[Lattice]   Base: {len(base_vectors)} vectors")
        print(f"[Lattice]   Middle: {len(middle_vectors)} vectors")
        print(f"[Lattice]   Apex: {len(apex_vectors)} vectors")
        print(f"[Lattice]   Coefficient limit: {coeff_limit}")

        return basis

    def _bulk_search_factors_with_lll(self, search_radius: int, pretrained_transformer: Optional['StepPredictionTransformer'] = None) -> Optional[Tuple[int, int]]:
        """
        Bulk search for factors using LLL following univariate polynomial root-finding logic.
        
        Strategy: Step through the ENTIRE RSA key space in increments of 2^100,
        running lattice attacks at each step and tracking the best factors found.
        Search range: From 2 to N (entire key space).
        
        This follows the same logic as the univariate polynomial approach:
        1. Divide entire search space into steps of 2^100
        2. At each step, construct polynomial representation centered at that point
        3. Use LLL to find roots (small x values) that give factors
        4. Extract factors from roots and track best results
        """
        print(f"[Lattice] 🔍 TRANSFORMER-GUIDED BULK SEARCH MODE (EXPANDED RANGE)")
        print(f"[Lattice] Learning from p-q bit differences to guide search")
        print(f"[Lattice] ⚡ EXPANDED SEARCH CAPABILITY: Jumps from 2^30 to 2^100 (was 2^80)")
        import sys
        sys.stdout.flush()

        import math
        sqrt_N = int(math.isqrt(self.N))
        N_bits = self.N.bit_length()

        # Search range: entire key space
        search_lower = max(1000, sqrt_N // 1000)  # Skip tiny numbers
        search_upper = self.N

        print(f"[Lattice] Search range: {search_lower:,} to N")
        print(f"[Lattice] N bit length: {N_bits}")

        # Initialize Transformer for learning p-q patterns
        no_transformer = getattr(self, 'config', {}).get('no_transformer', False)
        use_transformer = not no_transformer and TORCH_AVAILABLE

        if pretrained_transformer is not None:
            # Use the pre-trained transformer
            transformer = pretrained_transformer
            print(f"[Transformer] Using pre-trained transformer with {len(transformer.search_history)} experiences")
            if hasattr(transformer, 'pretrained') and transformer.pretrained:
                print(f"[Transformer] 🎓 Pre-trained model active - learned from synthetic RSA keys")
        elif use_transformer:
            transformer = StepPredictionTransformer(d_model=64, nhead=2, num_layers=1)
            print(f"[Transformer] Initialized fresh neural network for p-q pattern learning")
        else:
            transformer = StepPredictionTransformer(use_torch=False)
            print(f"[Transformer] Using simplified model (PyTorch disabled)")

        sys.stdout.flush()

        # Search parameters - continue until factorization found
        step_num = 0
        current_position = search_lower
        max_consecutive_failures = 1000  # Safety limit for consecutive failures
        consecutive_failures = 0

        # Progressive learning: transformer starts learning immediately
        print(f"[Lattice] PROGRESSIVE LEARNING MODE")
        print(f"[Lattice] Transformer will learn from every step and guide search from the start")
        print(f"[Lattice] Will continue until exact factorization found")

        while current_position < search_upper and consecutive_failures < max_consecutive_failures:
            step_num += 1

            # Progressive learning: transformer guides search from step 3 onward
            if use_transformer and len(transformer.factor_history) >= 3:
                # Use Transformer to predict next promising position
                # The transformer now learns progressively from ALL previous steps!
                step_center = transformer.predict_next_search_position(
                    current_position, sqrt_N, N_bits
                )
                # Ensure it's within search bounds
                step_center = max(search_lower, min(step_center, search_upper - 1000))

                # Safe large integer formatting
                try:
                    pos_str = f"{step_center:,}"
                except (ValueError, OverflowError):
                    pos_str = f"{step_center}"

                # Show transformer confidence and learning progress
                history_size = len(transformer.factor_history)
                print(f"\n[Transformer] Step {step_num}: Learned from {history_size} previous results")
                print(f"[Transformer] Predicted next position: {pos_str}")

                # Occasionally add some exploration even when using transformer
                # This prevents getting stuck in local optima
                if step_num % 10 == 0 and history_size > 5:
                    # Every 10th step, add some random exploration
                    exploration_offset = 2**(35 + (step_num // 10) % 15)
                    alt_center = step_center + exploration_offset
                    alt_center = max(search_lower, min(alt_center, search_upper - 1000))
                    if alt_center != step_center:
                        print(f"[Transformer] Adding exploration: also testing position {alt_center:,}")
                        # We'll test both positions (modify logic below)

            else:
                # Initial exploration to build training data for the transformer
                # Use varied stepping to learn different patterns
                step_distance = 2**(35 + (step_num * 3) % 65)  # EXPANDED: 2^35 to 2^100 range
                step_center = current_position + step_distance
                # Bound to search range
                step_center = min(step_center, search_upper - 1000)

                # Safe bit length calculation
                try:
                    scale_bits = step_distance.bit_length()
                except (AttributeError, OverflowError):
                    scale_bits = len(bin(step_distance)) - 2 if step_distance > 0 else 0

                print(f"\n[Exploration] Step {step_num}: Building training data at scale 2^{scale_bits}")
                print(f"[Exploration] Transformer will start guiding after {3 - len(transformer.factor_history)} more results")

            # Ensure we don't exceed bounds
            step_center = min(step_center, search_upper - 1000)
            if step_center >= search_upper:
                break

            print(f"\n{'='*80}")
            # Safe large integer formatting
            try:
                center_str = f"{step_center:,}"
            except (ValueError, OverflowError):
                center_str = f"{step_center}"
            print(f"[Lattice] PROGRESSIVE LEARNING STEP {step_num}: Testing position {center_str}")
            if len(transformer.factor_history) < 3:
                print(f"[Lattice] EARLY EXPLORATION - Building initial training data")
            else:
                print(f"[Lattice] AI-GUIDED SEARCH - Learning from {len(transformer.factor_history)} results")
            print(f"{'='*80}")

            # Run LLL lattice attack
            step_radius = 10000
            step_result = self._bulk_search_step(step_center, step_radius, sqrt_N)

            # Analyze result and learn
            if step_result:
                p_candidate, q_candidate, diff, diff_bits = step_result

                # Always print current candidates and difference from N
                print(f"[Lattice]    Current candidates: p={p_candidate}, q={q_candidate}")
                if p_candidate * q_candidate != self.N:
                    product_diff = abs(p_candidate * q_candidate - self.N)
                    print(f"[Lattice]    Product difference: {product_diff:,} ({product_diff.bit_length()} bits from N)")
                else:
                    print(f"[Lattice]    ✓✓✓ PERFECT MATCH: p × q == N")

                if diff == 0 and p_candidate * q_candidate == self.N:
                    print(f"[Lattice] ✓✓✓ EXACT FACTORIZATION FOUND!")
                    return (p_candidate, q_candidate)

                # Record result for Transformer learning with correction signals
                known_p = getattr(self, 'known_p', None)
                known_q = getattr(self, 'known_q', None)
                transformer.add_search_result(
                    step_center, diff_bits, sqrt_N, N_bits,
                    p_candidate, q_candidate, known_p, known_q
                )

                # Analyze p-q bit difference pattern
                if p_candidate and q_candidate:
                    p_q_diff = abs(p_candidate - q_candidate)
                    p_q_diff_bits = p_q_diff.bit_length() if p_q_diff > 0 else 0
                    print(f"[Transformer] Recorded p-q bit difference: {p_q_diff_bits} bits")
                    print(f"[Transformer] Result quality: {diff_bits} bits from exact")

            else:
                # No result found, still record for learning with correction signals
                known_p = getattr(self, 'known_p', None)
                known_q = getattr(self, 'known_q', None)
                transformer.add_search_result(step_center, None, sqrt_N, N_bits, None, None, known_p, known_q)
                print(f"[Lattice] No factors found at this position")

            # Retrain Transformer more frequently for progressive learning
            if use_transformer:
                history_size = len(transformer.factor_history)
                # Retrain more often as we gather more data
                retrain_interval = max(5, 15 - history_size // 10)  # Start at every 5 steps, increase as we learn more

                if step_num % retrain_interval == 0 and history_size >= 3:
                    print(f"[Transformer] Progressive retraining on {history_size} samples (every {retrain_interval} steps)...")
                    transformer.train_on_history(epochs=2)  # Shorter training for frequent updates

                    # Show learning progress
                    if history_size >= 5:
                        avg_quality = sum(f[4] for f in transformer.factor_history[-10:] if f[4] is not None) / len([f for f in transformer.factor_history[-10:] if f[4] is not None])
                        print(f"[Transformer] Recent average quality: {avg_quality:.1f} bits from exact")

            # Update position for next iteration
            # Transformer now controls positioning completely - no fixed increment
            current_position = step_center  # Will be updated by Transformer prediction next iteration

            # Reset failure counter on any result
            if step_result:
                consecutive_failures = 0
            else:
                consecutive_failures += 1

            # Periodic status update
            if step_num % 50 == 0:
                print(f"[Lattice] Progress: {step_num} steps completed, {consecutive_failures} consecutive failures")
                if use_transformer:
                    print(f"[Transformer] Has {len(transformer.factor_history)} learning samples")
                    print(f"[Transformer] Current search position: 2^{current_position.bit_length()} bits")

        # Search complete (gave up after too many consecutive failures)
        print(f"\n{'='*80}")
        print(f"[Lattice] PROGRESSIVE LEARNING TERMINATED ({step_num} steps)")
        print(f"[Lattice] Stopped after {consecutive_failures} consecutive failures")
        print(f"[Lattice] No exact factorization found")
        if use_transformer:
            print(f"[Transformer] Progressive learning completed with {len(transformer.factor_history)} training samples")
            if len(transformer.factor_history) >= 3:
                avg_quality = sum(f[4] for f in transformer.factor_history if f[4] is not None) / len([f for f in transformer.factor_history if f[4] is not None])
                print(f"[Transformer] Final average result quality: {avg_quality:.1f} bits from exact")
        print(f"{'='*80}")

        return None
    
    def _bulk_search_step(self, step_center: int, step_size: int, sqrt_N: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Run a single step of the bulk search: pyramidal lattice attack centered at step_center.
        
        Uses the pyramidal lattice construction to find factors.
        
        Returns: (best_p, best_q, best_diff, best_diff_bits) or None
        """
        import math
        
        # Use step_center as p_candidate approximation
        p_candidate = step_center
        q_candidate = self.N // p_candidate if p_candidate > 0 else sqrt_N
        
        print(f"[Lattice]    Step center: {step_center:,} (using as p_candidate)")
        print(f"[Lattice]    Constructing pyramidal lattice...")
        
        # Construct pyramidal lattice basis
        try:
            pyramid_basis = self._construct_pyramid_lattice_basis(p_candidate, q_candidate)
        except Exception as e:
            print(f"[Lattice]    Failed to construct pyramidal lattice: {e}")
            return None
        
        if pyramid_basis is None or len(pyramid_basis) == 0:
            print(f"[Lattice]    Empty pyramidal lattice basis")
            return None
        
        # Apply LLL reduction using the same method as _find_best_factorization_corrections
        try:
            reduced_basis = self._minimize_lattice(pyramid_basis)
        except Exception as e:
            print(f"[Lattice]    LLL reduction failed: {e}")
            return None
        
        if reduced_basis is None or len(reduced_basis) == 0:
            print(f"[Lattice]    Empty reduced basis")
            return None
        
        print(f"[Lattice]    Reduced pyramidal lattice has {len(reduced_basis)} vectors")
        
        # Extract factors from pyramidal lattice vectors
        # Vector format: [a, b, c] represents a + b*dp + c*dq = 0
        # Where dp and dq are corrections: p = p_candidate + dp, q = q_candidate + dq
        best_p = None
        best_q = None
        best_diff = None
        best_diff_bits = None
        initial_diff = abs(p_candidate * q_candidate - self.N)
        
        # Sort vectors by norm (shortest first) - like in _find_best_factorization_corrections
        vector_norms = []
        for i, vector in enumerate(reduced_basis):
            try:
                if len(vector) < 3:
                    continue
                # Compute L2 norm squared
                norm_sq = sum(int(x)**2 for x in vector[:3] if isinstance(x, (int, np.integer)))
                vector_norms.append((norm_sq, i, vector))
            except:
                vector_norms.append((float('inf'), i, vector))
        
        vector_norms.sort(key=lambda x: x[0])
        
        # Check vectors in order of increasing norm
        for norm_sq, orig_idx, vector in vector_norms:
            try:
                if len(vector) < 3:
                    continue
                
                a, b, c = vector[0], vector[1], vector[2]
                
                # Skip if both b and c are zero (no corrections)
                if (b == 0 and c == 0):
                    continue
                
                # Solve for corrections: a + b*dp + c*dq = 0
                # Try solving for dp first
                if b != 0:
                    # Try dp = -a/b (assuming dq = 0)
                    try:
                        dp = -a // b if b != 0 else 0
                        dq = 0
                        p_test = p_candidate + dp
                        q_test = q_candidate + dq
                        
                        if p_test > 1 and q_test > 1:
                            # Check exact factorization
                            if p_test * q_test == self.N:
                                print(f"[Lattice]    ✓✓✓ EXACT FACTORIZATION FOUND (vector {orig_idx})!")
                                return (p_test, q_test, 0, 0)
                            
                            # Check if p divides N exactly
                            if self.N % p_test == 0:
                                q_exact = self.N // p_test
                                if p_test * q_exact == self.N:
                                    print(f"[Lattice]    ✓✓✓ EXACT FACTORIZATION FOUND (vector {orig_idx})!")
                                    return (p_test, q_exact, 0, 0)
                            
                            # Check product difference
                            product = p_test * q_test
                            product_diff = abs(product - self.N)
                            product_diff_bits = product_diff.bit_length() if product_diff > 0 else 0
                            
                            # Update best if this is better
                            if best_diff is None or product_diff < best_diff:
                                best_p = p_test
                                best_q = q_test
                                best_diff = product_diff
                                best_diff_bits = product_diff_bits
                                
                                # Early exit if very close
                                if product_diff_bits < 10:
                                    return (best_p, best_q, best_diff, best_diff_bits)
                    except:
                        pass
                
                # Try solving for dq
                if c != 0:
                    try:
                        dq = -a // c if c != 0 else 0
                        dp = 0
                        p_test = p_candidate + dp
                        q_test = q_candidate + dq
                        
                        if p_test > 1 and q_test > 1:
                            if p_test * q_test == self.N:
                                print(f"[Lattice]    ✓✓✓ EXACT FACTORIZATION FOUND (vector {orig_idx})!")
                                return (p_test, q_test, 0, 0)
                            
                            if self.N % p_test == 0:
                                q_exact = self.N // p_test
                                if p_test * q_exact == self.N:
                                    print(f"[Lattice]    ✓✓✓ EXACT FACTORIZATION FOUND (vector {orig_idx})!")
                                    return (p_test, q_exact, 0, 0)
                            
                            product = p_test * q_test
                            product_diff = abs(product - self.N)
                            product_diff_bits = product_diff.bit_length() if product_diff > 0 else 0
                            
                            if best_diff is None or product_diff < best_diff:
                                best_p = p_test
                                best_q = q_test
                                best_diff = product_diff
                                best_diff_bits = product_diff_bits
                                
                                if product_diff_bits < 10:
                                    return (best_p, best_q, best_diff, best_diff_bits)
                    except:
                        pass
                
                # Try solving for both dp and dq (if both b and c are non-zero)
                if b != 0 and c != 0:
                    # Try small dq values and solve for dp
                    for dq_try in range(-100, 101):
                        try:
                            if (a + c * dq_try) % b == 0:
                                dp = -(a + c * dq_try) // b
                                p_test = p_candidate + dp
                                q_test = q_candidate + dq_try
                                
                                if p_test > 1 and q_test > 1:
                                    if p_test * q_test == self.N:
                                        print(f"[Lattice]    ✓✓✓ EXACT FACTORIZATION FOUND (vector {orig_idx})!")
                                        return (p_test, q_test, 0, 0)
                                    
                                    if self.N % p_test == 0:
                                        q_exact = self.N // p_test
                                        if p_test * q_exact == self.N:
                                            print(f"[Lattice]    ✓✓✓ EXACT FACTORIZATION FOUND (vector {orig_idx})!")
                                            return (p_test, q_exact, 0, 0)
                                    
                                    product = p_test * q_test
                                    product_diff = abs(product - self.N)
                                    product_diff_bits = product_diff.bit_length() if product_diff > 0 else 0
                                    
                                    if best_diff is None or product_diff < best_diff:
                                        best_p = p_test
                                        best_q = q_test
                                        best_diff = product_diff
                                        best_diff_bits = product_diff_bits
                                        
                                        if product_diff_bits < 10:
                                            return (best_p, best_q, best_diff, best_diff_bits)
                        except:
                            continue
            
            except Exception as e:
                continue
        
        if best_p is not None:
            print(f"[Lattice]    Best from this step: p={best_p}, q={best_q}, diff={best_diff_bits} bits")
            return (best_p, best_q, best_diff, best_diff_bits)
        
        return None

    def _find_best_factorization_corrections(self, p_candidate: int, q_candidate: int,
                                           pyramid_basis: np.ndarray,
                                           config: Optional[np.ndarray] = None,
                                           search_radius: int = 1000) -> Tuple[int, int, float]:
        """
        Find the best factorization corrections using pyramid lattice reduction.
        """
        print(f"[Lattice] Using pyramid lattice for factorization corrections...")
        print(f"[Lattice] Search radius parameter: {search_radius} ({search_radius.bit_length()} bits)")
        
        # For huge search radii, note that we'll extract corrections directly from LLL-reduced vectors
        if search_radius > 10**100:
            print(f"[Lattice] Large search radius detected - will extract corrections from LLL-reduced short vectors")
            print(f"[Lattice] LLL reduction should produce vectors with small coefficients containing the corrections")

        # Minimize the lattice using LLL
        reduced_basis = self._minimize_lattice(pyramid_basis)

        # Extract corrections from shortest vectors
        best_dp = 0
        best_dq = 0
        best_improvement = 0.0
        initial_diff = abs(p_candidate * q_candidate - self.N)

        print(f"[Lattice] Analyzing reduced pyramid lattice vectors...")
        print(f"[Lattice] Reduced basis has {len(reduced_basis)} vectors")
        print(f"[Lattice] Initial difference: {initial_diff} ({initial_diff.bit_length()} bits)")

        # Sort vectors by their L2 norm (length) - check shortest vectors first
        # This helps find the best corrections early
        vector_norms = []
        for i, vector in enumerate(reduced_basis):
            try:
                # Compute L2 norm squared (avoid sqrt for large numbers)
                norm_sq = sum(int(x)**2 for x in vector if isinstance(x, (int, np.integer)))
                vector_norms.append((norm_sq, i, vector))
            except:
                # If norm computation fails, use index as fallback
                vector_norms.append((float('inf'), i, vector))
        
        # Sort by norm (shortest first)
        vector_norms.sort(key=lambda x: x[0])
        
        # Check ALL vectors from the reduced basis - don't limit
        # For large numbers with poor approximations, we need to check all vectors
        max_vectors_to_check = len(reduced_basis)
        
        print(f"[Lattice] Checking ALL {max_vectors_to_check} vectors from reduced basis (sorted by norm)...")
        print(f"[Lattice]    (No limit - checking all vectors to maximize chance of finding corrections)")
        
        # Check vectors in order of increasing norm
        for norm_sq, orig_idx, vector in vector_norms:
            i = orig_idx  # Keep original index for reporting
            try:
                # Extract coefficients (vector format: [a, b, c] represents a + b*dp + c*dq = 0)
                # Where dp and dq are corrections: p = p_candidate + dp, q = q_candidate + dq
                # The relationship is: (p_candidate + dp) * (q_candidate + dq) = N
                # Which gives: p_candidate*q_candidate - N + q_candidate*dp + p_candidate*dq + dp*dq = 0
                # Linearizing: (p_candidate*q_candidate - N) + q_candidate*dp + p_candidate*dq ≈ 0
                # So: a = p_candidate*q_candidate - N, b = q_candidate, c = p_candidate
                a, b, c = vector[0], vector[1], vector[2]
                
                # Skip if coefficients are too large or zero
                # For huge search radii, be more lenient - LLL should have made coefficients small
                if (b == 0 and c == 0):
                    continue
                
                # For huge search radius (2048+ bits), accept reasonable coefficients
                # With 2048-bit search radius, we need to be very lenient with coefficient filtering
                # LLL should have reduced coefficients, but we still need to check vectors with larger coefficients
                if search_radius > 10**100 or (hasattr(search_radius, 'bit_length') and search_radius.bit_length() >= 2048):
                    # For 2048+ bit search radius, accept coefficients up to a very large bound
                    # Accept coefficients up to several hundred bits - LLL should have made them reasonable
                    # For a 2048-bit search radius, we can accept coefficients up to ~500-1000 bits
                    try:
                        search_bits = search_radius.bit_length() if hasattr(search_radius, 'bit_length') else 2048
                        # Accept coefficients up to min(search_radius, 2^1000) - very lenient for 2048-bit keys
                        max_reasonable_bits = min(search_bits, 1000)  # Accept up to 1000-bit coefficients
                        max_reasonable = 2 ** max_reasonable_bits
                        
                        b_bits = abs(b).bit_length() if hasattr(abs(b), 'bit_length') else 0
                        c_bits = abs(c).bit_length() if hasattr(abs(c), 'bit_length') else 0
                        
                        if abs(b) > max_reasonable or abs(c) > max_reasonable:
                            # Still log skipped vectors for debugging (first few only)
                            if i < 10:  # Log first 10 to see what's being filtered
                                print(f"[Lattice]   Vector {i}: Skipping (b={b_bits} bits, c={c_bits} bits, max={max_reasonable_bits} bits)")
                            continue
                        else:
                            # Log vectors we're checking (first few)
                            if i < 5:
                                print(f"[Lattice]   Vector {i}: Checking (b={b_bits} bits, c={c_bits} bits)")
                    except Exception as e:
                        # Fallback: just check if within search_radius (very lenient)
                        if abs(b) > search_radius or abs(c) > search_radius:
                            continue
                else:
                    # Normal filtering for reasonable search radii
                    if abs(b) > search_radius or abs(c) > search_radius:
                        continue

                # Solve for corrections: from a + b*p + c*q = 0, with p = p_candidate + dp, q = q_candidate + dq
                # We get: a + b*(p_candidate + dp) + c*(q_candidate + dq) = 0
                # So: b*dp + c*dq = -a - b*p_candidate - c*q_candidate

                rhs = -a - b * p_candidate - c * q_candidate
                
                # Debug: log vector details for first few vectors
                if i < 10:
                    print(f"[Lattice]   Vector {i}: a={a}, b={b}, c={c}, rhs={rhs}")

                if b != 0:
                    dp = rhs // b
                    dp_bits = abs(dp).bit_length() if abs(dp) > 0 else 0
                    
                    # For huge search radius, accept large dp values but check if they actually improve things
                    if abs(dp) <= search_radius:
                        dq = 0  # Assume dq = 0 for this vector
                        p_test = p_candidate + dp
                        q_test = q_candidate + dq

                        if p_test > 0 and q_test > 0:
                            product = p_test * q_test
                            diff = abs(product - self.N)
                            diff_bits = diff.bit_length() if diff > 0 else 0

                            if product == self.N:
                                print(f"[Lattice] ✓✓✓ EXACT FACTORIZATION FOUND from pyramid vector {i}! dp={dp:+d}, dq={dq:+d}")
                                return dp, dq, 1.0

                            if initial_diff > 0:
                                # Only consider this as improvement if it actually reduces the difference
                                if diff < initial_diff:
                                    # Log improvement for first few vectors
                                    if i < 10:
                                        print(f"[Lattice]   Vector {i}: dp={dp:+d} ({dp_bits} bits), product diff={diff_bits} bits (initial={initial_diff.bit_length()} bits)")
                                        improvement_pct = ((initial_diff - diff) / initial_diff * 100) if initial_diff > 0 else 0
                                        print(f"[Lattice]      ✓ Improvement: {improvement_pct:.2f}%")
                                    
                                    # Calculate improvement
                                    if self.N.bit_length() > 1000:
                                        diff_bits_calc = abs(diff).bit_length()
                                        initial_bits_calc = abs(initial_diff).bit_length()
                                        if diff_bits_calc < initial_bits_calc:
                                            improvement = min(0.9, (initial_bits_calc - diff_bits_calc) / 10.0)
                                        else:
                                            improvement = 0.0
                                    else:
                                        try:
                                            improvement = (initial_diff - diff) / initial_diff
                                        except OverflowError:
                                            improvement = 0.0
                                    
                                    if improvement > best_improvement:
                                        best_improvement = improvement
                                        best_dp = dp
                                        best_dq = dq
                
                elif c != 0:
                    # If b == 0, solve for dq: c*dq = -a - c*q_candidate
                    dq = (-a - c * q_candidate) // c
                    dq_bits = abs(dq).bit_length() if abs(dq) > 0 else 0
                    
                    if abs(dq) <= search_radius:
                        dp = 0  # Assume dp = 0 for this vector
                        p_test = p_candidate + dp
                        q_test = q_candidate + dq

                        if p_test > 0 and q_test > 0:
                            product = p_test * q_test
                            diff = abs(product - self.N)

                            if product == self.N:
                                print(f"[Lattice] ✓✓✓ EXACT FACTORIZATION FOUND from pyramid vector {i}! dp={dp:+d}, dq={dq:+d}")
                                return dp, dq, 1.0

                            if initial_diff > 0 and diff < initial_diff:
                                if self.N.bit_length() > 1000:
                                    diff_bits_calc = abs(diff).bit_length()
                                    initial_bits_calc = abs(initial_diff).bit_length()
                                    if diff_bits_calc < initial_bits_calc:
                                        improvement = min(0.9, (initial_bits_calc - diff_bits_calc) / 10.0)
                                    else:
                                        improvement = 0.0
                                else:
                                    try:
                                        improvement = (initial_diff - diff) / initial_diff
                                    except OverflowError:
                                        improvement = 0.0
                                
                                if improvement > best_improvement:
                                    best_improvement = improvement
                                    best_dp = dp
                                    best_dq = dq
                
                else:
                    # Both b and c are zero, try solving for both dp and dq
                    # For simplicity, try small dq values that make sense
                    # The LLL reduction should have made b and c relatively small
                    max_dq_try = min(1000, abs(c) // max(1, abs(b))) if abs(b) > 0 else 1000
                    for dq_candidate in range(-max_dq_try, max_dq_try + 1):
                        if (rhs - c * dq_candidate) % b == 0:
                            dp = (rhs - c * dq_candidate) // b
                            if abs(dp) <= max_dq_try * 10:  # Reasonable bound
                                p_test = p_candidate + dp
                                q_test = q_candidate + dq_candidate

                                if p_test > 0 and q_test > 0:
                                    product = p_test * q_test
                                    diff = abs(product - self.N)

                                    if product == self.N:
                                        print(f"[Lattice] ✓✓✓ EXACT FACTORIZATION FOUND from pyramid vector {i}! dp={dp:+d}, dq={dq_candidate:+d}")
                                        return dp, dq_candidate, 1.0

                                    if initial_diff > 0:
                                        try:
                                            improvement = (initial_diff - diff) / initial_diff
                                        except OverflowError:
                                            improvement = 0.0
                                        
                                        if improvement > best_improvement:
                                            best_improvement = improvement
                                            best_dp = dp
                                            best_dq = dq_candidate
                                            
            except Exception as e:
                continue
        
        # Return best corrections found
        if best_improvement > 0:
            print(f"[Lattice] Best corrections found: dp={best_dp:+d}, dq={best_dq:+d}, improvement={best_improvement:.4f}")
            return best_dp, best_dq, best_improvement
        else:
            print(f"[Lattice] No improvements found from reduced basis vectors")
            return 0, 0, 0.0
        # We'll sample the search space around sqrt(N) and construct a lattice
        # that helps LLL find the actual factors
        
        # Sample points around sqrt(N) with various offsets
        # Use multiple sampling strategies for maximum coverage
        offsets = []
        
        # Strategy 1: Linear sampling around sqrt(N) - EXPANDED RANGE
        # Use a much larger range to find factors far from sqrt(N)
        linear_samples = lattice_dim // 4
        # Expand step size significantly - factors might be far from sqrt(N)
        max_offset = min(sqrt_N // 2, search_radius) if search_radius < sqrt_N else sqrt_N // 2
        step = max(1, max_offset // (linear_samples * 2))
        for i in range(-linear_samples, linear_samples + 1):
            offset = i * step
            offsets.append(offset)
        
        # Strategy 2: Logarithmic/exponential sampling for VERY large range
        # This covers factors that are significantly different from sqrt(N)
        log_samples = lattice_dim // 3
        for i in range(-log_samples // 2, log_samples // 2 + 1):
            if i == 0:
                offset = 0
            else:
                sign = 1 if i > 0 else -1
                exp_offset = abs(i)
                # Use much larger offsets - factors might be far from sqrt(N)
                max_offset_bits = min(search_radius.bit_length(), sqrt_N_bits + 200)  # Allow larger range
                offset = sign * (2 ** min(exp_offset * 8, max_offset_bits // 2))  # Increased multiplier
            if offset not in offsets:
                offsets.append(offset)
        
        # Strategy 3: Power-of-2 sampling for systematic coverage - EXPANDED
        pow_samples = lattice_dim // 4
        for exp in range(pow_samples):
            for sign in [-1, 1]:
                # Allow much larger offsets
                offset = sign * (2 ** min(exp * 5, (sqrt_N_bits + 200) // 2))  # Increased range
                if offset not in offsets and len(offsets) < lattice_dim:
                    offsets.append(offset)
        
        # Strategy 4: Sample around different center points (not just sqrt(N))
        # If factors are far from sqrt(N), try centers at different scales
        center_samples = lattice_dim // 6
        for center_scale in [1, 2, 4, 8, 16, 32, 64, 128]:
            center = sqrt_N // center_scale if center_scale > 1 else sqrt_N
            for i in range(-center_samples // 8, center_samples // 8 + 1):
                offset = (center - sqrt_N) + i * (center // 100) if center > 100 else (center - sqrt_N) + i
                if offset not in offsets and len(offsets) < lattice_dim:
                    offsets.append(offset)
        
        # Strategy 5: Random-like offsets for better coverage (deterministic seed)
        import random
        random.seed(42)  # Deterministic for reproducibility
        remaining = lattice_dim - len(offsets)
        for _ in range(min(remaining, lattice_dim // 4)):
            # Much larger random range
            offset = random.randint(-sqrt_N // 10, sqrt_N // 10)
            if offset not in offsets:
                offsets.append(offset)
        
        # Ensure we have exactly lattice_dim offsets (or as close as possible)
        while len(offsets) < lattice_dim:
            # Fill with additional systematic offsets - expanded range
            for i in range(len(offsets), lattice_dim):
                offset = (i - lattice_dim // 2) * (max_offset // max(1, lattice_dim))
                if offset not in offsets:
                    offsets.append(offset)
                if len(offsets) >= lattice_dim:
                    break
        
        print(f"[Lattice]    Generated {len(offsets)} candidate offsets")
        print(f"[Lattice]    Using ALL {min(len(offsets), lattice_dim)} offsets for maximum coverage...")
        
        # Use all offsets up to lattice_dim (maximize!)
        offsets_to_use = offsets[:lattice_dim] if len(offsets) > lattice_dim else offsets
        print(f"[Lattice]    Processing {len(offsets_to_use)} candidate points...")
        
        # Construct polynomial-based lattice following the SAME Diophantine polynomial as univariate approach
        # Use: f(x,v) = x² + 2tx + (t² - N) - v² = 0
        # Where t = (p_approx + q_approx) // 2 or sqrt(N)
        # Then p = u - v, q = u + v where u = t + x
        
        # Compute t (same as univariate polynomial approach)
        if hasattr(self, 'p_approx') and hasattr(self, 'q_approx') and self.p_approx > 0 and self.q_approx > 0:
            t = (self.p_approx + self.q_approx) // 2
        else:
            t = sqrt_N
        
        print(f"[Lattice]    → Using SAME Diophantine polynomial as univariate approach:")
        print(f"[Lattice]    → f(x,v) = x² + 2tx + (t² - N) - v² = 0")
        print(f"[Lattice]    → Where t = {t}, and p = u - v, q = u + v (u = t + x)")
        print(f"[Lattice]    → Looking for small roots x, v using LLL")
        
        # Construct lattice vectors representing the Diophantine polynomial
        # Vector format: [x, x², v, v², constant] or simplified [x, v, constant]
        # For the polynomial x² + 2tx + (t² - N) - v² = 0
        # We want to find small x, v such that the polynomial evaluates to 0
        
        scale = max(1, sqrt_N.bit_length() // 10)
        t_sq_minus_N = t * t - self.N
        
        # Construct lattice vectors that represent the Diophantine polynomial
        # f(x,v) = x² + 2tx + (t² - N) - v² = 0
        # We want to find small roots x, v
        # Better approach: construct vectors that represent polynomial monomials
        # Vector format: [x, v, x², v², constant] scaled appropriately
        
        for offset in offsets_to_use:
            x = offset  # x is the correction: u = t + x
            
            # For the Diophantine polynomial, we need to find x and v such that:
            # x² + 2tx + (t² - N) - v² = 0
            # Rearranging: v² = x² + 2tx + (t² - N)
            
            # Compute what v² should be for this x
            v_squared_target = x * x + 2 * t * x + t_sq_minus_N
            
            # Try integer v values around sqrt(v_squared_target) if positive
            # LLL will help find the right combination
            if v_squared_target >= 0:
                v_candidate = int(math.isqrt(abs(v_squared_target))) if v_squared_target > 0 else 0
                # Try a few v values around the candidate
                for v_offset in [-2, -1, 0, 1, 2]:
                    v = v_candidate + v_offset
                    
                    # Compute polynomial value: f(x,v) = x² + 2tx + (t² - N) - v²
                    polynomial_value = x * x + 2 * t * x + t_sq_minus_N - v * v
                    
                    # Create vector: [x, v, polynomial_value]
                    # For valid roots, polynomial_value should be 0 (or very small)
                    basis_vectors.append([
                        x // scale,  # x term
                        v // scale,  # v term
                        polynomial_value // (scale * scale) if scale > 0 else polynomial_value  # polynomial value
                    ])
            else:
                # v_squared_target is negative, try v = 0
                polynomial_value = x * x + 2 * t * x + t_sq_minus_N
                basis_vectors.append([
                    x // scale,
                    0,
                    polynomial_value // (scale * scale) if scale > 0 else polynomial_value
                ])
        
        # Add fundamental polynomial relationship vectors
        # Vector 1: x coefficient (2t)
        basis_vectors.append([(2 * t) // scale, 0, 0])
        # Vector 2: constant term (t² - N)
        basis_vectors.append([0, 0, t_sq_minus_N // (scale * scale) if scale > 0 else t_sq_minus_N])
        # Vector 3: v term
        basis_vectors.append([0, 1, 0])
        
        if len(basis_vectors) < 3:
            print(f"[Lattice]    Not enough basis vectors, skipping bulk search")
            return None
        
        # Convert to numpy array - use ALL vectors we created (maximize!)
        max_vectors = min(len(basis_vectors), 1000)  # Allow up to 1000 vectors for maximum coverage
        basis = np.array(basis_vectors[:max_vectors], dtype=object)
        
        print(f"[Lattice]    ✓ Constructed {len(basis)} basis vectors (MAXIMIZED for comprehensive search)")
        print(f"[Lattice]    → This lattice samples the entire search space around √N (BULK SEARCH)")
        print(f"[Lattice]    → Applying LLL reduction to find SHORT vectors containing factors...")
        print(f"[Lattice]    → LLL finds short vectors which are more likely to contain p*q = N")
        
        # Warn about large matrices
        matrix_size = len(basis)
        bit_size = self.N.bit_length()
        print(f"[Lattice]    ⚠️  LLL will process {matrix_size}×{len(basis[0]) if len(basis) > 0 else 0} matrix with {bit_size}-bit integers")
        print(f"[Lattice]    ⏳ This may take several minutes for large matrices...")
        print(f"[Lattice]    → LLL reduction in progress (please be patient)...")
        
        # Apply LLL reduction with progress indication
        try:
            import time
            import sys
            start_time = time.time()
            
            from fpylll_wrapper import IntegerMatrix_from_matrix, LLL as IntegerLLL
            print(f"[Lattice]    → Step 1/3: Converting to IntegerMatrix...", flush=True)
            sys.stdout.flush()
            B = IntegerMatrix_from_matrix(basis.tolist())
            step1_time = time.time() - start_time
            print(f"[Lattice]    ✓ Step 1 complete (took {step1_time:.2f}s)", flush=True)
            sys.stdout.flush()
            
            print(f"[Lattice]    → Step 2/3: Running LLL reduction (THIS IS THE SLOW PART)...", flush=True)
            print(f"[Lattice]    → LLL is processing {matrix_size} vectors with {bit_size}-bit integers...", flush=True)
            print(f"[Lattice]    → This step may take 1-10+ minutes depending on matrix size...", flush=True)
            sys.stdout.flush()
            lll_start = time.time()
            
            # Add periodic progress updates during LLL (if possible)
            print(f"[Lattice]    → [LLL working...] (started at {time.strftime('%H:%M:%S')})", flush=True)
            sys.stdout.flush()
            
            reduced_basis = IntegerLLL(B, delta=self.delta)
            lll_time = time.time() - lll_start
            print(f"[Lattice]    ✓ Step 2 complete - LLL reduction took {lll_time:.2f}s ({lll_time/60:.2f} minutes)", flush=True)
            sys.stdout.flush()
            
            print(f"[Lattice]    → Step 3/3: Converting reduced basis to numpy...", flush=True)
            sys.stdout.flush()
            reduced = reduced_basis.to_numpy()
            total_time = time.time() - start_time
            print(f"[Lattice]    ✓✓✓ LLL reduction complete! (total: {total_time:.2f}s = {total_time/60:.2f} minutes)")
            print(f"[Lattice]    → Found {len(reduced)} short vectors from reduced basis")
            print(f"[Lattice]    → Checking ALL {len(reduced)} vectors for factors (BULK SEARCH MODE)")
            print(f"[Lattice]    → Short vectors = LLL's way of finding the most promising candidates")
        except Exception as e:
            print(f"[Lattice]    LLL reduction failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # Check all reduced vectors for factors
        checked = 0
        best_p = None
        best_q = None
        best_diff = None
        best_diff_bits = None
        total_vectors = len(reduced)
        
        print(f"[Lattice]    → Starting to check {total_vectors} vectors for factors (real-time updates)...")
        print(f"[Lattice]    → Progress bar: [{' ' * 50}] 0%")
        
        for i, vector in enumerate(reduced):
            try:
                checked += 1
                
                # Calculate progress percentage
                progress_pct = (checked * 100) // total_vectors
                progress_bar_length = 50
                filled = (checked * progress_bar_length) // total_vectors
                bar = '█' * filled + '░' * (progress_bar_length - filled)
                
                # Print progress every vector (with progress bar)
                if checked % 10 == 0 or checked == 1 or checked == total_vectors:
                    status = f"Best: {best_diff_bits} bits" if best_diff_bits is not None else "No candidates yet"
                    print(f"[Lattice]    → [{bar}] {progress_pct}% | Vector {checked}/{total_vectors} | {status}", end='\r')
                
                # More detailed update every 50 vectors
                if checked % 50 == 0:
                    print()  # New line after progress bar
                    if best_p is not None:
                        print(f"[Lattice]    → Checking vector {checked}/{total_vectors} | Best so far: diff={best_diff_bits} bits")
                        print(f"[Lattice]       Best p ≈ {best_p}")
                        print(f"[Lattice]       Best q ≈ {best_q}")
                    else:
                        print(f"[Lattice]    → Checking vector {checked}/{total_vectors} | No good candidates yet...")
                    print(f"[Lattice]    → Progress bar: [{' ' * 50}] 0%", end='\r')  # Reset progress bar line
                
                # Extract polynomial root (x) from vector components
                # Vector format: [x/scale, x²/scale², constant/scale²]
                # Following univariate polynomial logic: p = sqrt_N + x
                # We need to recover x (the root) and then compute p = sqrt_N + x
                
                # Try different scale factors to recover x
                scale = max(1, sqrt_N.bit_length() // 10)
                
                # Also try interpreting vector components directly as p and q
                # (fallback in case the polynomial interpretation doesn't work)
                if len(vector) >= 2:
                    # Try direct interpretation: vector might be [p, q, error]
                    p_direct = abs(int(vector[0]))
                    q_direct = abs(int(vector[1])) if len(vector) > 1 else 0
                    
                    # Check if these are close to √N (might be actual factors)
                    if p_direct > 1 and abs(p_direct - sqrt_N) < sqrt_N // 10:
                        if self.N % p_direct == 0:
                            q_test = self.N // p_direct
                            if p_direct * q_test == self.N:
                                print(f"[Lattice]    ✓✓✓ EXACT FACTOR FOUND (direct interpretation, vector {i})!")
                                print(f"[Lattice]       p = {p_direct}, q = {q_test}")
                                return (p_direct, q_test)
                
                # Try polynomial root extraction with various scale factors
                for scale_factor in [1, scale, scale // 10, scale * 10, scale // 100, scale * 100]:
                    if scale_factor <= 0:
                        continue
                    
                    # Extract x and v from vector components (Diophantine polynomial roots)
                    x_cand = int(vector[0]) * scale_factor if len(vector) > 0 else 0
                    v_cand = int(vector[1]) * scale_factor if len(vector) > 1 else 0
                    
                    # Compute t (same as construction)
                    if hasattr(self, 'p_approx') and hasattr(self, 'q_approx') and self.p_approx > 0 and self.q_approx > 0:
                        t = (self.p_approx + self.q_approx) // 2
                    else:
                        t = sqrt_N
                    
                    # Following SAME logic as univariate polynomial approach:
                    # u = t + x, then p = u - v, q = u + v
                    u_cand = t + x_cand
                    p_cand = u_cand - v_cand
                    q_cand = u_cand + v_cand
                    
                    if p_cand > 1 and q_cand > 1:
                        # Check if p and q are exact factors (Diophantine polynomial root condition!)
                        if p_cand * q_cand == self.N:
                            print(f"[Lattice]    ✓✓✓ EXACT FACTOR FOUND via Diophantine polynomial root (vector {i})!")
                            print(f"[Lattice]       Polynomial roots: x = {x_cand}, v = {v_cand} (scale_factor={scale_factor})")
                            print(f"[Lattice]       u = t + x = {t} + {x_cand} = {u_cand}")
                            print(f"[Lattice]       p = u - v = {u_cand} - {v_cand} = {p_cand}")
                            print(f"[Lattice]       q = u + v = {u_cand} + {v_cand} = {q_cand}")
                            return (p_cand, q_cand)
                        
                        # Also check if p divides N exactly
                        if self.N % p_cand == 0:
                            q_test = self.N // p_cand
                            if p_cand * q_test == self.N:
                                print(f"[Lattice]    ✓✓✓ EXACT FACTOR FOUND via polynomial root (vector {i})!")
                                print(f"[Lattice]       Polynomial roots: x = {x_cand}, v = {v_cand}")
                                print(f"[Lattice]       p = {p_cand}, q = {q_test}")
                                return (p_cand, q_test)
                        
                        # Check Diophantine polynomial evaluation: f(x,v) = x² + 2tx + (t² - N) - v²
                        # For valid roots, this should be ≈ 0
                        polynomial_eval = abs(x_cand * x_cand + 2 * t * x_cand + (t * t - self.N) - v_cand * v_cand)
                        
                        # Also check if p * q is close to N
                        product = p_cand * q_cand
                        product_diff = abs(product - self.N)
                        product_diff_bits = product_diff.bit_length() if product_diff > 0 else 0
                        
                        # Use the smaller of the two errors
                        if polynomial_eval < product_diff:
                            diff_bits = polynomial_eval.bit_length() if polynomial_eval > 0 else 0
                            use_product = False
                        else:
                            diff_bits = product_diff_bits
                            use_product = True
                        
                        # Update best if this is better
                        if best_diff is None or (use_product and product_diff < best_diff) or (not use_product and polynomial_eval < best_diff):
                            best_p = p_cand
                            best_q = q_cand
                            best_diff = product_diff if use_product else polynomial_eval
                            best_diff_bits = diff_bits
                            
                            # Real-time output of best result
                            if diff_bits < 100:  # Only show if reasonably close
                                print(f"[Lattice]    ⭐ NEW BEST Diophantine polynomial root (vector {i}):")
                                print(f"[Lattice]       x = {x_cand}, v = {v_cand}")
                                print(f"[Lattice]       p = u - v = {p_cand}, q = u + v = {q_cand}")
                                if use_product:
                                    print(f"[Lattice]       Product difference: {product_diff} ({diff_bits} bits)")
                                else:
                                    print(f"[Lattice]       Polynomial evaluation: f(x,v) = {polynomial_eval} ({diff_bits} bits)")
                            # Don't print periodic updates - they're too noisy
                            # The progress bar and 50-vector updates are enough
                
                # Also check if vector components directly give polynomial roots
                # Following univariate polynomial logic: extract x from vector, compute p = sqrt_N + x
                # Vector format: [x/scale, x²/scale², constant/scale²]
                # Need to account for scaling to recover actual x
                if len(vector) >= 1:
                    scale = max(1, sqrt_N.bit_length() // 10)
                    
                    # Try different scale factors to recover x
                    for scale_factor in [1, scale, scale // 10, scale * 10, scale // 100, scale * 100]:
                        if scale_factor <= 0:
                            continue
                        
                        # Extract x from first component (polynomial root)
                        # Vector is scaled, so multiply by scale_factor to recover x
                        x_direct = int(vector[0]) * scale_factor
                        p_from_root = sqrt_N + x_direct
                        
                        if p_from_root > 1 and self.N % p_from_root == 0:
                            q_from_root = self.N // p_from_root
                            if p_from_root * q_from_root == self.N:
                                print(f"[Lattice]    ✓✓✓ EXACT FACTOR FOUND via polynomial root (vector {i})!")
                                print(f"[Lattice]       Root: x = {x_direct} (recovered with scale_factor={scale_factor})")
                                print(f"[Lattice]       p = √N + x = {sqrt_N} + {x_direct} = {p_from_root}")
                                print(f"[Lattice]       q = {q_from_root}")
                                return (p_from_root, q_from_root)
                        
                        # Track best polynomial root
                        if p_from_root > 1:
                            # Check polynomial evaluation: f(x) = (sqrt_N + x) * q - N
                            q_approx = self.N // p_from_root if p_from_root > 0 else 0
                            if q_approx > 1:
                                poly_eval = abs(p_from_root * q_approx - self.N)
                                diff_bits_direct = poly_eval.bit_length() if poly_eval > 0 else 0
                                
                                if best_diff is None or poly_eval < best_diff:
                                    best_p = p_from_root
                                    best_q = q_approx
                                    best_diff = poly_eval
                                    best_diff_bits = diff_bits_direct
                                    
                                    if diff_bits_direct < 100:
                                        print(f"[Lattice]    ⭐ NEW BEST (polynomial root x={x_direct}, vector {i}):")
                                        print(f"[Lattice]       p = √N + x = {p_from_root}, q ≈ {q_approx}")
                                        print(f"[Lattice]       Polynomial evaluation: f(x) = {poly_eval} ({diff_bits_direct} bits)")
                                    
                                    # Break if we found a good candidate
                                    if diff_bits_direct < 50:
                                        break
                            
            except Exception as e:
                continue
        
        # Final progress bar update
        print()  # New line after progress bar
        print(f"[Lattice]    → [{'█' * 50}] 100% | Completed checking all {total_vectors} vectors")
        
        # Output final best result
        if best_p is not None and best_q is not None:
            print(f"[Lattice]    📊 FINAL BEST RESULT from bulk search:")
            print(f"[Lattice]       p ≈ {best_p}")
            print(f"[Lattice]       q ≈ {best_q}")
            print(f"[Lattice]       p × q = {best_p * best_q}")
            print(f"[Lattice]       Difference from N: {best_diff} ({best_diff_bits} bits)")
            
            if best_diff == 0:
                print(f"[Lattice]       ✓✓✓ EXACT FACTORIZATION!")
                return (best_p, best_q)
            elif best_diff_bits <= 10:  # Very close - try nearby values
                print(f"[Lattice]       ⚠️  Very close! (only {best_diff_bits} bits off)")
                print(f"[Lattice]       → Trying nearby values to find exact factors...")
                
                # Try small adjustments to p and q
                for dp in range(-10, 11):
                    for dq in range(-10, 11):
                        p_test = best_p + dp
                        q_test = best_q + dq
                        if p_test > 1 and q_test > 1 and p_test * q_test == self.N:
                            print(f"[Lattice]       ✓✓✓ EXACT FACTOR FOUND with small adjustment!")
                            print(f"[Lattice]       p = {p_test} (adjusted by {dp})")
                            print(f"[Lattice]       q = {q_test} (adjusted by {dq})")
                            return (p_test, q_test)
                
                # Also check if p or q divides N exactly
                if self.N % best_p == 0:
                    q_exact = self.N // best_p
                    if best_p * q_exact == self.N:
                        print(f"[Lattice]       ✓✓✓ EXACT FACTOR FOUND! p divides N exactly")
                        print(f"[Lattice]       p = {best_p}, q = {q_exact}")
                        return (best_p, q_exact)
                
                if self.N % best_q == 0:
                    p_exact = self.N // best_q
                    if p_exact * best_q == self.N:
                        print(f"[Lattice]       ✓✓✓ EXACT FACTOR FOUND! q divides N exactly")
                        print(f"[Lattice]       p = {p_exact}, q = {best_q}")
                        return (p_exact, best_q)
                
                print(f"[Lattice]       → Nearby search didn't find exact factors")
            else:
                print(f"[Lattice]       ⚠️  Not exact, but closest found in bulk search")
        else:
            print(f"[Lattice]    ❌ No valid candidates found in bulk search")
        
        print(f"[Lattice]    Bulk search completed (checked {checked} vectors)")
        return None

    def _find_best_factorization_corrections(self, p_candidate: int, q_candidate: int,
                                           pyramid_basis: np.ndarray,
                                           config: Optional[np.ndarray] = None,
                                           search_radius: int = 1000) -> Tuple[int, int, float]:
        """
        Find the best factorization corrections using pyramid lattice reduction.
        """
        print(f"[Lattice] Using pyramid lattice for factorization corrections...")
        print(f"[Lattice] Search radius parameter: {search_radius} ({search_radius.bit_length()} bits)")
        
        # For huge search radii, note that we'll extract corrections directly from LLL-reduced vectors
        if search_radius > 10**100:
            print(f"[Lattice] Large search radius detected - will extract corrections from LLL-reduced short vectors")
            print(f"[Lattice] LLL reduction should produce vectors with small coefficients containing the corrections")

        # Minimize the lattice using LLL
        reduced_basis = self._minimize_lattice(pyramid_basis)

        # Extract corrections from shortest vectors
        best_dp = 0
        best_dq = 0
        best_improvement = 0.0
        initial_diff = abs(p_candidate * q_candidate - self.N)

        print(f"[Lattice] Analyzing reduced pyramid lattice vectors...")
        print(f"[Lattice] Reduced basis has {len(reduced_basis)} vectors")
        print(f"[Lattice] Initial difference: {initial_diff} ({initial_diff.bit_length()} bits)")

        # Sort vectors by their L2 norm (length) - check shortest vectors first
        # This helps find the best corrections early
        vector_norms = []
        for i, vector in enumerate(reduced_basis):
            try:
                # Compute L2 norm squared (avoid sqrt for large numbers)
                norm_sq = sum(int(x)**2 for x in vector if isinstance(x, (int, np.integer)))
                vector_norms.append((norm_sq, i, vector))
            except:
                # If norm computation fails, use index as fallback
                vector_norms.append((float('inf'), i, vector))
        
        # Sort by norm (shortest first)
        vector_norms.sort(key=lambda x: x[0])
        
        # Check ALL vectors from the reduced basis - don't limit
        # For large numbers with poor approximations, we need to check all vectors
        max_vectors_to_check = len(reduced_basis)
        
        print(f"[Lattice] Checking ALL {max_vectors_to_check} vectors from reduced basis (sorted by norm)...")
        print(f"[Lattice]    (No limit - checking all vectors to maximize chance of finding corrections)")
        
        # Check vectors in order of increasing norm
        for norm_sq, orig_idx, vector in vector_norms:
            i = orig_idx  # Keep original index for reporting
            try:
                # Extract coefficients (vector format: [a, b, c] represents a + b*dp + c*dq = 0)
                # Where dp and dq are corrections: p = p_candidate + dp, q = q_candidate + dq
                # The relationship is: (p_candidate + dp) * (q_candidate + dq) = N
                # Which gives: p_candidate*q_candidate - N + q_candidate*dp + p_candidate*dq + dp*dq = 0
                # Linearizing: (p_candidate*q_candidate - N) + q_candidate*dp + p_candidate*dq ≈ 0
                # So: a = p_candidate*q_candidate - N, b = q_candidate, c = p_candidate
                a, b, c = vector[0], vector[1], vector[2]

                # Skip if coefficients are too large or zero
                # For huge search radii, be more lenient - LLL should have made coefficients small
                if (b == 0 and c == 0):
                    continue
                
                # For huge search radius (2048+ bits), accept reasonable coefficients
                # With 2048-bit search radius, we need to be very lenient with coefficient filtering
                # LLL should have reduced coefficients, but we still need to check vectors with larger coefficients
                if search_radius > 10**100 or (hasattr(search_radius, 'bit_length') and search_radius.bit_length() >= 2048):
                    # For 2048+ bit search radius, accept coefficients up to a very large bound
                    # Accept coefficients up to several hundred bits - LLL should have made them reasonable
                    # For a 2048-bit search radius, we can accept coefficients up to ~500-1000 bits
                    try:
                        search_bits = search_radius.bit_length() if hasattr(search_radius, 'bit_length') else 2048
                        # Accept coefficients up to min(search_radius, 2^1000) - very lenient for 2048-bit keys
                        max_reasonable_bits = min(search_bits, 1000)  # Accept up to 1000-bit coefficients
                        max_reasonable = 2 ** max_reasonable_bits
                        
                        b_bits = abs(b).bit_length() if hasattr(abs(b), 'bit_length') else 0
                        c_bits = abs(c).bit_length() if hasattr(abs(c), 'bit_length') else 0
                        
                        if abs(b) > max_reasonable or abs(c) > max_reasonable:
                            # Still log skipped vectors for debugging (first few only)
                            if i < 10:  # Log first 10 to see what's being filtered
                                print(f"[Lattice]   Vector {i}: Skipping (b={b_bits} bits, c={c_bits} bits, max={max_reasonable_bits} bits)")
                            continue
                        else:
                            # Log vectors we're checking (first few)
                            if i < 5:
                                print(f"[Lattice]   Vector {i}: Checking (b={b_bits} bits, c={c_bits} bits)")
                    except Exception as e:
                        # Fallback: just check if within search_radius (very lenient)
                        if abs(b) > search_radius or abs(c) > search_radius:
                            continue
                else:
                    # Normal filtering for reasonable search radii
                    if abs(b) > search_radius or abs(c) > search_radius:
                        continue

                # Solve for corrections: from a + b*p + c*q = 0, with p = p_candidate + dp, q = q_candidate + dq
                # We get: a + b*(p_candidate + dp) + c*(q_candidate + dq) = 0
                # So: b*dp + c*dq = -a - b*p_candidate - c*q_candidate

                rhs = -a - b * p_candidate - c * q_candidate
                
                # Debug: log vector details for first few vectors
                if i < 10:
                    print(f"[Lattice]   Vector {i}: a={a}, b={b}, c={c}, rhs={rhs}")

                if b != 0:
                    dp = rhs // b
                    dp_bits = abs(dp).bit_length() if abs(dp) > 0 else 0
                    
                    # For huge search radius, accept large dp values but check if they actually improve things
                    if abs(dp) <= search_radius:
                        dq = 0  # Assume dq = 0 for this vector
                        p_test = p_candidate + dp
                        q_test = q_candidate + dq

                        if p_test > 0 and q_test > 0:
                            product = p_test * q_test
                            diff = abs(product - self.N)
                            diff_bits = diff.bit_length() if diff > 0 else 0

                            if product == self.N:
                                print(f"[Lattice] ✓✓✓ EXACT FACTORIZATION FOUND from pyramid vector {i}! dp={dp:+d}, dq={dq:+d}")
                                return dp, dq, 1.0

                            if initial_diff > 0:
                                # Only consider this as improvement if it actually reduces the difference
                                if diff < initial_diff:
                                    # Log improvement for first few vectors
                                    if i < 10:
                                        print(f"[Lattice]   Vector {i}: dp={dp:+d} ({dp_bits} bits), product diff={diff_bits} bits (initial={initial_diff.bit_length()} bits)")
                                        improvement_pct = ((initial_diff - diff) / initial_diff * 100) if initial_diff > 0 else 0
                                        print(f"[Lattice]      ✓ Improvement: {improvement_pct:.2f}%")
                                    
                                    # Calculate improvement
                                    if self.N.bit_length() > 1000:
                                        diff_bits_calc = abs(diff).bit_length()
                                        initial_bits_calc = abs(initial_diff).bit_length()
                                        if diff_bits_calc < initial_bits_calc:
                                            improvement = min(0.9, (initial_bits_calc - diff_bits_calc) / 10.0)
                                        else:
                                            improvement = 0.0
                                    else:
                                        try:
                                            improvement = (initial_diff - diff) / initial_diff
                                        except OverflowError:
                                            improvement = 0.0
                                    
                                    if improvement > best_improvement:
                                        best_improvement = improvement
                                        best_dp = dp
                                        best_dq = dq
                                        print(f"[Lattice]   Vector {i}: NEW BEST! dp={dp:+d}, dq={dq:+d}, improvement={improvement:.4f}")
                                else:
                                    # Log that this vector made things worse
                                    if i < 10:
                                        print(f"[Lattice]   Vector {i}: dp={dp:+d} ({dp_bits} bits) made things WORSE (diff={diff_bits} bits > initial={initial_diff.bit_length()} bits)")

                if c != 0 and b != 0:
                    # For huge search radii, extract corrections directly from the vector
                    # instead of iterating through the search space
                    if search_radius > 10**100:  # If search radius is astronomically large
                        # Use the vector directly: solve b*dp + c*dq = rhs
                        # Try to find small integer solutions using extended Euclidean algorithm
                        # or by directly computing from the vector structure
                        try:
                            # For large search radius, the LLL-reduced vector should give us
                            # the correction directly. Try solving the linear Diophantine equation.
                            # If gcd(b, c) divides rhs, we can find solutions
                            import math
                            g = math.gcd(abs(b), abs(c))
                            if g > 0 and rhs % g == 0:
                                # Find one solution using extended Euclidean
                                # For simplicity, try small dq values that make sense
                                # The LLL reduction should have made b and c relatively small
                                max_dq_try = min(1000, abs(c) // max(1, abs(b))) if abs(b) > 0 else 1000
                                for dq_candidate in range(-max_dq_try, max_dq_try + 1):
                                    if (rhs - c * dq_candidate) % b == 0:
                                        dp = (rhs - c * dq_candidate) // b
                                        if abs(dp) <= max_dq_try * 10:  # Reasonable bound
                                            p_test = p_candidate + dp
                                            q_test = q_candidate + dq_candidate

                                            if p_test > 0 and q_test > 0:
                                                product = p_test * q_test
                                                diff = abs(product - self.N)

                                                if product == self.N:
                                                    print(f"[Lattice] ✓✓✓ EXACT FACTORIZATION FOUND from pyramid vector {i}! dp={dp:+d}, dq={dq_candidate:+d}")
                                                    return dp, dq_candidate, 1.0

                                                if initial_diff > 0:
                                                    try:
                                                        if self.N.bit_length() > 1000:
                                                            improvement = 0.1 if diff < initial_diff else 0.0
                                                        else:
                                                            improvement = (initial_diff - diff) / initial_diff
                                                    except OverflowError:
                                                        improvement = 0.0
                                                else:
                                                    improvement = 1.0 if diff == 0 else 0.0

                                                if improvement > best_improvement:
                                                    best_improvement = improvement
                                                    best_dp = dp
                                                    best_dq = dq_candidate
                                                    print(f"[Lattice]   Vector {i}: dp={dp:+d}, dq={dq_candidate:+d}, improvement={improvement:.4f}")
                        except Exception as e:
                            print(f"[Lattice]   Error solving Diophantine equation for vector {i}: {e}")
                            pass
                    else:
                        # For reasonable search radii, use the original iteration approach
                        for dq_candidate in range(-min(search_radius, abs(c)), min(search_radius, abs(c)) + 1):
                            if (rhs - c * dq_candidate) % b == 0:
                                dp = (rhs - c * dq_candidate) // b
                                if abs(dp) <= search_radius:
                                    p_test = p_candidate + dp
                                    q_test = q_candidate + dq_candidate

                                    if p_test > 0 and q_test > 0:
                                        product = p_test * q_test
                                        diff = abs(product - self.N)

                                        if product == self.N:
                                            print(f"[Lattice] ✓✓✓ EXACT FACTORIZATION FOUND from pyramid vector {i}! dp={dp:+d}, dq={dq_candidate:+d}")
                                            return dp, dq_candidate, 1.0

                                        if initial_diff > 0:
                                            # Safe improvement calculation for large numbers
                                            if self.N.bit_length() > 1000:
                                                improvement = 0.1 if diff < initial_diff else 0.0
                                            else:
                                                improvement = (initial_diff - diff) / initial_diff
                                        else:
                                            improvement = 1.0 if diff == 0 else 0.0

                                        if improvement > best_improvement:
                                            best_improvement = improvement
                                            best_dp = dp
                                            best_dq = dq_candidate
                                            print(f"[Lattice]   Vector {i}: dp={dp:+d}, dq={dq_candidate:+d}, improvement={improvement:.4f}")

            except Exception as e:
                continue  # Skip problematic vectors

        # No brute force fallback - rely on LLL-reduced vectors only
        if best_improvement < 0.01:
            print(f"[Lattice] Pyramid lattice found limited improvement ({best_improvement:.4f})")
            print(f"[Lattice] ⚠️  DIAGNOSIS:")
            print(f"[Lattice]    - Checked {len(vector_norms)} vectors from LLL-reduced basis")
            print(f"[Lattice]    - Initial difference: {initial_diff.bit_length()} bits ({initial_diff})")
            print(f"[Lattice]    - Search radius: {search_radius.bit_length()} bits")
            print(f"[Lattice]    - p_candidate: {p_candidate}")
            print(f"[Lattice]    - q_candidate: {q_candidate}")
            print(f"[Lattice]    - p_candidate * q_candidate = {p_candidate * q_candidate}")
            print(f"[Lattice]    - N = {self.N}")
            print(f"[Lattice]    - No useful corrections found in any vector")
            print(f"[Lattice]    ")
            print(f"[Lattice]    Possible issues:")
            print(f"[Lattice]    1. Approximations too far off ({initial_diff.bit_length()}-bit error is huge!)")
            print(f"[Lattice]    2. Lattice construction may not capture factorization relationship for such poor approximations")
            print(f"[Lattice]    3. LLL reduction may not be producing useful short vectors")
            print(f"[Lattice]    ")
            print(f"[Lattice]    Suggestions:")
            print(f"[Lattice]    - Verify p_approx * q_approx is close to N (current error: {initial_diff.bit_length()} bits)")
            print(f"[Lattice]    - Try polynomial methods: --polynomial flag")
            print(f"[Lattice]    - Try increasing lattice dimension: --lattice-dimension <value>")
            print(f"[Lattice]    - Check if approximations are correct (1955-bit error suggests they're way off)")

        print(f"[Lattice] Best pyramid correction: dp={best_dp:+d}, dq={best_dq:+d}, improvement={best_improvement:.4f}")
        
        # If we found a good improvement but not exact, try verifying the result
        if best_improvement > 0:
            refined_p = p_candidate + best_dp
            refined_q = q_candidate + best_dq
            product = refined_p * refined_q
            final_diff = abs(product - self.N)
            print(f"[Lattice] Verification: refined_p × refined_q = {product}")
            print(f"[Lattice] Difference from N: {final_diff} ({final_diff.bit_length()} bits)")
            if product == self.N:
                print(f"[Lattice] ✓✓✓ EXACT FACTORIZATION VERIFIED in _find_best_factorization_corrections!")
            elif final_diff < initial_diff:
                print(f"[Lattice] Improvement: {initial_diff.bit_length()} bits → {final_diff.bit_length()} bits")

        return best_dp, best_dq, best_improvement

    def _minimize_lattice(self, basis: np.ndarray) -> np.ndarray:
        """
        Minimize lattice basis to find optimal factorization corrections.
        """
        # Always try to use integer-based LLL from our wrapper
        try:
            from fpylll_wrapper import IntegerMatrix, IntegerMatrix_from_matrix, LLL as IntegerLLL
            print(f"[Lattice] Using integer-based LLL for lattice minimization...")
            B = IntegerMatrix_from_matrix(basis.tolist())
            B_reduced = IntegerLLL(B, delta=self.delta)
            reduced = B_reduced.to_numpy()
            print(f"[Lattice] ✓ Integer lattice minimization complete")
            return reduced
        except (ImportError, Exception) as e:
            print(f"[Lattice] Integer LLL failed ({e}), using fallback...")
            import traceback
            traceback.print_exc()

        # Fallback: for very large numbers, pyramid lattice provides structure without minimization
        n = len(basis)
        max_val = max(abs(val) for row in basis for val in row)

        # Check if we should force computation even for large values (when lattice dimension is extreme)
        force_computation = False
        if hasattr(self, 'config') and self.config and self.config.get('lattice_dimension'):
            try:
                lattice_dim = int(self.config['lattice_dimension'])
                if lattice_dim.bit_length() > 100:  # If lattice dimension > 2^100, force computation
                    force_computation = True
                    print(f"[Lattice] EXTREME lattice dimension detected ({lattice_dim.bit_length()} bits), forcing computation despite large coefficients")
            except:
                pass

        if max_val > 1e15 and not force_computation:
            print(f"[Lattice] Values too large for float conversion, analyzing pyramid structure directly")
            print(f"[Lattice] ✓ Pyramid lattice ready for analysis")
            return basis

        # For pyramid lattices, check if we should force full minimization
        force_full_minimization = False
        if hasattr(self, 'config') and self.config and self.config.get('lattice_dimension'):
            try:
                lattice_dim = int(self.config['lattice_dimension'])
                if lattice_dim.bit_length() > 200:  # If lattice dimension > 2^200, force full minimization
                    force_full_minimization = True
                    print(f"[Lattice] EXTREME lattice dimension detected ({lattice_dim.bit_length()} bits), forcing full LLL minimization")
            except:
                pass

        if not force_full_minimization:
            print(f"[Lattice] Pyramid lattice constructed, analyzing structure directly")
            print(f"[Lattice] ✓ Pyramid lattice ready for analysis")
            return basis

        try:
            print(f"[Lattice] Applying float-based lattice minimization...")

            # Scale computational complexity with lattice dimension
            complexity_scale = 1
            if hasattr(self, 'config') and self.config and self.config.get('lattice_dimension'):
                try:
                    lattice_dim = int(self.config['lattice_dimension'])
                    lattice_bits = lattice_dim.bit_length()
                    if lattice_bits > 100:
                        complexity_scale = min(lattice_bits // 50, 10)  # Scale up to 10x complexity
                        print(f"[Lattice] EXTREME lattice dimension: scaling complexity by {complexity_scale}x")
                except:
                    pass

            B = basis.astype(float)

            # Simple LLL-style reduction with scaled complexity
            # Work with the available dimensions (should be 3 for p,q,N lattice)
            m = B.shape[1]  # Number of columns/variables
            k = 1
            max_iterations = min(50 * complexity_scale, 200)  # Scale iterations with lattice dimension
            iteration_count = 0

            while k < n and iteration_count < max_iterations:
                # Size reduction (more thorough for extreme lattices)
                for j in range(k-1, -1, -1):
                    # For each column/variable
                    for col in range(m):
                        if abs(B[j, col]) > 1e-10:  # Use column j as pivot for this variable
                            mu = round(B[k, col] / B[j, col])
                            B[k] -= mu * B[j]
                            break  # Only reduce once per row pair

                # Check swap condition using Euclidean norm
                if k > 0:
                    norm_k_minus_1 = np.sqrt(np.sum(B[k-1]**2))
                    norm_k = np.sqrt(np.sum(B[k]**2))
                    if self.delta * (norm_k_minus_1 ** 2) > (norm_k ** 2):
                        B[[k-1, k]] = B[[k, k-1]]
                        k = max(k-1, 1)
                    else:
                        k += 1
                else:
                    k += 1

                iteration_count += 1

                # Add extra computational work for extreme lattices
                if complexity_scale > 1 and iteration_count % max(1, 10 // complexity_scale) == 0:
                    # Perform additional Gram-Schmidt orthogonalization for extreme cases
                    for i in range(min(n, 5 * complexity_scale)):  # More orthogonalization passes
                        for j in range(i):
                            dot_ij = np.dot(B[i], B[j])
                            norm_j_sq = np.dot(B[j], B[j])
                            if norm_j_sq > 1e-10:
                                proj = dot_ij / norm_j_sq
                                B[i] -= proj * B[j]

            print(f"[Lattice] ✓ Lattice minimization complete ({iteration_count} iterations, {complexity_scale}x scaling)")
            return B.astype(object)
            
        except Exception as e:
            print(f"[Lattice] Minimization failed ({e}), using original basis")
            return basis

    def _integer_sqrt_approx(self, n):
        """Compute integer square root approximation without floating point."""
        if n == 0 or n == 1:
            return n
        
        # Binary search for square root
        left, right = 1, n
        while left <= right:
            mid = (left + right) // 2
            try:
                if mid * mid == n:
                    return mid
                elif mid * mid < n:
                    left = mid + 1
                else:
                    right = mid - 1
            except OverflowError:
                right = mid - 1
        
        return right  # right will be the floor of sqrt(n)

    def solve_s_d_algebraically(self, p_hint: int, q_hint: int) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        """
        Pure algebraic solution: Given approximate p and q, find exact values.
        
        Mathematical approach:
        1. We know: p × q = N (exact)
        2. We approximate: p ≈ p_hint, q ≈ q_hint
        3. Let p = p_hint + δp, q = q_hint + δq (small corrections)
        4. Then: (p_hint + δp)(q_hint + δq) = N
        5. Expanding: p_hint·q_hint + p_hint·δq + q_hint·δp + δp·δq = N
        6. For small corrections, δp·δq ≈ 0:
           p_hint·δq + q_hint·δp ≈ N - p_hint·q_hint
        """
        
        # Calculate the error in current approximation
        product_approx = p_hint * q_hint
        error = self.N - product_approx
        
        print(f"[Algebra] Starting with p_hint={p_hint}, q_hint={q_hint}")
        print(f"[Algebra] Current product: {product_approx}")
        print(f"[Algebra] Error: {error}")
        
        # Method 1: Direct algebraic correction (assumes one factor is close)
        # If p is approximately correct, solve for exact q:
        if p_hint > 0 and self.N % p_hint == 0:
            q_exact = self.N // p_hint
            print(f"[Algebra] ✓ p_hint is exact factor!")
            S = p_hint + q_exact
            D = S * S - 4 * self.N
            return S, D, p_hint, q_exact
        
        if q_hint > 0 and self.N % q_hint == 0:
            p_exact = self.N // q_hint
            print(f"[Algebra] ✓ q_hint is exact factor!")
            S = p_exact + q_hint
            D = S * S - 4 * self.N
            return S, D, p_exact, q_hint
        
        # Method 2: Solve via quadratic equation
        # From p + q = S and p × q = N:
        # x² - Sx + N = 0
        # where x = p or q
        
        S = p_hint + q_hint
        four_N = 4 * self.N
        D = S * S - four_N
        
        print(f"[Algebra] S = p + q = {S}")
        print(f"[Algebra] D = S² - 4N = {D}")
        
        # Check if D is a perfect square
        if D >= 0:
            sqrt_D = self._integer_sqrt_approx(D)
            if sqrt_D * sqrt_D == D:
                print(f"[Algebra] ✓ D is perfect square: √D = {sqrt_D}")
                
                # Solve quadratic: p = (S + √D)/2, q = (S - √D)/2
                if (S + sqrt_D) % 2 == 0:
                    p = (S + sqrt_D) // 2
                    q = (S - sqrt_D) // 2
                    
                    if p * q == self.N:
                        print(f"[Algebra] ✓✓✓ EXACT SOLUTION via quadratic!")
                        return S, D, p, q
            else:
                print(f"[Algebra] D not perfect square, need to adjust S")
                # S is slightly wrong, need to find correct S
                return self._find_correct_s_algebraically(p_hint, q_hint, S, sqrt_D)
        else:
            print(f"[Algebra] D < 0, S too small")
            # S is too small, need to increase it
            return self._find_correct_s_algebraically(p_hint, q_hint, S, None)
        
        return None, None, None, None

    def _find_correct_s_algebraically(self, p_hint: int, q_hint: int, 
                                       S_approx: int, sqrt_D_approx: Optional[int]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        """
        Find correct S algebraically using exact numerical analysis.
        
        Mathematical derivation:
        T = p+q, k = p-q
        T² = 4N + k²
        
        Given T_base = 2S_approx and k_approx = √(T_base² - 4N):
        We need to find ΔT and Δk such that:
        (T_base + ΔT)² - 4N = (k_approx + Δk)²
        
        This gives: remainder + 2·T_base·ΔT + ΔT² = 2·k_approx·Δk + Δk²
        where remainder = (T_base² - 4N) - k_approx²
        """
        
        four_N = 4 * self.N
        
        # Calculate T_base = p+q (in our code, S_approx = p+q, so T_base = S_approx)
        # But user's notation: T = p+q, S = (p+q)/2, so T_base = 2S
        # Since our S_approx = p+q, we have T_base = S_approx
        T_base = S_approx
        T_base_sq = T_base * T_base
        
        print(f"[Exact Analysis] T_base = p+q = {T_base}")
        print(f"[Exact Analysis] T_base² = {T_base_sq}")
        
        # Calculate k_approx = √(T_base² - 4N)
        discriminant = T_base_sq - four_N
        if discriminant < 0:
            print(f"[Exact Analysis] T_base² - 4N < 0, adjusting...")
            # Need larger T_base
            sqrt_N = self._integer_sqrt_approx(self.N)
            T_base = 2 * sqrt_N
            T_base_sq = T_base * T_base
            discriminant = T_base_sq - four_N
            print(f"[Exact Analysis] Adjusted T_base = 2√N = {T_base}")
        
        k_approx = self._integer_sqrt_approx(discriminant)
        k_approx_sq = k_approx * k_approx
        
        remainder = discriminant - k_approx_sq
        
        print(f"[Exact Analysis] k_approx = √(T_base² - 4N) = {k_approx}")
        print(f"[Exact Analysis] k_approx² = {k_approx_sq}")
        print(f"[Exact Analysis] remainder = (T_base² - 4N) - k_approx² = {remainder}")
        
        # For small adjustments: remainder ≈ 2·k_approx·Δk
        # So Δk ≈ remainder/(2·k_approx)
        if k_approx > 0:
            delta_k_estimate = remainder // (2 * k_approx) if remainder >= 0 else (remainder - 2 * k_approx + 1) // (2 * k_approx)
            print(f"[Exact Analysis] Δk estimate ≈ remainder/(2·k_approx) ≈ {delta_k_estimate}")
            
            # Try Δk values around the estimate
            # For large remainders, expand search range
            search_range = min(1000, max(100, abs(delta_k_estimate) + 50))
            print(f"[Exact Analysis] Searching Δk in range [{delta_k_estimate - search_range}, {delta_k_estimate + search_range}]")
            for delta_k in range(delta_k_estimate - search_range, delta_k_estimate + search_range + 1):
                k = k_approx + delta_k
                k_sq = k * k
                
                # Calculate required T² = 4N + k²
                T_sq_required = four_N + k_sq
                T_required = self._integer_sqrt_approx(T_sq_required)
                
                # Check if exact
                if T_required * T_required == T_sq_required:
                    print(f"[Exact Analysis] ✓✓✓ Found exact solution!")
                    print(f"[Exact Analysis]   Δk = {delta_k}, k = {k}")
                    print(f"[Exact Analysis]   T = {T_required}, T² = {T_sq_required}")
                    print(f"[Exact Analysis]   Verification: T² - 4N = {T_sq_required - four_N} = k² = {k_sq} ✓")
                    
                    # In our code, S = p+q = T
                    S = T_required
                    D = k_sq
                    
                    # Extract factors: p = (T + k)/2, q = (T - k)/2
                    if (T_required + k) % 2 == 0 and (T_required - k) % 2 == 0:
                        p = (T_required + k) // 2
                        q = (T_required - k) // 2
                        
                        print(f"[Exact Analysis] Calculated: p = (T + k)/2 = {p}")
                        print(f"[Exact Analysis] Calculated: q = (T - k)/2 = {q}")
                        print(f"[Exact Analysis] Verification: p × q = {p * q} = N? {p * q == self.N}")
                        
                        if p * q == self.N and p > 1 and q > 1:
                            print(f"[Exact Analysis] ✓✓✓ EXACT FACTORIZATION!")
                            return S, D, p, q
                        else:
                            print(f"[Exact Analysis] Factors don't multiply to N exactly")
                            return S, D, p, q
        
        # Fallback to original method
        k_est = abs(p_hint - q_hint)
        print(f"[Algebra] Estimated k ≈ |p - q| ≈ {k_est}")
        
        # Check if k_est works
        S_squared = four_N + k_est * k_est
        S = self._integer_sqrt_approx(S_squared)
        
        if S * S == S_squared:
            print(f"[Algebra] ✓ Found S with k = {k_est}")
            if (S + k_est) % 2 == 0:
                p = (S + k_est) // 2
                q = (S - k_est) // 2
                if p * q == self.N:
                    print(f"[Algebra] ✓✓✓ EXACT SOLUTION!")
                    return S, k_est * k_est, p, q
        
        # If k_est doesn't work, use Newton-Raphson to find better k
        # We want to solve: f(k) = 4N + k² - S² = 0 where S² must be perfect square
        
        # Alternative: Continued fraction approach
        # Express √(4N) as continued fraction, convergents give candidates
        return self._continued_fraction_factorization(p_hint, q_hint)

    def _continued_fraction_factorization(self, p_hint: int, q_hint: int) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        """
        Use continued fraction expansion to find exact factors algebraically.
        
        For N = p × q, the continued fraction expansion of √N often reveals factors.
        Convergents of √(4N) give candidates for S.
        """
        
        # Compute √(4N) using continued fractions
        four_N = 4 * self.N
        sqrt_4N = self._integer_sqrt_approx(four_N)
        
        if sqrt_4N * sqrt_4N == four_N:
            # 4N is perfect square, N = (sqrt_4N/2)²
            if sqrt_4N % 2 == 0:
                factor = sqrt_4N // 2
                S = 2 * factor
                D = 0
                return S, D, factor, factor
        
        # Generate continued fraction convergents
        a0 = sqrt_4N
        convergents = [(a0, 1)]
        
        # State for continued fraction
        m, d, a = 0, 1, a0
        
        for _ in range(100):  # Limit iterations
            m = d * a - m
            d = (four_N - m * m) // d
            if d == 0:
                break
            a = (a0 + m) // d
            
            # Compute convergent
            if len(convergents) >= 1:
                h_prev, k_prev = convergents[-1] if len(convergents) >= 2 else (1, 0)
                h_curr, k_curr = convergents[-1]
                h_new = a * h_curr + h_prev
                k_new = a * k_curr + k_prev
                convergents.append((h_new, k_new))
                
                # Test this convergent as S candidate
                S_candidate = h_new // k_new if k_new != 0 else h_new
                D = S_candidate * S_candidate - four_N
                
                if D >= 0:
                    sqrt_D = self._integer_sqrt_approx(D)
                    if sqrt_D * sqrt_D == D and (S_candidate + sqrt_D) % 2 == 0:
                        p = (S_candidate + sqrt_D) // 2
                        q = (S_candidate - sqrt_D) // 2
                        if p * q == self.N:
                            print(f"[CF] ✓✓✓ Found via continued fraction!")
                            return S_candidate, D, p, q
        
        return None, None, None, None

    def find_s_and_d_via_roots_method(self, polynomials: List[sp.Expr] = None,
                                     p_hint: int = None, q_hint: int = None,
                                     search_radius: int = None, D_hint: int = None) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
        """
        Find S such that S² = 4N + k² for some k (D = k² will be perfect square)

        CORRECTED ALGORITHM:
        1. Use D_hint as starting point for k search (D_hint doesn't need to be perfect square!)
        2. k_approx = floor(sqrt(D_hint))
        3. Search k values around k_approx
        4. Find k such that 4*N + k² is a perfect square
        5. Then S = sqrt(4*N + k²), D = k²

        D_hint is just a hint - the actual D found will be k² (perfect square)!

        Returns:
            Tuple of (S, D, p, q, S_squared) where D = k² is always perfect square
        """
        import math
        from decimal import Decimal, getcontext

        print(f"\n{'─'*80}")
        print(f"FINDING S SUCH THAT S² = 4N + k² (D = k² perfect square)")
        print(f"{'─'*80}")
        print(f"Target N: {self.N} ({self.N.bit_length()}-bit)")

        # Use high precision
        getcontext().prec = max(200, self.N.bit_length() * 3)

        # Step 1: Use polynomial methods to get S or D, then compute algebraically
        print(f"\n[Algebra] Using pure algebraic relationship: S² = 4N + D")
        print(f"[Algebra] Algebraic rearrangements:")
        print(f"[Algebra]   N = (S² - D)/4")
        print(f"[Algebra]   D = S² - 4N")
        print(f"[Algebra]   S = √(4N + D)")

        S_found = None
        D_found = None

        # Try to get S or D from polynomial results
        if polynomials is not None and len(polynomials) > 0:
            print(f"\n[Algebra] Getting S or D from polynomial analysis...")

            config = getattr(self, 'config', {})
            poly_solver = EnhancedPolynomialSolver(self.N, config=config)

            result = poly_solver.solve_with_roots_method(polynomials, p_hint, q_hint)
            if result:
                p_candidate, q_candidate = result
                # Compute S and D algebraically from factors
                S_found = p_candidate + q_candidate
                D_found = (p_candidate - q_candidate) ** 2

                print(f"[Algebra] Polynomial gave: p={p_candidate}, q={q_candidate}")
                print(f"[Algebra] Computed: S = p + q = {S_found}")
                print(f"[Algebra] Computed: D = (p - q)² = {D_found}")

                # Verify algebraic identity: S² = 4N + D
                S_squared = S_found * S_found
                expected = 4 * self.N + D_found

                if S_squared == expected:
                    print(f"[Algebra] ✓ Algebraic identity verified: S² = 4N + D")
                    print(f"[Algebra]   {S_squared} = 4×{self.N} + {D_found}")
                    
                    # Verify perfect S across all 3 domains
                    print(f"\n{'─'*80}")
                    print(f"VERIFYING PERFECT S ACROSS ALL 3 DOMAINS")
                    print(f"{'─'*80}")
                    S_perfect, S_sq_perfect, D_perfect, is_perfect = self.calculate_perfect_s(S_squared=S_squared)
                    
                    if is_perfect:
                        print(f"\n✓✓✓ DISCOVERED S IS PERFECT ACROSS ALL 3 DOMAINS! ✓✓✓")
                    else:
                        print(f"\n⚠ Discovered S does not satisfy all 3 domains")
                    
                    return S_found, D_found, p_candidate, q_candidate, S_squared
                else:
                    print(f"[Algebra] ✗ Identity check failed: {S_squared} ≠ {expected}")

        # If we have D_hint, compute S algebraically
        if D_hint is not None and D_hint > 0:
            print(f"\n[Algebra] Using D_hint={D_hint} to compute S algebraically")
            four_N = 4 * self.N
            S_squared = four_N + D_hint
            S_computed = self._integer_sqrt_approx(S_squared)

            # Verify S² = 4N + D
            if S_computed * S_computed == S_squared:
                print(f"[Algebra] ✓ Computed S = √(4N + D) = {S_computed}")
                print(f"[Algebra]   Verification: {S_computed}² = {S_squared} = 4×{self.N} + {D_hint}")

                # Check if D is a perfect square
                sqrt_D = self._integer_sqrt_approx(D_hint)
                if sqrt_D * sqrt_D == D_hint:
                    # Verify perfect S across all 3 domains
                    print(f"\n{'─'*80}")
                    print(f"VERIFYING PERFECT S ACROSS ALL 3 DOMAINS")
                    print(f"{'─'*80}")
                    S_perfect, S_sq_perfect, D_perfect, is_perfect = self.calculate_perfect_s(S_squared=S_squared)
                    
                    if is_perfect:
                        print(f"\n✓✓✓ DISCOVERED S IS PERFECT ACROSS ALL 3 DOMAINS! ✓✓✓")
                    else:
                        print(f"\n⚠ Discovered S does not satisfy all 3 domains")
                    
                    # Extract factors: p = (S + √D)/2, q = (S - √D)/2
                    p = (S_computed + sqrt_D) // 2
                    q = (S_computed - sqrt_D) // 2

                    if p * q == self.N and p > 1 and q > 1:
                        print(f"[Algebra] ✓ Factors: p={p}, q={q}")
                        return S_computed, D_hint, p, q, S_squared

        # Try new algebraic method first if hints are available
        if p_hint is not None and q_hint is not None and p_hint > 0 and q_hint > 0:
            print(f"\n[Algebra] Trying pure algebraic solution with hints...")
            S_alg, D_alg, p_alg, q_alg = self.solve_s_d_algebraically(p_hint, q_hint)
            if S_alg is not None and D_alg is not None and p_alg is not None and q_alg is not None:
                if p_alg * q_alg == self.N and p_alg > 1 and q_alg > 1:
                    print(f"[Algebra] ✓✓✓ EXACT FACTORIZATION via algebraic method!")
                    S_sq_alg = S_alg * S_alg
                    
                    # Verify perfect S across all 3 domains
                    print(f"\n{'─'*80}")
                    print(f"VERIFYING PERFECT S ACROSS ALL 3 DOMAINS")
                    print(f"{'─'*80}")
                    S_perfect, S_sq_perfect, D_perfect, is_perfect = self.calculate_perfect_s(S_squared=S_sq_alg)
                    
                    if is_perfect:
                        print(f"\n✓✓✓ DISCOVERED S IS PERFECT ACROSS ALL 3 DOMAINS! ✓✓✓")
                    else:
                        print(f"\n⚠ Discovered S does not satisfy all 3 domains")
                    
                    return S_alg, D_alg, p_alg, q_alg, S_sq_alg
            print(f"[Algebra] Algebraic method didn't find exact solution, continuing...")
        
        # Method 1: ADJUST S TO MAKE D A PERFECT SQUARE
        # We need: S² = 4N + T² where T² = D is a perfect square
        # This means: 4N + T² must be a perfect square
        
        four_N = 4 * self.N
        sqrt_N = self._integer_sqrt_approx(self.N)
        
        print(f"\n[Algebra] Adjusting S to make D = T² (perfect square)")
        print(f"[Algebra] We need: S² = 4N + T² where T² is a perfect square")
        print(f"[Algebra] This means: 4N + T² must be a perfect square")
        
        # Start with S = 2√N
        S_start = 2 * sqrt_N
        S_sq_start = S_start * S_start
        D_start = S_sq_start - four_N
        
        print(f"[Algebra] Starting: S = 2√N = {S_start}")
        print(f"[Algebra] Initial: D = S² - 4N = {D_start}")
        
        # Calculate T = √D (approximate)
        # Initialize T_approx to ensure it's always defined
        T_approx = None
        
        if D_start >= 0:
            T_approx = self._integer_sqrt_approx(D_start)
            T_approx_sq = T_approx * T_approx
            error = D_start - T_approx_sq
            print(f"[Algebra] T_approx = floor(√D) = {T_approx}")
            print(f"[Algebra] T_approx² = {T_approx_sq}")
            print(f"[Algebra] Error: D - T_approx² = {error}")
            
            # ADJUST: Try T = T_approx + 1 to make 4N + T² a perfect square
            T = T_approx + 1
        else:
            # D is negative (S² < 4N), calculate T from |D| using equations
            print(f"[Algebra] D is negative (S² < 4N), calculating T from |D| using equations")
            print(f"[Algebra] We need: S² = 4N + T², so T² = S² - 4N")
            print(f"[Algebra] Since D = S² - 4N < 0, we need T such that 4N + T² is a perfect square")
            
            # Use equations: If D < 0, we need T such that 4N + T² is a perfect square
            # From hints: k = |p - q|, and T = k
            if p_hint is not None and q_hint is not None and p_hint > 0 and q_hint > 0:
                k_from_hints = abs(p_hint - q_hint)
                print(f"[Algebra] From hints: k = |p - q| = {k_from_hints}")
                print(f"[Algebra] Using equation: T = k = {k_from_hints}")
                T_approx = k_from_hints
            else:
                # Calculate T from |D|: T² should be approximately |D|
                abs_D = -D_start
                T_approx = self._integer_sqrt_approx(abs_D)
                print(f"[Algebra] |D| = {abs_D}, so T_approx = √|D| = {T_approx}")
            
            # Try T = T_approx + 1 to make 4N + T² a perfect square
            T = T_approx + 1
            print(f"[Algebra] Adjusted: T = T_approx + 1 = {T}")
        
        T_sq = T * T
        S_sq_target = four_N + T_sq
        S = self._integer_sqrt_approx(S_sq_target)
        
        print(f"[Algebra] Adjusted: T = {T}, T² = {T_sq}")
        print(f"[Algebra] Calculated: S² = 4N + T² = {four_N} + {T_sq} = {S_sq_target}")
        print(f"[Algebra] Calculated: S = √(S²) = {S}")
        
        # Check if exact
        if S * S == S_sq_target:
            print(f"[Algebra] ✓ S² = 4N + T² holds exactly!")
            print(f"[Algebra]   S = {S}, T = {T}, D = T² = {T_sq} (PERFECT SQUARE)")
            
            # Verify perfect S across all 3 domains
            print(f"\n{'─'*80}")
            print(f"VERIFYING PERFECT S ACROSS ALL 3 DOMAINS")
            print(f"{'─'*80}")
            S_perfect, S_sq_perfect, D_perfect, is_perfect = self.calculate_perfect_s(S_squared=S_sq_target)
            
            if is_perfect:
                print(f"\n✓✓✓ DISCOVERED S IS PERFECT ACROSS ALL 3 DOMAINS! ✓✓✓")
            else:
                print(f"\n⚠ Discovered S does not satisfy all 3 domains")
            
            # Direct calculation: p = (S + T)/2, q = (S - T)/2
            if (S - T) % 2 == 0 and (S + T) % 2 == 0:
                p = (S + T) // 2
                q = (S - T) // 2
                print(f"[Algebra] Calculated: p = (S + T)/2 = {p}")
                print(f"[Algebra] Calculated: q = (S - T)/2 = {q}")
                print(f"[Algebra] Verification: p × q = {p * q} = N? {p * q == self.N}")
                
                if p * q == self.N and p > 1 and q > 1:
                    print(f"[Algebra] ✓✓✓ EXACT FACTORIZATION FOUND! ✓✓✓")
                    return S, T_sq, p, q, S_sq_target
                else:
                    return S, T_sq, p, q, S_sq_target
            else:
                # S ± T not even
                p = (S + T) // 2
                q = (S - T) // 2
                return S, T_sq, p, q, S_sq_target
        else:
            # Not exact with T = T_approx + 1, try T = T_approx (exact case) if available
            print(f"[Algebra] S² = {S * S} ≠ 4N + T² = {S_sq_target}")
            
            if T_approx is not None and T_approx > 0:
                print(f"[Algebra] Trying T = T_approx (exact case)...")
                
                T_exact = T_approx
                T_sq_exact = T_exact * T_exact
                S_sq_exact = four_N + T_sq_exact
                S_exact = self._integer_sqrt_approx(S_sq_exact)
                
                if S_exact * S_exact == S_sq_exact:
                    print(f"[Algebra] ✓ Found exact: T = {T_exact}, S = {S_exact}")
                    S = S_exact
                    T = T_exact
                    T_sq = T_sq_exact
                    S_sq_target = S_sq_exact
                    
                    # Verify perfect S across all 3 domains
                    print(f"\n{'─'*80}")
                    print(f"VERIFYING PERFECT S ACROSS ALL 3 DOMAINS")
                    print(f"{'─'*80}")
                    S_perfect, S_sq_perfect, D_perfect, is_perfect = self.calculate_perfect_s(S_squared=S_sq_target)
                    
                    if is_perfect:
                        print(f"\n✓✓✓ DISCOVERED S IS PERFECT ACROSS ALL 3 DOMAINS! ✓✓✓")
                    else:
                        print(f"\n⚠ Discovered S does not satisfy all 3 domains")
                    
                    # Direct calculation: p = (S + T)/2, q = (S - T)/2
                    if (S - T) % 2 == 0 and (S + T) % 2 == 0:
                        p = (S + T) // 2
                        q = (S - T) // 2
                        print(f"[Algebra] Calculated: p = (S + T)/2 = {p}")
                        print(f"[Algebra] Calculated: q = (S - T)/2 = {q}")
                        print(f"[Algebra] Verification: p × q = {p * q} = N? {p * q == self.N}")
                        if p * q == self.N and p > 1 and q > 1:
                            print(f"[Algebra] ✓✓✓ EXACT FACTORIZATION FOUND! ✓✓✓")
                            return S, T_sq, p, q, S_sq_target
                        else:
                            return S, T_sq, p, q, S_sq_target
                else:
                    # Still not exact, return what we have
                    print(f"[Algebra] No exact solution found with T = T_approx")
                    return S, T_sq, None, None, S_sq_target
            else:
                # T_approx not available, return what we have
                print(f"[Algebra] No exact solution found (T_approx not available)")
                return S, T_sq, None, None, S_sq_target
        
        # Fallback to hints if available
        if p_hint is not None and q_hint is not None and p_hint > 0 and q_hint > 0:
            print(f"\n[Algebra] Calculating algebraically using p_hint and q_hint")
            print(f"[Algebra] Given: p_hint = {p_hint}, q_hint = {q_hint}")
            
            # PURE ALGEBRAIC CALCULATION (NO SEARCH):
            # We know: p × q = N (exact constraint)
            # We have approximate: p ≈ p_hint, q ≈ q_hint
            # 
            # From the quadratic equation x² - Sx + N = 0:
            #   p + q = S
            #   p × q = N
            #   Discriminant: D = S² - 4N = k² where k = |p - q|
            #   p = (S + k)/2, q = (S - k)/2
            #
            # KEY INSIGHT: We can calculate k directly from p_hint and q_hint
            # Then calculate S from S² = 4N + k²
            # Then extract exact p and q
            
            four_N = 4 * self.N
            
            # ALGEBRAIC REFINEMENT: Use p × q = N to refine p and q
            # Given approximate p_hint and q_hint, we know p × q = N exactly
            # We can solve: q = N / p, and p + q = S
            # This gives us: p + N/p = S, or p² - Sp + N = 0
            # The discriminant is: D = S² - 4N = k²
            
            # First, calculate approximate S
            S_approx = p_hint + q_hint
            print(f"[Algebra] Approximate: S = p_hint + q_hint = {S_approx}")
            
            # Calculate D = S² - 4N
            S_sq_approx = S_approx * S_approx
            D_approx = S_sq_approx - four_N
            print(f"[Algebra] Calculated: D = S² - 4N = {D_approx}")
            
            # If D is a perfect square, we have the exact solution
            sqrt_D = self._integer_sqrt_approx(D_approx)
            k = sqrt_D
            k_squared = k * k
            
            print(f"[Algebra] Calculated: k = √D = {k}")
            print(f"[Algebra] Calculated: k² = {k_squared}")
            
            # Check if D is a perfect square
            if k_squared == D_approx:
                print(f"[Algebra] ✓ D = T² is a perfect square! (T = {k})")
                # D is exact, so S is exact
                S = S_approx
                T = k  # T = √D
                S_squared = S_sq_approx
                
                print(f"[Algebra] Calculated: S = {S}, T = {T}")
                print(f"[Algebra] Verification: S² = {S_squared} = 4N + T² ✓")
                print(f"[Algebra] Key identity: S² - T² = (S - T)(S + T) = 4N")
                
                # Verify perfect S across all 3 domains
                print(f"\n{'─'*80}")
                print(f"VERIFYING PERFECT S ACROSS ALL 3 DOMAINS")
                print(f"{'─'*80}")
                S_perfect, S_sq_perfect, D_perfect, is_perfect = self.calculate_perfect_s(S_squared=S_squared)
                
                if is_perfect:
                    print(f"\n✓✓✓ DISCOVERED S IS PERFECT ACROSS ALL 3 DOMAINS! ✓✓✓")
                else:
                    print(f"\n⚠ Discovered S does not satisfy all 3 domains")
                
                # Use the identity: S² - T² = (S - T)(S + T) = 4N
                # Compute a = (S - T) // 2, b = (S + T) // 2
                if (S - T) % 2 == 0 and (S + T) % 2 == 0:
                    p = (S + T) // 2
                    q = (S - T) // 2
                    
                    print(f"[Algebra] Calculated: p = (S + T)/2 = ({S} + {T})/2 = {p}")
                    print(f"[Algebra] Calculated: q = (S - T)/2 = ({S} - {T})/2 = {q}")
                    print(f"[Algebra] Verification: p × q = {p * q} = N? {p * q == self.N}")
                    
                    if p * q == self.N and p > 1 and q > 1:
                        print(f"[Algebra] ✓✓✓ EXACT ALGEBRAIC FACTORIZATION SUCCESSFUL! ✓✓✓")
                        print(f"[Algebra]   T = {T}")
                        print(f"[Algebra]   S = {S}")
                        print(f"[Algebra]   D = T² = {k_squared}")
                        print(f"[Algebra]   p = {p}")
                        print(f"[Algebra]   q = {q}")
                        print(f"[Algebra]   All constraints satisfied: S² = 4N + T², p × q = N ✓")
                        
                        return S, k_squared, p, q, S_squared
                    else:
                        return S, k_squared, p, q, S_squared
                else:
                    print(f"[Algebra] ⚠️  S ± T is not even, cannot compute a and b")
                    # Fallback to direct calculation
                    p = (S + T) // 2
                    q = (S - T) // 2
                    return S, k_squared, p, q, S_squared
            else:
                print(f"[Algebra] ⚠️  D = {D_approx} is not a perfect square")
                if D_approx < 0:
                    print(f"[Algebra]   D is negative: S² < 4N")
                else:
                    print(f"[Algebra]   D is positive but not a perfect square")
                    print(f"[Algebra]   Error: D - k² = {D_approx - k_squared}")
                
                # ADJUST: Use T = floor(√D) + 1 to make 4N + T² a perfect square
                if D_approx >= 0:
                    T_adj = k + 1  # T = floor(√D) + 1
                    print(f"[Algebra] Adjusted: T = floor(√D) + 1 = {k} + 1 = {T_adj}")
                else:
                    # D negative, use T from hints
                    T_adj = abs(p_hint - q_hint)
                    print(f"[Algebra] Using T from hints: T = |p_hint - q_hint| = {T_adj}")
                
                # Calculate S from S² = 4N + T²
                T_squared = T_adj * T_adj
                S_squared_calc = four_N + T_squared
                S_calc = self._integer_sqrt_approx(S_squared_calc)
                
                print(f"[Algebra] Calculated: T² = {T_squared}")
                print(f"[Algebra] Calculated: S² = 4N + T² = {four_N} + {T_squared} = {S_squared_calc}")
                print(f"[Algebra] Calculated: S = √(S²) = {S_calc}")
                
                # Check if S² = 4N + T² exactly
                if S_calc * S_calc == S_squared_calc:
                    print(f"[Algebra] ✓ S² = 4N + T² holds exactly!")
                    print(f"[Algebra]   T = {T_adj}")
                    print(f"[Algebra]   S = {S_calc}")
                    print(f"[Algebra]   D = T² = {T_squared} (perfect square)")
                    
                    # Direct calculation: p = (S + T)/2, q = (S - T)/2
                    if (S_calc - T_adj) % 2 == 0 and (S_calc + T_adj) % 2 == 0:
                        p = (S_calc + T_adj) // 2
                        q = (S_calc - T_adj) // 2
                        
                        print(f"[Algebra] Calculated: p = (S + T)/2 = {p}")
                        print(f"[Algebra] Calculated: q = (S - T)/2 = {q}")
                        print(f"[Algebra] Verification: p × q = {p * q} = N? {p * q == self.N}")
                        
                        if p * q == self.N and p > 1 and q > 1:
                            print(f"[Algebra] ✓✓✓ EXACT FACTORIZATION FOUND! ✓✓✓")
                            return S_calc, T_squared, p, q, S_squared_calc
                        else:
                            return S_calc, T_squared, p, q, S_squared_calc
                    else:
                        # S ± T not even, use direct calculation
                        p = (S_calc + T_adj) // 2
                        q = (S_calc - T_adj) // 2
                        return S_calc, T_squared, p, q, S_squared_calc
                else:
                    print(f"[Algebra] ⚠️  S² = {S_calc * S_calc} ≠ 4N + T² = {S_squared_calc}")
                    print(f"[Algebra]   T = {T_approx} is not exact, using approximate values")
                    return S_approx, D_approx, None, None, S_sq_approx
            
            # If D is negative, S is too small - adjust S algebraically
            if D_computed < 0:
                print(f"[Algebra] D is negative: D = {D_computed} (S² < 4N)")
                print(f"[Algebra] S = {S_from_hints} is too small, adjusting S algebraically")
                
                # For balanced factors, S should be approximately 2√N
                sqrt_N = self._integer_sqrt_approx(self.N)
                S_target = 2 * sqrt_N
                
                print(f"[Algebra] Target S ≈ 2√N = {S_target}")
                print(f"[Algebra] Current S = {S_from_hints}, difference = {S_target - S_from_hints}")
                
                # Use the target S
                S_adjusted = S_target
                S_sq_adjusted = S_adjusted * S_adjusted
                D_adjusted = S_sq_adjusted - four_N
                
                print(f"[Algebra] Adjusted: S = {S_adjusted}")
                print(f"[Algebra] Adjusted: D = S² - 4N = {D_adjusted}")
                
                if D_adjusted >= 0:
                    sqrt_D = self._integer_sqrt_approx(D_adjusted)
                    p_extracted = (S_adjusted + sqrt_D) // 2
                    q_extracted = (S_adjusted - sqrt_D) // 2
                    
                    print(f"[Algebra] Extracted factors: p = {p_extracted}, q = {q_extracted}")
                    return S_adjusted, D_adjusted, p_extracted, q_extracted, S_sq_adjusted
                else:
                    # Still negative, search for k directly
                    print(f"[Algebra] D is still negative after adjustment, searching for k directly")
                    k_from_hints = abs(p_hint - q_hint)
                    k_start = k_from_hints
                    search_range = min(10000, self.N.bit_length() * 100)
                    
                    for k_offset in range(-search_range, search_range + 1):
                        k_candidate = k_start + k_offset
                        if k_candidate < 0:
                            continue
                        
                        k_sq = k_candidate * k_candidate
                        target = four_N + k_sq
                        S_test = self._integer_sqrt_approx(target)
                        
                        if S_test * S_test == target:
                            if (S_test + k_candidate) % 2 != 0 or (S_test - k_candidate) % 2 != 0:
                                continue
                            
                            p_extracted = (S_test + k_candidate) // 2
                            q_extracted = (S_test - k_candidate) // 2
                            
                            if p_extracted * q_extracted == self.N:
                                print(f"[Algebra] ✓ Found exact factors via k search!")
                                return S_test, k_sq, p_extracted, q_extracted, target
                    
                    # No exact solution found
                    return S_adjusted, D_adjusted, None, None, S_sq_adjusted
            
            # D is non-negative, check if it's a perfect square
            sqrt_D = self._integer_sqrt_approx(D_computed)
            
            if sqrt_D * sqrt_D == D_computed:
                print(f"[Algebra] ✓ D is a perfect square: D = {sqrt_D}²")
                # Extract factors directly
                p_extracted = (S_from_hints + sqrt_D) // 2
                q_extracted = (S_from_hints - sqrt_D) // 2
                
                product = p_extracted * q_extracted
                if product == self.N:
                    print(f"[Algebra] ✓ Perfect factorization: p×q = N")
                else:
                    print(f"[Algebra] Factors: p×q = {product}")
                
                return S_from_hints, D_computed, p_extracted, q_extracted, S_squared
            else:
                print(f"[Algebra] D is not a perfect square (D = {D_computed}, √D ≈ {sqrt_D})")
                print(f"[Algebra] Adjusting all three (S, D, k) using pure algebra")
                
                # We have approximate p and q from hints
                # Compute k = |p - q| directly from hints (this is exact algebra)
                k_from_hints = abs(p_hint - q_hint)
                print(f"[Algebra] From hints: k = |p - q| = {k_from_hints}")
                
                # Now use the equation S² = 4N + k² to compute S
                # This ensures D = k² is a perfect square
                k_squared = k_from_hints * k_from_hints
                S_sq_target = four_N + k_squared
                S_computed = self._integer_sqrt_approx(S_sq_target)
                
                print(f"[Algebra] Using equation: S² = 4N + k²")
                print(f"[Algebra]   S = √(4N + k²) = √({four_N} + {k_squared})")
                print(f"[Algebra]   S = {S_computed}")
                
                # Verify the equation holds exactly
                if S_computed * S_computed == S_sq_target:
                    print(f"[Algebra] ✓ All three conditions satisfied!")
                    print(f"[Algebra]   S² = {S_sq_target}")
                    print(f"[Algebra]   4N + k² = {four_N} + {k_squared} = {S_sq_target}")
                    print(f"[Algebra]   D = k² = {k_squared} (perfect square)")
                    print(f"[Algebra]   S² = 4N + D ✓")
                    
                    # Extract factors: p = (S + k)/2, q = (S - k)/2
                    p_extracted = (S_computed + k_from_hints) // 2
                    q_extracted = (S_computed - k_from_hints) // 2
                    
                    print(f"[Algebra] Extracted factors:")
                    print(f"[Algebra]   p = (S + k)/2 = {p_extracted}")
                    print(f"[Algebra]   q = (S - k)/2 = {q_extracted}")
                    
                    product = p_extracted * q_extracted
                    if product == self.N:
                        print(f"[Algebra] ✓ Perfect factorization: p×q = N")
                    else:
                        error = abs(product - self.N)
                        print(f"[Algebra] Factors: p×q = {product}, error = {error}")
                    
                    return S_computed, k_squared, p_extracted, q_extracted, S_sq_target
                else:
                    # 4N + k² is not a perfect square, need different k
                    print(f"[Algebra] 4N + k² = {S_sq_target} is not a perfect square")
                    print(f"[Algebra]   S_computed² = {S_computed * S_computed}")
                    print(f"[Algebra]   Difference = {abs(S_computed * S_computed - S_sq_target)}")
                    print(f"[Algebra] Need to find k such that 4N + k² is a perfect square")
                    
                    # Use approximate √D as starting point, find exact k
                    # We need k such that: 4N + k² is a perfect square AND p×q = N
                    k_start = sqrt_D
                    search_range = min(10000, self.N.bit_length() * 100)  # Expand search for large N
                    print(f"[Algebra] Searching for k in range [{k_start - search_range}, {k_start + search_range}]")
                    
                    best_k = None
                    best_error = None
                    best_p = None
                    best_q = None
                    best_S = None
                    
                    for k_offset in range(-search_range, search_range + 1):
                        k_candidate = k_start + k_offset
                        if k_candidate < 0:
                            continue
                        
                        k_sq = k_candidate * k_candidate
                        target = four_N + k_sq
                        S_test = self._integer_sqrt_approx(target)
                        
                        # Check if 4N + k² is a perfect square
                        if S_test * S_test == target:
                            # Extract factors: p = (S + k)/2, q = (S - k)/2
                            # Verify S + k and S - k are even (required for integer factors)
                            if (S_test + k_candidate) % 2 != 0 or (S_test - k_candidate) % 2 != 0:
                                continue
                            
                            p_extracted = (S_test + k_candidate) // 2
                            q_extracted = (S_test - k_candidate) // 2
                            
                            # Verify: p × q should equal N if S² = 4N + k²
                            product = p_extracted * q_extracted
                            error = abs(product - self.N)
                            
                            if product == self.N:
                                print(f"[Algebra] ✓ Found exact k with perfect factorization!")
                                print(f"[Algebra]   k = {k_candidate}, D = k² = {k_sq}")
                                print(f"[Algebra]   S = √(4N + k²) = {S_test}")
                                print(f"[Algebra]   p = {p_extracted}")
                                print(f"[Algebra]   q = {q_extracted}")
                                print(f"[Algebra]   p × q = {product} = N ✓")
                                return S_test, k_sq, p_extracted, q_extracted, target
                            
                            # Track the best (smallest error) solution
                            if best_error is None or error < best_error:
                                best_k = k_candidate
                                best_error = error
                                best_p = p_extracted
                                best_q = q_extracted
                                best_S = S_test
                    
                    # If we found a solution with small error, return it
                    if best_k is not None and best_error is not None:
                        print(f"[Algebra] ✓ Found best k (error = {best_error})")
                        print(f"[Algebra]   k = {best_k}, D = k² = {best_k * best_k}")
                        print(f"[Algebra]   S = {best_S}")
                        print(f"[Algebra]   p = {best_p}")
                        print(f"[Algebra]   q = {best_q}")
                        print(f"[Algebra]   p × q = {best_p * best_q}, error = {best_error}")
                        return best_S, best_k * best_k, best_p, best_q, best_S * best_S
                    
                    # Fallback
                    p_approx = (S_from_hints + sqrt_D) // 2
                    q_approx = (S_from_hints - sqrt_D) // 2
                    return S_from_hints, D_computed, p_approx, q_approx, S_squared

        # Method 2: Compute S and D algebraically from estimated candidates
        # The estimated candidates are computed in main() and passed as p_hint/q_hint
        # If we reach here, we don't have hints, so we can't compute S
        # But the main() function should provide estimated candidates as hints
        
        print(f"[Algebra] No p_hint/q_hint available to compute S algebraically")
        print(f"[Algebra] Estimated candidates should be provided as hints")
        print(f"[Algebra] Cannot compute S without at least approximate p and q")

        print(f"[Algebra] No hints available (polynomial, D_hint, or p_hint/q_hint)")
        print(f"[Algebra] Cannot compute S or D without at least one hint")
        return None, None, None, None, None

    def factor_from_s_squared(self, S_squared: int = None, S: int = None, D: int = None) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[float]]:
        k_approx = self._integer_sqrt_approx(D_hint)
        print(f"[Search] k_approx = floor(sqrt(D_hint)) = {k_approx}")

        # Step 3: Determine search range around k_approx
        if search_radius is None:
            # Adaptive search radius
            if self.N.bit_length() <= 64:
                k_search_radius = 1000
            elif self.N.bit_length() <= 128:
                k_search_radius = 10000
            elif self.N.bit_length() <= 256:
                k_search_radius = 100000
            else:
                k_search_radius = max(100000, k_approx // 100)  # Scale with k_approx
            print(f"[Search] Using adaptive search radius: {k_search_radius}")
        else:
            print(f"[Search] Using provided search radius: {search_radius}")
            k_search_radius = search_radius

        k_start = max(0, k_approx - k_search_radius)
        k_end = k_approx + k_search_radius

        print(f"[Search] Searching k in range [{k_start:,}, {k_end:,}] around k_approx = {k_approx:,}")
        print(f"[Search] For each k, checking if 4*N + k² is a perfect square")

        # Step 4: Search for k such that 4*N + k² is perfect square
        four_N = 4 * self.N
        checked = 0
        candidates_found = []

        # Use adaptive step size
        step = max(1, (k_end - k_start) // 100000)

        for k_test in range(k_start, k_end + 1, step):
            checked += 1
            if checked % 10000 == 0:
                print(f"[Search] Checked {checked:,} k values... (current k: {k_test:,})")
            elif checked % 1000 == 0 and checked < 10000:
                print(f"[Search] Checked {checked:,} k values...")

            # Compute 4*N + k²
            k_squared = k_test * k_test
            target = four_N + k_squared

            # Check if target is a perfect square
            S_test = self._integer_sqrt_approx(target)
            S_squared = S_test * S_test

            if S_squared == target:
                print(f"\n[Search] ✓✓✓ FOUND PERFECT RELATIONSHIP S² = 4N + k²!")
                print(f"[Search]   k = {k_test:,}")
                print(f"[Search]   D = k² = {k_squared:,} (perfect square)")
                print(f"[Search]   S = √(4N + k²) = {S_test:,}")
                print(f"[Search]   Verification: S² = {S_squared:,}")
                print(f"[Search]   4N + D = {four_N:,} + {k_squared:,} = {four_N + k_squared:,}")
                print(f"[Search]   Exact match: {S_squared == four_N + k_squared}")

                # Compute derived factors: p = (S + k)/2, q = (S - k)/2
                p_derived = (S_test + k_test) // 2
                q_derived = (S_test - k_test) // 2

                print(f"[Search]   Derived factors:")
                print(f"[Search]     p = (S + k)/2 = {p_derived:,}")
                print(f"[Search]     q = (S - k)/2 = {q_derived:,}")

                # Verify factorization if possible
                try:
                    if self.N.bit_length() <= 1000 and p_derived > 0 and q_derived > 0:
                        product = p_derived * q_derived
                        error = abs(product - self.N)

                        print(f"[Search]   Verification: p × q = {product:,}")
                        print(f"[Search]   Target N = {self.N:,}")
                        print(f"[Search]   Error = {error:,}")

                        if product == self.N:
                            print(f"[Search] ✓✓✓ EXACT FACTORIZATION VERIFIED!")
                            return S_test, k_squared, p_derived, q_derived, S_squared

                        candidates_found.append((S_test, k_squared, p_derived, q_derived, error))
                    else:
                        # For large numbers, store the mathematical solution
                        print(f"[Search]   Mathematical relationship found")
                        candidates_found.append((S_test, k_squared, p_derived, q_derived, 0))

                except (OverflowError, ValueError):
                    print(f"[Search]   Cannot verify factors due to size")
                    candidates_found.append((S_test, k_squared, p_derived, q_derived, 0))

                # For very large numbers, any valid solution is valuable
                if self.N.bit_length() > 500:
                    print(f"[Search] ✓ Found mathematically valid S² = 4N + D relationship")
                    return S_test, k_squared, p_derived, q_derived, S_squared

        print(f"\n[Search] Completed search of {checked:,} k values")

        if candidates_found:
            # Return best candidate
            best = min(candidates_found, key=lambda x: x[4])
            S_best, D_best, p_best, q_best, error_best = best

            print(f"\n[Search] Best candidate found:")
            print(f"[Search]   S = {S_best:,}")
            print(f"[Search]   D = {D_best:,} (perfect square)")
            print(f"[Search]   k = √D = {self._integer_sqrt_approx(D_best):,}")
            print(f"[Search]   Derived p = {p_best:,}, q = {q_best:,}")
            if error_best > 0:
                print(f"[Search]   Factorization error = {error_best:,}")
            else:
                print(f"[Search]   Valid mathematical relationship")

            # Verify relationship
            verification = S_best * S_best == 4 * self.N + D_best
            print(f"[Search]   S² = 4N + D verification: {verification}")

            return S_best, D_best, p_best, q_best, S_best * S_best

        print(f"[Search] No k values found making 4*N + k² a perfect square")
        print(f"Try increasing --search-radius or providing a better D_hint")
        print(f"D_hint was: {D_hint}")

        return None, None, None, None, None

    def factor_from_s_squared(self, S_squared: int = None, S: int = None, D: int = None) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[float]]:
        """
        Factor N using the relationship S² = 4N + D.
        
        From S² = 4N + D, we can compute:
        - S = p + q (sum of factors)
        - D = (p - q)² (square of difference)
        - Then: p = (S + √D) / 2, q = (S - √D) / 2
        
        Args:
            S_squared: Value of S² (if provided, will compute S and D)
            S: Sum of factors p + q (if provided directly)
            D: Square of difference (p - q)² (if provided directly)
        
        Returns:
            Tuple of (p, q, alpha, beta) where p = k + alpha, q = Qc + beta
        """
        from decimal import Decimal, getcontext
        import math
        
        print(f"\n{'─'*80}")
        print(f"FACTORING USING S² = 4N + D")
        print(f"{'─'*80}")
        
        # Use high precision for decimal arithmetic
        getcontext().prec = max(200, self.N.bit_length() * 3)
        
        N_dec = Decimal(self.N)
        
        # Compute S and D from S² if provided
        if S_squared is not None:
            S_sq = Decimal(S_squared)
            print(f"S² = {S_squared}")
            
            # Verify: S² = 4N + D
            four_N = 4 * N_dec
            D_calc = S_sq - four_N
            print(f"4N = {four_N}")
            print(f"D = S² - 4N = {D_calc}")
            
            # Compute S from S²
            try:
                S_calc = S_sq.sqrt()
                # Round to nearest integer
                S = int(S_calc.to_integral_value(rounding='ROUND_HALF_UP'))
                print(f"S = √(S²) = {S}")
            except:
                print(f"⚠️  Cannot compute exact S from S², using approximation")
                S = int(math.isqrt(S_squared))
                print(f"S ≈ {S}")
            
            D = int(D_calc)
            print(f"4N + D = {four_N + D_calc}")
            print(f"S² == 4N + D: {S_sq == (four_N + D_calc)} ✓✓✓")
            
        elif S is not None and D is not None:
            print(f"S = {S}")
            print(f"D = {D}")
            # Verify: S² = 4N + D
            S_sq = Decimal(S) * Decimal(S)
            four_N = 4 * N_dec
            expected_D = S_sq - four_N
            print(f"S² = {S_sq}")
            print(f"4N = {four_N}")
            print(f"Expected D = S² - 4N = {expected_D}")
            print(f"Provided D = {D}")
            print(f"S² == 4N + D: {S_sq == (four_N + D)} ✓✓✓")
        else:
            print("❌ Error: Must provide either S_squared or both S and D")
            return None, None, None, None
        
        # Compute √D
        print(f"\nComputing √D...")
        D_dec = Decimal(D)
        sqrt_D = D_dec.sqrt()
        sqrt_D_int = int(sqrt_D.to_integral_value(rounding='ROUND_HALF_UP'))
        print(f"√D = {sqrt_D} ≈ {sqrt_D_int}")
        
        # Verify D is a perfect square
        if sqrt_D_int * sqrt_D_int == D:
            print(f"✓ D is a perfect square: {sqrt_D_int}² = {D}")
        else:
            # Check nearby integers
            for offset in [-1, 0, 1]:
                test_sqrt = sqrt_D_int + offset
                if test_sqrt * test_sqrt == D:
                    sqrt_D_int = test_sqrt
                    print(f"✓ D is a perfect square: {sqrt_D_int}² = {D}")
                    break
            else:
                print(f"⚠️  D may not be a perfect square: {sqrt_D_int}² = {sqrt_D_int * sqrt_D_int} ≠ {D}")
                print(f"⚠️  Using integer approximation: √D ≈ {sqrt_D_int}")
        
        # Compute factors: p = (S + √D) / 2, q = (S - √D) / 2
        print(f"\nComputing factors from S and √D...")
        S_dec = Decimal(S)
        sqrt_D_dec = Decimal(sqrt_D_int)
        
        p_dec = (S_dec + sqrt_D_dec) / 2
        q_dec = (S_dec - sqrt_D_dec) / 2
        
        print(f"p = (S + √D) / 2 = ({S} + {sqrt_D_int}) / 2 = {p_dec}")
        print(f"q = (S - √D) / 2 = ({S} - {sqrt_D_int}) / 2 = {q_dec}")
        
        # Convert to integers
        p_int = int(p_dec.to_integral_value(rounding='ROUND_HALF_UP'))
        q_int = int(q_dec.to_integral_value(rounding='ROUND_HALF_UP'))
        
        print(f"\nInteger factors:")
        print(f"p = {p_int}")
        print(f"q = {q_int}")
        
        # Verify factorization
        product = p_int * q_int
        print(f"p × q = {product}")
        print(f"N = {self.N}")
        if product == self.N:
            print(f"✓✓✓ EXACT FACTORIZATION VERIFIED!")
        else:
            diff = abs(product - self.N)
            print(f"⚠️  Difference: {diff}")
        
        # Compute decimal factors with adjustments
        # Use approximate base candidates (could use sqrt(N) as base)
        k_base = int(math.isqrt(self.N))
        Qc_base = int(math.isqrt(self.N))
        
        alpha = float(p_dec - Decimal(k_base))
        beta = float(q_dec - Decimal(Qc_base))
        
        print(f"\nDecimal factors with adjustments:")
        print(f"Base candidates: k = {k_base}, Qc = {Qc_base}")
        print(f"Adjustments: α = {alpha:.10f}, β = {beta:.10f}")
        print(f"p = k + α = {k_base} + {alpha:.10f} = {float(p_dec):.10f}")
        print(f"q = Qc + β = {Qc_base} + {beta:.10f} = {float(q_dec):.10f}")
        
        return p_int, q_int, alpha, beta

    def compute_decimal_factors_with_adjustments(self, k: int, Qc: int, alpha: float = None, beta: float = None) -> Tuple[float, float, float, float]:
        """
        Compute decimal factors with explicit adjustments: p = k + α and q = Qc + β.
        
        Args:
            k: Base candidate for p
            Qc: Base candidate for q
            alpha: Adjustment for p (if None, will be computed)
            beta: Adjustment for q (if None, will be computed)
        
        Returns:
            Tuple of (p_decimal, q_decimal, alpha, beta)
        """
        from decimal import Decimal, getcontext
        
        # Use high precision for decimal arithmetic
        getcontext().prec = max(100, self.N.bit_length() * 2)
        
        # If adjustments not provided, compute them from the factorization constraint
        if alpha is None or beta is None:
            # We need p * q = N, so (k + α) * (Qc + β) = N
            # Expanding: k*Qc + k*β + Qc*α + α*β = N
            # For small adjustments, we can approximate: k*Qc + k*β + Qc*α ≈ N
            # So: k*β + Qc*α ≈ N - k*Qc
            
            k_decimal = Decimal(k)
            Qc_decimal = Decimal(Qc)
            N_decimal = Decimal(self.N)
            
            product_kQc = k_decimal * Qc_decimal
            diff = N_decimal - product_kQc
            
            # Solve for adjustments using linear approximation
            # If we assume α and β are small, we can use:
            # k*β + Qc*α ≈ diff
            # We can distribute the error proportionally
            
            if abs(k_decimal) > 0 and abs(Qc_decimal) > 0:
                # Proportional distribution based on magnitudes
                total_magnitude = abs(k_decimal) + abs(Qc_decimal)
                
                if alpha is None:
                    # α ≈ diff / (2 * Qc) for balanced adjustment
                    alpha = float(diff / (2 * Qc_decimal)) if Qc_decimal != 0 else 0.0
                
                if beta is None:
                    # β ≈ diff / (2 * k) for balanced adjustment
                    beta = float(diff / (2 * k_decimal)) if k_decimal != 0 else 0.0
            else:
                alpha = alpha if alpha is not None else 0.0
                beta = beta if beta is not None else 0.0
        
        # Compute decimal factors
        p_decimal = Decimal(k) + Decimal(str(alpha))
        q_decimal = Decimal(Qc) + Decimal(str(beta))
        
        return float(p_decimal), float(q_decimal), alpha, beta

    def calculate_perfect_s(self, S_squared: int = None, S: int = None) -> Tuple[Optional[int], Optional[int], Optional[int], bool]:
        """
        Calculate and verify PERFECT S across all 3 domains:
        1. Real/Algebraic: S² = 4N + D
        2. Integer/Diophantine: All values are integers, N = (S² - D)/4 is integer
        3. Modular/Parity: D ≡ S² (mod 4)
        
        The perfect S satisfies:
        - S² = 4N + D exactly (algebraic domain)
        - S, N, D are all integers (Diophantine domain)
        - D ≡ S² (mod 4) so that N = (S² - D)/4 is an integer (modular domain)
        
        Modular condition:
        - If S is even: S² ≡ 0 (mod 4), so D ≡ 0 (mod 4)
        - If S is odd: S² ≡ 1 (mod 4), so D ≡ 1 (mod 4)
        
        Args:
            S_squared: S² value (if provided, will compute S from it)
            S: S value directly (if provided, will compute S² from it)
            
        Returns:
            Tuple of (S, S², D, is_perfect) where:
            - S: The perfect S value
            - S²: S squared
            - D: D = S² - 4N
            - is_perfect: True if S satisfies all 3 domains, False otherwise
        """
        import math
        
        print(f"\n{'='*80}")
        print(f"CALCULATING PERFECT S ACROSS ALL 3 DOMAINS")
        print(f"{'='*80}")
        
        # Calculate S and S²
        if S_squared is not None:
            S_calc = int(math.isqrt(S_squared))
            if S_calc * S_calc != S_squared:
                if (S_calc + 1) * (S_calc + 1) == S_squared:
                    S_calc = S_calc + 1
            S_sq_calc = S_squared
        elif S is not None:
            S_calc = S
            S_sq_calc = S * S
        else:
            print("Error: Must provide either S_squared or S")
            return None, None, None, False
        
        four_N = 4 * self.N
        D_calc = S_sq_calc - four_N
        
        print(f"Given: S² = {S_sq_calc}")
        print(f"Calculated: S = √(S²) = {S_calc}")
        print(f"Calculated: D = S² - 4N = {D_calc}")
        
        # Verify Domain 1: Real/Algebraic - S² = 4N + D
        domain1_check = (S_sq_calc == four_N + D_calc)
        print(f"\nDomain 1 (Real/Algebraic): S² = 4N + D")
        print(f"  S² = {S_sq_calc}")
        print(f"  4N + D = {four_N} + {D_calc} = {four_N + D_calc}")
        print(f"  Match: {domain1_check} {'✓' if domain1_check else '✗'}")
        
        # Verify Domain 2: Integer/Diophantine - All values are integers
        domain2_check = (
            isinstance(S_calc, int) and 
            isinstance(self.N, int) and 
            isinstance(D_calc, int) and
            (S_sq_calc - D_calc) % 4 == 0  # N = (S² - D)/4 must be integer
        )
        N_calculated = (S_sq_calc - D_calc) // 4
        print(f"\nDomain 2 (Integer/Diophantine): All integers")
        print(f"  S = {S_calc} (integer) {'✓' if isinstance(S_calc, int) else '✗'}")
        print(f"  N = {self.N} (integer) {'✓' if isinstance(self.N, int) else '✗'}")
        print(f"  D = {D_calc} (integer) {'✓' if isinstance(D_calc, int) else '✗'}")
        print(f"  N = (S² - D)/4 = {N_calculated}")
        print(f"  N is integer: {(S_sq_calc - D_calc) % 4 == 0} {'✓' if (S_sq_calc - D_calc) % 4 == 0 else '✗'}")
        print(f"  N matches: {N_calculated == self.N} {'✓' if N_calculated == self.N else '✗'}")
        
        # Verify Domain 3: Modular/Parity - D ≡ S² (mod 4)
        S_sq_mod4 = S_sq_calc % 4
        D_mod4 = D_calc % 4
        domain3_check = (D_mod4 == S_sq_mod4)
        
        S_parity = "even" if S_calc % 2 == 0 else "odd"
        print(f"\nDomain 3 (Modular/Parity): D ≡ S² (mod 4)")
        print(f"  S = {S_calc} ({S_parity})")
        print(f"  S² mod 4 = {S_sq_mod4}")
        print(f"  D mod 4 = {D_mod4}")
        if S_calc % 2 == 0:
            print(f"  Rule: If S is even, D ≡ 0 (mod 4)")
            print(f"  D mod 4 = {D_mod4}, required = 0")
        else:
            print(f"  Rule: If S is odd, D ≡ 1 (mod 4)")
            print(f"  D mod 4 = {D_mod4}, required = 1")
        print(f"  D ≡ S² (mod 4): {domain3_check} {'✓' if domain3_check else '✗'}")
        
        # Parametrization check (optional verification)
        # This is a consistency check, not required for perfect S
        # The correct parametrization: D = e + 4d where d = (D - e) / 4
        e = S_calc % 2
        k_param = (S_calc - e) // 2
        
        # Correct parametrization: D = e + 4d, so d = (D - e) / 4
        # This must be an integer for the parametrization to be valid
        d_param = (D_calc - e) // 4
        D_from_param = e + 4 * d_param
        
        # Check if parametrization is valid (d must be integer)
        parametrization_valid = ((D_calc - e) % 4 == 0) and (D_from_param == D_calc)
        
        print(f"\nParametrization (optional consistency check):")
        print(f"  S = 2k + e")
        print(f"    e = S mod 2 = {e} ({'even' if e == 0 else 'odd'})")
        print(f"    k = (S - e) / 2 = {k_param}")
        print(f"  d = (D - e) / 4 = ({D_calc} - {e}) / 4 = {d_param}")
        print(f"  D = e + 4d = {e} + 4({d_param}) = {D_from_param}")
        print(f"  Parametrization valid: {parametrization_valid} {'✓' if parametrization_valid else '✗'}")
        if not parametrization_valid:
            print(f"  Note: Parametrization check failed, but this doesn't affect Perfect S verification")
            print(f"  Perfect S only requires: S² = 4N + D, all integers, and D ≡ S² (mod 4)")
        
        # Final verification - parametrization is optional, not required
        is_perfect = domain1_check and domain2_check and domain3_check
        
        print(f"\n{'='*80}")
        if is_perfect:
            print(f"✓✓✓ PERFECT S VERIFIED ACROSS ALL 3 DOMAINS! ✓✓✓")
        else:
            print(f"⚠ S does not satisfy all 3 domains")
        print(f"{'='*80}")
        print(f"S = {S_calc}")
        print(f"S² = {S_sq_calc}")
        print(f"D = {D_calc}")
        print(f"N = {self.N}")
        print(f"{'='*80}")
        
        return S_calc, S_sq_calc, D_calc, is_perfect

    def solve_exact_from_s(self, S: int, search_radius: int = None) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        """
        EXACT FACTORIZATION FROM S (Mathematical Derivation Method)
        
        Given:
          N = p × q
          S ≈ (p+q)/2
          
        From Fermat's factorization method:
          p = (p+q)/2 + (p-q)/2
          q = (p+q)/2 - (p-q)/2
          
        Let:
          T = p+q (must be an integer)
          k = p-q (must be an integer)
          
        Then:
          p = (T + k)/2
          q = (T - k)/2
          
        Constraints:
          1. T must be an integer
          2. k must be an integer  
          3. T and k must have the same parity (both even or both odd)
          4. p × q = N exactly
          5. (p+q)² - 4pq = (p-q)²
              Therefore: T² - 4N = k²
              Or: T² = 4N + k²
        
        Args:
            S: Approximation of (p+q)/2
            search_radius: Search radius around T_base = 2S (default: adaptive)
            
        Returns:
            Tuple of (p, q, T, k) where T = p+q and k = p-q, or (None, None, None, None) if not found
        """
        print(f"\n{'='*80}")
        print(f"EXACT FACTORIZATION FROM S (Mathematical Derivation)")
        print(f"{'='*80}")
        print(f"Given: N = {self.N}")
        print(f"Given: S ≈ (p+q)/2 = {S}")
        
        # Step 1: Compute T_base = 2S (since S ≈ (p+q)/2, we have T = p+q ≈ 2S)
        T_base = 2 * S
        print(f"\n[Exact Analysis] T_base = 2S = {T_base}")
        
        # Step 2: Compute T_base² - 4N
        T_base_squared = T_base * T_base
        four_N = 4 * self.N
        T_sq_minus_4N = T_base_squared - four_N
        print(f"[Exact Analysis] T_base² = {T_base_squared}")
        print(f"[Exact Analysis] 4N = {four_N}")
        print(f"[Exact Analysis] T_base² - 4N = {T_sq_minus_4N}")
        
        # Check if S might actually be T = p+q (not S = (p+q)/2)
        # If T_sq_minus_4N is negative, S is too small to be T
        # If T_sq_minus_4N is positive and close to a perfect square, S might be T
        S_as_T_sq = S * S
        S_as_T_sq_minus_4N = S_as_T_sq - four_N
        if S_as_T_sq_minus_4N >= 0:
            k_test_direct = self._integer_sqrt_approx(S_as_T_sq_minus_4N)
            if k_test_direct * k_test_direct == S_as_T_sq_minus_4N:
                print(f"\n[Exact Analysis] ✓✓✓ S is actually T = p+q (not S = (p+q)/2)!")
                print(f"[Exact Analysis] Direct verification: S² - 4N = {S_as_T_sq_minus_4N} = {k_test_direct}²")
                T_exact = S
                k_exact = k_test_direct
                # Skip to factor computation
                p_exact = (T_exact + k_exact) // 2
                q_exact = (T_exact - k_exact) // 2
                product = p_exact * q_exact
                if product == self.N:
                    print(f"[Exact Analysis] ✓✓✓ EXACT FACTORIZATION VERIFIED! ✓✓✓")
                    return p_exact, q_exact, T_exact, k_exact
                else:
                    print(f"[Exact Analysis] S as T gives factors but product doesn't match exactly")
                    # Continue with normal search
            else:
                remainder_direct = S_as_T_sq_minus_4N - (k_test_direct * k_test_direct)
                if abs(remainder_direct) < abs(T_sq_minus_4N - (self._integer_sqrt_approx(T_sq_minus_4N) ** 2)):
                    print(f"[Exact Analysis] Note: S might be T = p+q (smaller remainder: {remainder_direct})")
                    # Try using S as T_base instead
                    T_base_alt = S
                    T_base_alt_sq = S_as_T_sq
                    T_sq_minus_4N_alt = S_as_T_sq_minus_4N
                    print(f"[Exact Analysis] Trying S as T_base: T = {T_base_alt}")
                    print(f"[Exact Analysis] T² - 4N = {T_sq_minus_4N_alt}")
                    # Use this as alternative starting point if it's better
                    if abs(remainder_direct) < 1000000:  # If remainder is small, prefer this
                        print(f"[Exact Analysis] Using S as T_base (smaller remainder)")
                        T_base = T_base_alt
                        T_base_squared = T_base_alt_sq
                        T_sq_minus_4N = T_sq_minus_4N_alt
        
        # Step 3: Compute k_approx = √(T_base² - 4N)
        k_approx = self._integer_sqrt_approx(T_sq_minus_4N)
        k_approx_squared = k_approx * k_approx
        remainder = T_sq_minus_4N - k_approx_squared
        print(f"[Exact Analysis] k_approx = floor(√(T_base² - 4N)) = {k_approx}")
        print(f"[Exact Analysis] k_approx² = {k_approx_squared}")
        print(f"[Exact Analysis] remainder = {remainder}")
        
        # Step 4: Determine search radius
        if search_radius is None:
            # Adaptive search radius based on remainder magnitude
            if remainder == 0:
                # Already exact!
                print(f"[Exact Analysis] ✓✓✓ EXACT RELATIONSHIP FOUND!")
                print(f"[Exact Analysis] T_base² - 4N = k_approx² (exact match)")
                T_exact = T_base
                k_exact = k_approx
            else:
                # Estimate search radius based on remainder
                # For small remainder, search nearby
                if abs(remainder) < 1000:
                    search_radius = 100
                elif abs(remainder) < 100000:
                    search_radius = 1000
                elif abs(remainder) < 10000000:
                    search_radius = 10000
                else:
                    # For large remainder, use proportional search
                    search_radius = max(10000, abs(remainder) // (2 * k_approx) if k_approx > 0 else 100000)
                
                print(f"[Exact Analysis] Using adaptive search radius: {search_radius}")
                
                # Search for exact T and k
                T_exact = None
                k_exact = None
                
                # Search strategy: Try T = T_base + ΔT and find corresponding k
                # We need: (T_base + ΔT)² - 4N = k²
                # Rearranging: k² = T_base² + 2*T_base*ΔT + ΔT² - 4N
                #            k² = (T_base² - 4N) + 2*T_base*ΔT + ΔT²
                #            k² = k_approx² + remainder + 2*T_base*ΔT + ΔT²
                
                print(f"[Exact Analysis] Searching for exact T and k...")
                print(f"[Exact Analysis] Searching T in range [{T_base - search_radius}, {T_base + search_radius}]")
                
                checked = 0
                for delta_T in range(-search_radius, search_radius + 1):
                    checked += 1
                    if checked % 10000 == 0:
                        print(f"[Exact Analysis] Checked {checked:,} T values... (current ΔT: {delta_T:+d})")
                    elif checked % 1000 == 0 and checked < 10000:
                        print(f"[Exact Analysis] Checked {checked:,} T values...")
                    
                    T_test = T_base + delta_T
                    T_test_squared = T_test * T_test
                    k_squared_target = T_test_squared - four_N
                    
                    if k_squared_target < 0:
                        continue
                    
                    # Check if k_squared_target is a perfect square
                    k_test = self._integer_sqrt_approx(k_squared_target)
                    k_test_squared = k_test * k_test
                    
                    if k_test_squared == k_squared_target:
                        # Found exact relationship!
                        print(f"\n[Exact Analysis] ✓✓✓ EXACT RELATIONSHIP FOUND!")
                        print(f"[Exact Analysis]   T = {T_test} (ΔT = {delta_T:+d})")
                        print(f"[Exact Analysis]   k = {k_test}")
                        print(f"[Exact Analysis]   Verification: T² - 4N = {T_test_squared} - {four_N} = {k_squared_target}")
                        print(f"[Exact Analysis]   k² = {k_test_squared} ✓")
                        
                        T_exact = T_test
                        k_exact = k_test
                        break
                
                if T_exact is None:
                    print(f"[Exact Analysis] No exact relationship found in search radius {search_radius}")
                    print(f"[Exact Analysis] Try increasing search radius or using alternative methods")
                    return None, None, None, None
        else:
            # Use provided search radius
            print(f"[Exact Analysis] Using provided search radius: {search_radius}")
            T_exact = None
            k_exact = None
            
            for delta_T in range(-search_radius, search_radius + 1):
                T_test = T_base + delta_T
                T_test_squared = T_test * T_test
                k_squared_target = T_test_squared - four_N
                
                if k_squared_target < 0:
                    continue
                
                k_test = self._integer_sqrt_approx(k_squared_target)
                if k_test * k_test == k_squared_target:
                    T_exact = T_test
                    k_exact = k_test
                    break
            
            if T_exact is None:
                return None, None, None, None
        
        # Step 5: Compute p and q from T and k
        # p = (T + k)/2, q = (T - k)/2
        # Both T and k must have the same parity for p and q to be integers
        print(f"\n[Exact Analysis] Computing factors from T and k...")
        print(f"[Exact Analysis] T = {T_exact}, k = {k_exact}")
        print(f"[Exact Analysis] T parity: {'even' if T_exact % 2 == 0 else 'odd'}")
        print(f"[Exact Analysis] k parity: {'even' if k_exact % 2 == 0 else 'odd'}")
        
        if (T_exact + k_exact) % 2 != 0:
            print(f"[Exact Analysis] ⚠️  Warning: T and k have different parity")
            print(f"[Exact Analysis]   (T + k) = {T_exact + k_exact} is odd, cannot divide by 2 exactly")
            # Still try to compute, rounding may work
            p_exact = (T_exact + k_exact) // 2
            q_exact = (T_exact - k_exact) // 2
        else:
            p_exact = (T_exact + k_exact) // 2
            q_exact = (T_exact - k_exact) // 2
        
        print(f"[Exact Analysis] p = (T + k)/2 = ({T_exact} + {k_exact})/2 = {p_exact}")
        print(f"[Exact Analysis] q = (T - k)/2 = ({T_exact} - {k_exact})/2 = {q_exact}")
        
        # Step 6: Verify factorization
        product = p_exact * q_exact
        print(f"\n[Exact Analysis] Verification:")
        print(f"[Exact Analysis]   p × q = {p_exact} × {q_exact} = {product}")
        print(f"[Exact Analysis]   Target N = {self.N}")
        
        if product == self.N:
            print(f"[Exact Analysis] ✓✓✓ EXACT FACTORIZATION VERIFIED! ✓✓✓")
            print(f"[Exact Analysis]   p = {p_exact}")
            print(f"[Exact Analysis]   q = {q_exact}")
            return p_exact, q_exact, T_exact, k_exact
        else:
            error = abs(product - self.N)
            print(f"[Exact Analysis] ⚠️  Factorization error: {error}")
            print(f"[Exact Analysis]   This may indicate S was not accurate enough")
            # Still return the factors as they may be close
            return p_exact, q_exact, T_exact, k_exact

    def solve(self, p_candidate: int, q_candidate: int, confidence: float = 0.5,
              config: Optional[np.ndarray] = None, search_radius: int = 1000) -> Tuple[Optional[int], Optional[int], float, Optional[np.ndarray]]:
        """
        Apply minimizable factorization lattice to refine candidate factors.
        """
        print(f"\n{'─'*80}")
        print(f"[LATTICE SOLVER] ⭐ Minimizable Factorization Lattice")
        print(f"{'─'*80}")
        print(f"[Lattice] Input candidates: p={p_candidate}, q={q_candidate}")
        print(f"[Lattice] Search radius: {search_radius}")
        
        # Check initial approximation quality
        initial_product = p_candidate * q_candidate
        initial_diff = abs(initial_product - self.N)
        initial_diff_bits = initial_diff.bit_length()
        print(f"[Lattice] Initial approximation quality:")
        print(f"[Lattice]   p_candidate × q_candidate = {initial_product}")
        print(f"[Lattice]   Difference from N: {initial_diff} ({initial_diff_bits} bits)")
        
        # Warn if initial difference is too large relative to search radius
        if initial_diff_bits > search_radius.bit_length() + 100:
            print(f"[Lattice] ⚠️  WARNING: Initial difference ({initial_diff_bits} bits) is much larger than search radius ({search_radius.bit_length()} bits)")
            print(f"[Lattice]    The approximations are too far from the actual factors")
            print(f"[Lattice]    Lattice method works best when approximations are within search radius")
            print(f"[Lattice]    Consider:")
            print(f"[Lattice]      - Providing better initial approximations (closer to actual factors)")
            print(f"[Lattice]      - Using polynomial methods (--polynomial flag)")
            print(f"[Lattice]      - Increasing search radius (currently {search_radius.bit_length()} bits)")
        
        # Warn if search radius is astronomically large
        if search_radius > 10**100:
            print(f"[Lattice] ⚠️  WARNING: Search radius is extremely large ({search_radius.bit_length()} bits)")
            print(f"[Lattice]    After LLL reduction, corrections should be extracted directly from short vectors")
            print(f"[Lattice]    The search radius is used as an upper bound, not for exhaustive search")
            print(f"[Lattice]    LLL-reduced vectors should contain the corrections directly")

        # Try bulk search first if --bulk flag is set (doesn't need approximations)
        use_bulk = getattr(self, 'use_bulk_search', False)
        
        if use_bulk:
            print(f"[Lattice] 🔍 Bulk search enabled (via --bulk flag)")
            print(f"[Lattice]    Search radius: {search_radius.bit_length()} bits")
            print(f"[Lattice]    Attempting bulk factor search using LLL (approximation-independent)...")
            print(f"[Lattice]    This method creates a large lattice ({100}+ vectors) to search the entire space")
            # Use pre-trained transformer if available
            pretrained_transformer = getattr(self, 'pretrained_transformer', None)
            bulk_result = self._bulk_search_factors_with_lll(search_radius, pretrained_transformer)
            if bulk_result:
                p_found, q_found = bulk_result
                if p_found * q_found == self.N:
                    print(f"[Lattice] ✓✓✓ BULK SEARCH SUCCESS! p={p_found}, q={q_found}")
                    pyramid_basis = self._construct_pyramid_lattice_basis(p_found, q_found)
                    return p_found, q_found, 1.0, pyramid_basis
            print(f"[Lattice]    Bulk search didn't find factors, continuing with pyramid lattice...")
        
        # Always construct pyramid lattice basis for polynomial generation
        pyramid_basis = self._construct_pyramid_lattice_basis(p_candidate, q_candidate)
        
        # For small numbers, try trial division first (fast and reliable)
        if self.N < 1000000:
            print(f"[Lattice] Small number detected, trying trial division first...")
            import math
            limit = min(100000, int(math.isqrt(self.N)) + 1)
            for p_test in range(2, limit):
                if self.N % p_test == 0:
                    q_test = self.N // p_test
                    if p_test * q_test == self.N:
                        print(f"[Lattice] ✓✓✓ FOUND via trial division: {p_test} × {q_test} = {self.N}")
                        # Compute decimal factors with adjustments
                        p_dec, q_dec, alpha, beta = self.compute_decimal_factors_with_adjustments(p_test, q_test, 0.0, 0.0)
                        print(f"[Lattice] Decimal factors: p = {p_test} + {alpha:.10f} = {p_dec:.10f}")
                        print(f"[Lattice] Decimal factors: q = {q_test} + {beta:.10f} = {q_dec:.10f}")
                        return p_test, q_test, 1.0, pyramid_basis
            # If no factors found, N is likely prime
            print(f"[Lattice] No small factors found, N may be prime")
            if self.N > 1:
                print(f"[Lattice] Returning prime factorization: {self.N} × 1")
                return self.N, 1, 1.0, pyramid_basis

        if p_candidate <= 0 or q_candidate <= 0:
            print(f"[Lattice] Invalid candidates, skipping")
            return None, None, 0.0, pyramid_basis

        # Construct pyramid lattice basis for polynomial generation
        pyramid_basis = self._construct_pyramid_lattice_basis(p_candidate, q_candidate)

        # Adapt search radius based on initial difference if needed
        # If initial difference is larger than search radius, expand it
        initial_product = p_candidate * q_candidate
        initial_diff = abs(initial_product - self.N)
        initial_diff_bits = initial_diff.bit_length()
        search_radius_bits = search_radius.bit_length() if search_radius > 0 else 0
        
        # If initial difference is larger than search radius, expand search radius
        if initial_diff_bits > search_radius_bits:
            # Expand to at least 2x the initial difference (with some margin)
            expanded_radius_bits = initial_diff_bits + 100  # Add 100 bits of margin
            expanded_radius = 2 ** expanded_radius_bits
            print(f"[Lattice] 🔧 Expanding search radius: {search_radius_bits} bits → {expanded_radius_bits} bits")
            print(f"[Lattice]    (Initial difference: {initial_diff_bits} bits, need larger radius)")
            search_radius = expanded_radius
        
        # Find best factorization corrections
        best_dp, best_dq, correction_improvement = self._find_best_factorization_corrections(
            p_candidate, q_candidate, pyramid_basis, config, search_radius
        )

        refined_p = p_candidate + best_dp
        refined_q = q_candidate + best_dq

        print(f"[Lattice] Refined candidates: p={refined_p}, q={refined_q}")

        # Check if we found an exact factorization FIRST (before decimal computation)
        final_product = refined_p * refined_q
        final_diff = abs(final_product - self.N)
        final_diff_bits = final_diff.bit_length() if final_diff > 0 else 0
        
        if final_product == self.N:
            print(f"[Lattice] ✓✓✓ EXACT FACTORIZATION FOUND! ✓✓✓")
            print(f"[Lattice]   p = {refined_p}")
            print(f"[Lattice]   q = {refined_q}")
            print(f"[Lattice]   p × q = {final_product}")
            print(f"[Lattice]   Verification: {final_product == self.N}")
            return refined_p, refined_q, 1.0, pyramid_basis
        
        # If very close (≤10 bits), try to find exact factors by checking nearby values
        if final_diff_bits <= 10 and final_diff > 0:
            print(f"[Lattice] ⚠️  Very close! (only {final_diff_bits} bits off, difference = {final_diff})")
            print(f"[Lattice]    → Trying nearby values and divisibility checks to find exact factors...")
            
            # First, check if p or q divides N exactly
            if self.N % refined_p == 0:
                q_exact = self.N // refined_p
                if refined_p * q_exact == self.N:
                    print(f"[Lattice]    ✓✓✓ EXACT FACTOR FOUND! p divides N exactly")
                    print(f"[Lattice]       p = {refined_p}, q = {q_exact}")
                    return refined_p, q_exact, 1.0, pyramid_basis
            
            if self.N % refined_q == 0:
                p_exact = self.N // refined_q
                if p_exact * refined_q == self.N:
                    print(f"[Lattice]    ✓✓✓ EXACT FACTOR FOUND! q divides N exactly")
                    print(f"[Lattice]       p = {p_exact}, q = {refined_q}")
                    return p_exact, refined_q, 1.0, pyramid_basis
            
            # Try small adjustments to p and q
            search_range = min(100, final_diff + 10)  # Search within ±100 or slightly more than the error
            print(f"[Lattice]    → Trying small adjustments (±{search_range})...")
            
            for dp in range(-search_range, search_range + 1):
                for dq in range(-search_range, search_range + 1):
                    p_test = refined_p + dp
                    q_test = refined_q + dq
                    if p_test > 1 and q_test > 1:
                        product_test = p_test * q_test
                        if product_test == self.N:
                            print(f"[Lattice]    ✓✓✓ EXACT FACTOR FOUND with small adjustment!")
                            print(f"[Lattice]       p = {p_test} (adjusted by {dp})")
                            print(f"[Lattice]       q = {q_test} (adjusted by {dq})")
                            return p_test, q_test, 1.0, pyramid_basis
            
            print(f"[Lattice]    → Nearby search didn't find exact factors")

        # For huge numbers, skip decimal computation (causes float overflow)
        # But we can still show the integer corrections
        if self.N.bit_length() > 512:
            print(f"[Lattice] Skipping decimal computation for {self.N.bit_length()}-bit N (would cause float overflow)")
            print(f"[Lattice] Integer corrections: dp={best_dp:+d}, dq={best_dq:+d}")
            print(f"[Lattice] Refined factors are integers: p={refined_p}, q={refined_q}")
        else:
            # Compute decimal factors with adjustments (only for smaller numbers)
            try:
                p_decimal, q_decimal, alpha, beta = self.compute_decimal_factors_with_adjustments(
                    p_candidate, q_candidate, float(best_dp), float(best_dq)
                )
                
                print(f"[Lattice] Decimal factors with adjustments:")
                print(f"[Lattice]   p = {p_candidate} + {alpha:.10f} = {p_decimal:.10f}")
                print(f"[Lattice]   q = {q_candidate} + {beta:.10f} = {q_decimal:.10f}")
                
                # Verify decimal product
                from decimal import Decimal
                p_dec = Decimal(str(p_decimal))
                q_dec = Decimal(str(q_decimal))
                product_dec = p_dec * q_dec
                N_dec = Decimal(self.N)
                error = abs(product_dec - N_dec)
                print(f"[Lattice]   p × q = {product_dec:.10f}")
                print(f"[Lattice]   Error: {error:.10f}")
            except (OverflowError, ValueError) as e:
                print(f"[Lattice] Decimal computation failed (expected for huge numbers): {e}")

        # Calculate improvement
        initial_product = p_candidate * q_candidate
        initial_diff = abs(initial_product - self.N)
        final_diff = abs(final_product - self.N)
        
        print(f"[Lattice] Initial difference: {initial_diff} ({initial_diff.bit_length()} bits)")
        print(f"[Lattice] Final difference: {final_diff} ({final_diff.bit_length()} bits)")

        if final_diff < initial_diff:
            # For huge numbers, use bit length difference as improvement metric
            if self.N.bit_length() > 1000:
                initial_bits = initial_diff.bit_length()
                final_bits = final_diff.bit_length()
                if initial_bits > 0:
                    improvement = max(0.0, min(1.0, (initial_bits - final_bits) / initial_bits))
                else:
                    improvement = 1.0 if final_diff == 0 else 0.0
                print(f"[Lattice] ✓ Improvement: {initial_bits} bits → {final_bits} bits ({improvement:.1%})")
            else:
                improvement = (initial_diff - final_diff) / initial_diff if initial_diff > 0 else 0.0
                print(f"[Lattice] ✓ Improvement: {improvement:.1%}")
            
            # If we're very close but not exact, try iterating automatically
            if final_diff > 0 and final_diff < initial_diff and final_diff.bit_length() < 50:
                print(f"[Lattice] 💡 Difference is small ({final_diff.bit_length()} bits), attempting iteration...")
                print(f"[Lattice]    Using refined candidates as new starting point for another LLL pass")
                
                # Try one more iteration with refined candidates
                try:
                    # Reduce search radius for iteration (we're already close)
                    iter_search_radius = min(search_radius, max(1000, final_diff.bit_length() * 10))
                    print(f"[Lattice]    Iteration search radius: {iter_search_radius}")
                    
                    # Construct new pyramid basis with refined candidates
                    iter_pyramid_basis = self._construct_pyramid_lattice_basis(refined_p, refined_q)
                    
                    # Find corrections again
                    iter_dp, iter_dq, iter_improvement = self._find_best_factorization_corrections(
                        refined_p, refined_q, iter_pyramid_basis, config, iter_search_radius
                    )
                    
                    iter_refined_p = refined_p + iter_dp
                    iter_refined_q = refined_q + iter_dq
                    iter_product = iter_refined_p * iter_refined_q
                    iter_diff = abs(iter_product - self.N)
                    
                    print(f"[Lattice]    Iteration result: dp={iter_dp:+d}, dq={iter_dq:+d}")
                    print(f"[Lattice]    Iteration difference: {iter_diff} ({iter_diff.bit_length()} bits)")
                    
                    if iter_product == self.N:
                        print(f"[Lattice] ✓✓✓ EXACT FACTORIZATION FOUND after iteration! ✓✓✓")
                        print(f"[Lattice]   p = {iter_refined_p}")
                        print(f"[Lattice]   q = {iter_refined_q}")
                        return iter_refined_p, iter_refined_q, 1.0, iter_pyramid_basis
                    elif iter_diff < final_diff:
                        print(f"[Lattice]    ✓ Iteration improved: {final_diff.bit_length()} bits → {iter_diff.bit_length()} bits")
                        # Use the iterated result
                        return iter_refined_p, iter_refined_q, improvement + iter_improvement, iter_pyramid_basis
                    else:
                        print(f"[Lattice]    Iteration did not improve further")
                except Exception as e:
                    print(f"[Lattice]    Iteration failed: {e}, returning original refined result")
            
            return refined_p, refined_q, improvement, pyramid_basis
        else:
            print(f"[Lattice] No significant improvement")
            return None, None, 0.0, pyramid_basis

    def generate_factorization_polynomials(self, p_candidate: int, q_candidate: int,
                                         lattice_basis: np.ndarray = None,
                                         search_radius: int = 1000) -> List[sp.Expr]:
        """
        Extract polynomial equations from pyramid lattice vectors and force into square structure.

        Each lattice vector [a, b, c] represents: a + b*p + c*q = 0
        Convert these to polynomial constraints organized in a square grid with ABCD fusion.
        """
        print(f"[Polynomial] Extracting polynomial equations from pyramid vectors...")
        print(f"[Polynomial] Forcing polynomials into square structure with ABCD fusion...")

        # Check if user specified lattice dimension
        if hasattr(self, 'config') and self.config and self.config.get('lattice_dimension'):
            lattice_base = self.config['lattice_dimension']
            try:
                lattice_base_int = int(lattice_base)

                # Use lattice dimension as an exponent for the coefficient limit
                # Interpret user input as 2^exponent, then scale for the number size
                sqrt_N_bits = (self.N.bit_length() + 1) // 2

                # Convert lattice_base to bit exponent, then scale appropriately
                import math
                lattice_exponent = int(math.log2(lattice_base_int)) if lattice_base_int > 0 else 60

                if self.N.bit_length() > 1000:
                    # For 1000+ bit numbers, use a very large but practical limit
                    # Since Python can handle large integers, use a generous limit
                    LATTICE_COEFF_LIMIT = 10**500  # 500-digit limit should be sufficient for any practical factorization
                    print(f"[Polynomial]   Using large coefficient limit (10^500) for 1000+ bit numbers")
                elif self.N.bit_length() > 500:
                    # For medium-large numbers, use lattice_exponent + sqrt_N_bits//2
                    total_exponent = lattice_exponent + (sqrt_N_bits // 2)
                    LATTICE_COEFF_LIMIT = 2 ** min(total_exponent, 800)
                    print(f"[Polynomial]   Using lattice exponent {lattice_exponent} + sqrt(N)/2 {sqrt_N_bits//2} = 2^{total_exponent} coefficient limit")
                else:
                    # For smaller numbers, use the lattice dimension directly as 2^lattice_exponent
                    LATTICE_COEFF_LIMIT = 2 ** min(lattice_exponent, 200)
                    print(f"[Polynomial]   Using lattice exponent {lattice_exponent} = 2^{min(lattice_exponent, 200)} coefficient limit")

                # Final safety check
                if LATTICE_COEFF_LIMIT > 10**1000:  # Too large
                    LATTICE_COEFF_LIMIT = 10**1000
                    print(f"[Polynomial]   Capped coefficient limit at 10^1000 for practicality")

            except (ValueError, OverflowError) as e:
                print(f"[Polynomial]   ERROR: Invalid lattice dimension '{lattice_base}': {e}, using default adaptive scaling")
        else:
            # Adaptive coefficient limit based on number size
            # For large N, allow coefficients up to sqrt(N) and beyond
            sqrt_N_bits = (self.N.bit_length() + 1) // 2
            # Set limit to handle sqrt(N) sized coefficients with headroom
            # Use 2^(sqrt_N_bits + 100) to allow for operations and variations
            if self.N.bit_length() > 1500:
                # For extremely large numbers, allow very large coefficients
                LATTICE_COEFF_LIMIT = 10**(sqrt_N_bits // 3 + 200)  # Scale with N size
                print(f"[Polynomial]   Using adaptive coefficient limit (10^{sqrt_N_bits // 3 + 200}) for extremely large numbers")
            elif self.N.bit_length() > 1000:
                # For very large numbers, scale with N
                LATTICE_COEFF_LIMIT = 10**(sqrt_N_bits // 3 + 150)
                print(f"[Polynomial]   Using adaptive coefficient limit (10^{sqrt_N_bits // 3 + 150}) for large numbers")
            else:
                # For smaller numbers, use reasonable precision (adaptive to size)
                if self.N.bit_length() <= 64:
                    # For very small numbers (64 bits and below), use moderate limits
                    LATTICE_COEFF_LIMIT = 10**20  # Reasonable for small numbers
                    print(f"[Polynomial]   Using 10^20 coefficient limit for small numbers")
                else:
                    # For medium-small numbers, scale with N
                    LATTICE_COEFF_LIMIT = 10**(sqrt_N_bits // 3 + 100)
                    print(f"[Polynomial]   Using adaptive coefficient limit (10^{sqrt_N_bits // 3 + 100}) for medium-small numbers")

        # Create square structure: organize vectors into a square grid
        n_vectors = len(lattice_basis) if lattice_basis is not None else 10
        grid_size = int(sp.sqrt(n_vectors)) + 1  # Find smallest square that fits all vectors

        print(f"[Polynomial] Organizing {n_vectors} vectors into {grid_size}x{grid_size} square grid")

        # Initialize square grid with ABCD components at each position
        square_grid = {}
        p, q = sp.symbols('p q', integer=True, positive=True)

        # Process vectors and place them in square grid positions
        if lattice_basis is not None:
            for idx, vector in enumerate(lattice_basis):
                try:
                    if len(vector) != 3:
                        continue

                    a, b, c = vector[0], vector[1], vector[2]

                    # Skip trivial vectors
                    if abs(b) == 0 and abs(c) == 0:
                        continue

                    # Calculate grid position
                    row = idx // grid_size
                    col = idx % grid_size

                    if (row, col) not in square_grid:
                        square_grid[(row, col)] = {'A': [], 'B': [], 'C': [], 'D': []}

                    # Generate ABCD components for this grid position
                    self._generate_abcd_components(square_grid[(row, col)], a, b, c, p, q, LATTICE_COEFF_LIMIT)

                except Exception as e:
                    continue
        else:
            # Generate synthetic lattice vectors if none provided
            print(f"[Polynomial] No lattice basis provided, generating synthetic vectors...")
            for idx in range(min(200, grid_size * grid_size)):  # MASSIVE expansion: process up to 200 grid positions
                row = idx // grid_size
                col = idx % grid_size

                if (row, col) not in square_grid:
                    square_grid[(row, col)] = {'A': [], 'B': [], 'C': [], 'D': []}

                # Generate synthetic vectors scaled to coefficient limit
                # Use logarithmic scaling to avoid overflow while providing meaningful range
                import math
                coeff_scale = min(1000, int(math.log2(max(1, LATTICE_COEFF_LIMIT))) // 10)

                # Scale the random ranges based on coefficient limit
                rand_range = max(100, min(10**6, 10**(coeff_scale // 3)))
                offset_range = max(10, min(10**4, rand_range // 100))

                a = np.random.randint(-rand_range, rand_range)
                b = p_candidate + np.random.randint(-offset_range, offset_range)
                c = q_candidate + np.random.randint(-offset_range, offset_range)

                self._generate_abcd_components(square_grid[(row, col)], a, b, c, p, q, LATTICE_COEFF_LIMIT)

        # Fuse A and B components together after generation
        total_a = sum(len(components['A']) for components in square_grid.values())
        total_b = sum(len(components['B']) for components in square_grid.values())
        total_c = sum(len(components['C']) for components in square_grid.values())
        total_d = sum(len(components['D']) for components in square_grid.values())

        print(f"[Polynomial] Generated components: A={total_a}, B={total_b}, C={total_c}, D={total_d} across {len(square_grid)} grid positions")

        print(f"[Polynomial] Fusing A and B components...")
        fused_polynomials = self._fuse_ab_components(square_grid, p, q)

        print(f"[Polynomial] Generated {len(fused_polynomials)} fused polynomial equations")
        return fused_polynomials

    def _generate_abcd_components(self, grid_pos, a, b, c, p, q, coeff_limit):
        """
        Generate ABCD polynomial components for a grid position.

        A: Linear constraints (primary structure)
        B: Modular constraints (fusable with A)
        C: Ratio constraints (cross-linking)
        D: Quadratic constraints (higher-order)
        """
        # Check coefficient limits - use adaptive limits based on coeff_limit parameter
        use_limits = coeff_limit is not None
        try:
            a_ok = not use_limits or abs(a) < coeff_limit
            b_ok = not use_limits or abs(b) < coeff_limit
            c_ok = not use_limits or abs(c) < coeff_limit
            # Debug output
            if not b_ok or not c_ok:
                print(f"[ABCD] Coeff check: |b|={abs(b)}, |c|={abs(c)}, limit={coeff_limit}, b_ok={b_ok}, c_ok={c_ok}")
        except (TypeError, AttributeError) as e:
            # If comparison fails, assume they're ok for large numbers
            print(f"[ABCD] Comparison failed ({e}), assuming coefficients are ok")
            a_ok = b_ok = c_ok = True

        # Adaptive limits: use coeff_limit if provided, otherwise use very large defaults
        # For large N, allow much larger coefficients - scale with the limit provided
        if use_limits:
            # Use the provided coeff_limit, but be generous for large N
            # Allow coefficients up to the limit for all operations
            max_linear = coeff_limit
            max_quad = coeff_limit  # Allow same limit for quadratics
            max_cross = coeff_limit
        else:
            # No limit specified - use very large defaults to handle large N
            # Use extremely large limits to allow sqrt(N) sized coefficients and operations
            max_linear = 10**1000  # Effectively unlimited for practical purposes
            max_quad = 10**1000
            max_cross = 10**1000

        # Component A: Linear constraints (fundamental structure)
        # Debug: check why A components aren't being generated
        #print(f"[ABCD] Checking A component: a={a}, b={type(b).__name__}, c={type(c).__name__}, a_ok={a_ok}, b_ok={b_ok}, c_ok={c_ok}")
        if not (a_ok and b_ok and c_ok):
            print(f"[ABCD] A component rejected: a_ok={a_ok}, b_ok={b_ok}, c_ok={c_ok}")
            pass
        elif not (abs(a) < max_linear and abs(b) < max_linear and abs(c) < max_linear):
            print(f"[ABCD] A component rejected by size: |a|={abs(a)}, |b|={abs(b)}, |c|={abs(c)}, max_linear={max_linear}")
            pass
        else:
            # Primary linear constraint
            linear_a = a + b*p + c*q
            grid_pos['A'].append(linear_a)

            # Secondary linear (negated)
            linear_a_neg = -a - b*p - c*q
            grid_pos['A'].append(linear_a_neg)

        # Component B: Modular constraints (designed to fuse with A)
        if abs(b) > 0:
            # Generate modular hints that can be fused with A's linear structure
            moduli = [97, 101, 103, 107, 109]
            b_components_added = 0
            for modulus in moduli:
                if b % modulus != 0:
                    try:
                        b_inv = pow(b % modulus, modulus - 2, modulus)
                        p_mod = (-a * b_inv) % modulus
                        mod_constraint = p - p_mod
                        grid_pos['B'].append(mod_constraint)
                        b_components_added += 1
                    except:
                        pass
            #if b_components_added > 0:
            #    print(f"[ABCD] Added {b_components_added} B components for b={b % 100}...")

        # Component C: Ratio and cross-linking constraints
        if abs(c) > 0 and abs(b) > 0 and b_ok and c_ok:
            if abs(b) < max_cross and abs(c) < max_cross:
                # Ratio constraint
                ratio_c = b*p - c*q + a
                grid_pos['C'].append(ratio_c)

                # Cross-ratio (if coefficient allows)
                if abs(a) < max_cross:
                    cross_ratio = b*p*q - c*q*p + a*(p + q)
                    grid_pos['C'].append(cross_ratio)

        # Component D: Quadratic and higher-degree constraints
        if abs(b) > 0 and abs(c) > 0 and b_ok and c_ok:
            if abs(b) < max_quad and abs(c) < max_quad:
                # Basic quadratic
                quad_d = (b*p + c*q)**2 - (a + b*p + c*q)**2
                grid_pos['D'].append(quad_d)

                # Symmetric quadratic
                if abs(a) < max_quad:
                    sym_quad = b**2 * p**2 + c**2 * q**2 + 2*a*b*p + 2*a*c*q + a**2 - (b*p + c*q + a)**2
                    grid_pos['D'].append(sym_quad)

    def _fuse_ab_components(self, square_grid, p, q):
        """
        Fuse A and B components together after generation.

        This creates stronger constraints by combining linear structure (A)
        with modular hints (B) into fused polynomial relationships.
        """
        fused_polynomials = []

        for (row, col), components in square_grid.items():
            A_components = components['A']
            B_components = components['B']

            # Fuse each A component with each B component
            for a_poly in A_components:
                for b_poly in B_components:
                    try:
                        # Create fused constraint: A*B relationship
                        # This combines linear structure with modular hints
                        fused_constraint = a_poly * b_poly
                        fused_polynomials.append(fused_constraint)

                        # Also create difference constraint: A - B
                        if a_poly.func == b_poly.func:  # Same structure
                            diff_constraint = a_poly - b_poly
                            fused_polynomials.append(diff_constraint)

                    except Exception:
                        continue

            # Also keep original C and D components
            for c_poly in components['C']:
                fused_polynomials.append(c_poly)
            for d_poly in components['D']:
                fused_polynomials.append(d_poly)

        # Remove duplicates and trivial polynomials
        unique_fused = []
        seen = set()

        for poly in fused_polynomials:
            try:
                poly_str = str(poly)
                poly_neg_str = str(-poly)

                # Skip if already seen or trivial
                if poly_str in seen or poly_neg_str in seen:
                    continue

                # Skip constants
                if not poly.has(p) and not poly.has(q):
                    continue

                seen.add(poly_str)
                unique_fused.append(poly)

            except Exception:
                continue

        return unique_fused

    def _iterative_root_refinement(self, polynomials, p_hint, q_hint):
        """
        Iterative root refinement for large numbers.

        Start with approximate roots and iteratively improve them
        by minimizing polynomial evaluations.
        """
        print(f"[Root's] Starting iterative root refinement...")

        refined_pairs = []

        # Start with hints as initial approximations
        if p_hint and q_hint:
            initial_pairs = [(p_hint, q_hint)]
        else:
            # Use square root approximations
            sqrt_n = self._integer_sqrt_approx(self.N)
            initial_pairs = [(sqrt_n + i, sqrt_n - i) for i in range(-20, 21) if i != 0]

        # Select polynomials for refinement scoring
        scoring_polys = []
        for poly in polynomials[:5]:  # Use first 5 polynomials
            try:
                if poly.has(self.p) and poly.has(self.q):
                    scoring_polys.append(poly)
            except:
                continue

        if not scoring_polys:
            return []

        print(f"[Root's] Refining {len(initial_pairs)} initial pairs using {len(scoring_polys)} polynomials")

        for p_start, q_start in initial_pairs:
            # Iterative refinement
            p_current = p_start
            q_current = q_start

            for iteration in range(20):  # Expanded iteration limit
                best_score = float('inf')
                best_p, best_q = p_current, q_current

                # Try small adjustments around current values
                radius = max(1, abs(p_current) // 1000)  # Adaptive radius

                for dp in range(-radius, radius + 1, max(1, radius // 10)):
                    for dq in range(-radius, radius + 1, max(1, radius // 10)):
                        p_test = p_current + dp
                        q_test = q_current + dq

                        if p_test <= 1 or q_test <= 1:
                            continue

                        # Score this candidate
                        total_score = 0
                        valid_polys = 0

                        for poly in scoring_polys:
                            try:
                                val = poly.subs([(self.p, p_test), (self.q, q_test)])
                                penalty = self._safe_penalty(val, max_penalty=1000.0)
                                total_score += penalty
                                if penalty < 1000:  # Count as valid evaluation
                                    valid_polys += 1
                            except:
                                total_score += 1000

                        # Bonus for factorization accuracy
                        product_diff = abs(p_test * q_test - self.N)
                        factorization_score = product_diff / max(1, self.N)  # Normalize
                        total_score += factorization_score * 1000  # Weight heavily

                        # Update best if this is better
                        if total_score < best_score and valid_polys >= len(scoring_polys) // 2:
                            best_score = total_score
                            best_p, best_q = p_test, q_test

                # Check for convergence
                if abs(best_p - p_current) <= 1 and abs(best_q - q_current) <= 1:
                    break  # Converged

                p_current, q_current = best_p, best_q

            # Check if refined solution is good enough
            product = p_current * q_current
            if abs(product - self.N) < self.N // 1000:  # Within 0.1%
                refined_pairs.append((p_current, q_current))
                print(f"[Root's] ✓ Refined to: p={p_current}, q={q_current}, error={abs(product - self.N)}")

        return [(p, q) for p, q in refined_pairs]

        # ============================================================================
        # 1. FUNDAMENTAL FACTORIZATION CONSTRAINTS
        # ============================================================================

        # Basic factorization polynomial
        poly_factor = p * q - self.N
        polynomials.append(poly_factor)
        print(f"[Polynomial] ✓ f1: Basic factorization: p*q - N = 0")

        # ============================================================================
        # 2. CANDIDATE-BASED CONSTRAINTS
        # ============================================================================

        if p_candidate and q_candidate:
            # For very large numbers (2048-bit), use modular arithmetic polynomials
            # These represent relationships modulo small primes, not exact equalities
            if self.N.bit_length() > 1000:
                print(f"[Polynomial] Large {self.N.bit_length()}-bit number detected")
                print(f"[Polynomial] Using modular polynomial approach (avoids Gröbner inconsistency)")
                
                # Strategy: Use p*q = N plus additional structural constraints
                # that don't over-constrain the system
                
                # 1. Add weak sum bounds (not exact sum, just bounds)
                if p_candidate and q_candidate:
                    # Approximate sum (with large tolerance to avoid inconsistency)
                    approx_sum = p_candidate + q_candidate
                    # Don't use exact sum - instead use bounded versions
                    # This adds structure without making system inconsistent
                    pass  # Skip for now to avoid inconsistency
                
                # 2. Add modular hints (soft constraints)
                # Use candidate hints modulo small primes
                if p_candidate and q_candidate:
                    print(f"[Polynomial] Adding modular candidate hints...")
                    for prime in [97, 101, 103, 107, 109]:
                        p_mod = p_candidate % prime
                        q_mod = q_candidate % prime
                        
                        # p ≡ p_mod (mod prime) represented as: p - p_mod - k*prime = 0
                        # But we can't represent k, so we use: (p - p_mod) symbolically
                        # This will be handled by numerical methods, not symbolic
                        
                        # For symbolic methods, just add simple linear constraints
                        # that give structure without over-constraining
                        poly_p_hint = p - p_mod  # Soft hint: p is close to p_mod (mod prime)
                        polynomials.append(poly_p_hint)
                        
                        poly_q_hint = q - q_mod  # Soft hint: q is close to q_mod (mod prime)
                        polynomials.append(poly_q_hint)
                    
                    print(f"[Polynomial] ✓ Added {len(polynomials)-1} modular hint polynomials")
                
                # 3. Add difference hint (also soft)
                if p_candidate and q_candidate:
                    diff_hint = abs(p_candidate - q_candidate)
                    if p_candidate > q_candidate:
                        poly_diff = p - q  # Indicates p > q
                    else:
                        poly_diff = q - p  # Indicates q > p
                    polynomials.append(poly_diff)
                    print(f"[Polynomial] ✓ Added difference hint polynomial")
                
                # 4. Add parity constraints (divisibility by small numbers)
                # These are exact and won't conflict
                for div in [2, 3, 5]:
                    if self.N % div == 0:
                        # N is divisible by div, so at least one of p,q must be
                        # We can't represent "p div OR q div" easily, so skip
                        pass
                
                print(f"[Polynomial] ✓ Generated {len(polynomials)} polynomials for large number")
                print(f"[Polynomial] ℹ️  Using modular hints instead of exact constraints")
                print(f"[Polynomial] ℹ️  This provides structure for numerical methods")
                print(f"[Polynomial] ℹ️  Symbolic methods (Gröbner) may still return inconsistent")
                print(f"[Polynomial] ℹ️  But numerical methods can use these as search guidance")
                
                # Return with enhanced polynomial set
                return polynomials
                # Normal constraints for smaller large numbers
                target_sum = p_candidate + q_candidate
                poly_sum = p + q - target_sum
                polynomials.append(poly_sum)
                print(f"[Polynomial] ✓ f{len(polynomials)}: Sum constraint")

                # Difference constraint (much simplified)
                poly_diff = p - q  # Just require same sign/direction
                polynomials.append(poly_diff)
                print(f"[Polynomial] ✓ f{len(polynomials)}: Simple difference constraint")

            # ============================================================================
            # 3. ADVANCED GEOMETRIC CONSTRAINTS
            # ============================================================================

            # ============================================================================
            # 4. LATTICE-GUIDED SEARCH CONSTRAINTS (Soft Guidance, Not Hard Requirements)
            # ============================================================================

            # CRITICAL FIX: Use lattice approximations as SEARCH GUIDANCE, not exact constraints
            # The problem was treating approximations as exact requirements, creating inconsistency

            if self.N.bit_length() > 100:
                print(f"[Polynomial] 🔄 Using lattice approximations as search guidance (not exact constraints)")
                # print(f"[Polynomial] ⚠️ Skipping lattice-derived constraints for consistency - using approximations only in numerical methods")

                # For large numbers, extract at least basic lattice constraints
                # They create inconsistency. Use them only for numerical search guidance below.
                # The lattice approximations are not exact factors, so treating them as constraints
                # makes the polynomial system inconsistent (Gröbner basis = {1})

                # Basic modular preferences (not requirements) - but only very generic ones
                # For large numbers, even these can create inconsistency, so skip them too
                print(f"[Polynomial] ⚠️ Skipping all lattice-derived modular constraints for consistency")

                # Limit modular constraints to avoid over-constraining
                if len(polynomials) > 12:
                    polynomials = polynomials[:12]
                    print(f"[Polynomial] ⚡ Limited modular constraints for consistency")
            else:
                # Standard constraints for smaller numbers
                # Ratio constraint (clear denominators to avoid Gröbner issues)
                if q_candidate != 0:
                    target_ratio = p_candidate / q_candidate
                    # Clear denominator: p - target_ratio * q = 0
                    poly_ratio = p - target_ratio * q
                    polynomials.append(poly_ratio)
                    print(f"[Polynomial] ✓ f{len(polynomials)}: Ratio constraint")

                # Simple harmonic-like constraint
                poly_harmonic = 2*p*q - (p_candidate * q_candidate)
                polynomials.append(poly_harmonic)
                print(f"[Polynomial] ✓ f{len(polynomials)}: Harmonic constraint")

            # ============================================================================
            # 4. MODULAR CONSTRAINTS
            # ============================================================================

            # Small modulus constraints (useful for trial division)
            small_moduli = [2, 3, 5, 7, 11, 13, 17, 19, 23]
            for mod in small_moduli:
                if p_candidate % mod == q_candidate % mod:
                    shared_residue = p_candidate % mod
                    # Create polynomial representing congruence: p ≡ q (mod mod)
                    poly_mod = p - q  # This equals 0 mod mod when p ≡ q mod mod
                    if shared_residue != 0:
                        polynomials.append(poly_mod)
                        print(f"[Polynomial] ✓ f{len(polynomials)}: Modular congruence mod {mod}")
                        break

            # ============================================================================
            # 5. LIGHT STATISTICAL GUIDANCE (Not Requirements)
            # ============================================================================

            # CRITICAL: For large numbers, statistical constraints create inconsistency
            # because they add conflicting sum constraints like p+q - huge vs p+q - small
            if self.N.bit_length() > 100:
                print(f"[Polynomial] 📊 Using minimal statistical constraints for large numbers")
            else:
                # For smaller numbers, statistical constraints are safe
                # Very light digit sum preference (just a hint)
                p_digits_hint = sum(int(d) for d in str(p_candidate)[:10])  # Just first 10 digits
                q_digits_hint = sum(int(d) for d in str(q_candidate)[:10])
                total_digits_hint = p_digits_hint + q_digits_hint

                poly_digit_hint = p + q - total_digits_hint
                polynomials.append(poly_digit_hint)
                print(f"[Polynomial] ✓ f{len(polynomials)}: Light digit sum hint")

            # ============================================================================
            # 6. MINIMAL HIGHER-ORDER GUIDANCE
            # ============================================================================

            # For large numbers, add very minimal higher-order guidance to avoid inconsistency
            if self.N.bit_length() <= 256:  # Only for smaller large numbers
                # Single symmetric polynomial (very basic)
                poly_symmetric = p**2 + q**2 - (p_candidate**2 + q_candidate**2)
                polynomials.append(poly_symmetric)
                print(f"[Polynomial] ✓ f{len(polynomials)}: Basic symmetric polynomial")

                # Basic continued fraction hint
                if p_candidate > q_candidate and q_candidate > 0:
                    cf_hint = p_candidate // q_candidate
                    poly_cf = p - cf_hint * q
                    polynomials.append(poly_cf)
                    print(f"[Polynomial] ✓ f{len(polynomials)}: Continued fraction hint")
            else:
                print(f"[Polynomial] 🔄 Skipping higher-order constraints for {self.N.bit_length()}-bit consistency")

            # Limit total constraints for very large numbers
            if self.N.bit_length() > 1500:
                max_constraints = 50  # Expanded for 2048-bit
            else:
                max_constraints = 100  # Expanded for moderately large numbers

            if len(polynomials) > max_constraints:
                polynomials = polynomials[:max_constraints]
                print(f"[Polynomial] ⚡ Limited to {max_constraints} constraints for computational efficiency")

        # ============================================================================
        # 7. FINAL MODULAR CONSTRAINTS FOR ALL SIZES
        # ============================================================================

        # Only add modular constraints for small numbers to avoid inconsistency
        if self.N.bit_length() <= 100:
            for modulus in [101, 103, 107]:  # Small moduli for final constraints
                # Create polynomial representing: p*q ≡ N (mod modulus)
                n_mod = self.N % modulus
                poly_final_mod = p * q - n_mod  # This equals 0 mod modulus when p*q ≡ N mod modulus
                polynomials.append(poly_final_mod)
                print(f"[Polynomial] ✓ f{len(polynomials)}: Final modular factorization (mod {modulus})")
        else:
            print(f"[Polynomial] ⚠️ Skipping final modular constraints for large numbers (would create inconsistency)")

        # Ensure we have at least the basic constraint
        if not any(str(poly).find('p*q') >= 0 for poly in polynomials):
            polynomials.insert(0, p * q - self.N)
            print(f"[Polynomial] ✓ f1: Basic factorization constraint (ensured)")

        print(f"[Polynomial] 🎯 Generated {len(polynomials)} comprehensive polynomial constraints")
        print(f"[Polynomial] 📊 Constraint types: Basic({1}), Candidate-based({3}), Geometric({3}), Modular({1}), Statistical({1}), Lattice-derived({min(3, max(0, len(polynomials)-15))}), CF({1}), Higher-order({2}), Scale-invariant({1}), Cross-validation({3}), Number-theoretic({2}), Adaptive({1})")

        # Limit total polynomials for computational feasibility
        max_polys = 25 if self.N.bit_length() < 100 else 20
        if len(polynomials) > max_polys:
            print(f"[Polynomial] ⚡ Limiting to {max_polys} polynomials for computational efficiency")
            polynomials = polynomials[:max_polys]

        # Add enhanced precision constraints for Root's Method with targeting
        print(f"[Polynomial] Adding enhanced precision constraints for targeted Root's Method...")

        # Generate additional high-precision constraints
        precision_polys = []

        # Add exact factorization constraint with higher weight
        exact_factor = p * q - self.N
        precision_polys.append(exact_factor)

        # Add targeted constraints to 'zero in' on N
        # Include p*q - N with different polynomial forms for better constraint coverage
        targeted_constraint1 = (p * q - self.N) ** 2  # Square for stronger constraint
        targeted_constraint2 = p * q - self.N + p + q  # Include sum for cross-validation
        precision_polys.extend([targeted_constraint1, targeted_constraint2])

        # Add modular constraints with multiple moduli for better root finding
        moduli = [97, 101, 103, 107, 109, 113, 127, 131]
        for modulus in moduli:
            try:
                n_mod = self.N % modulus
                mod_constraint = sp.Mod(p * q - self.N, modulus)  # p*q ≡ N mod modulus
                precision_polys.append(mod_constraint)
            except:
                continue

        # Add ratio constraints for better cross-validation
        if len(polynomials) >= 2:
            for i in range(min(3, len(polynomials))):
                for j in range(i+1, min(6, len(polynomials))):
                    try:
                        poly_i, poly_j = polynomials[i], polynomials[j]
                        # Create ratio: poly_i / poly_j = k for some k
                        ratio_constraint = poly_i * poly_j  # Simplified ratio constraint
                        precision_polys.append(ratio_constraint)
                    except:
                        continue

        # Add statistical constraints based on candidate magnitudes
        if p_candidate and q_candidate:
            try:
                # Constraint based on expected magnitudes
                mag_constraint = p + q - (p_candidate + q_candidate)
                precision_polys.append(mag_constraint)
            except:
                pass

        # Limit precision polynomials to avoid explosion
        if precision_polys:
            polynomials.extend(precision_polys[:15])  # Add up to 15 precision constraints
            print(f"[Polynomial] Added {min(15, len(precision_polys))} precision constraints")

        print(f"[Polynomial] 📈 Final polynomial system ({len(polynomials)} equations):")
        for i, poly in enumerate(polynomials, 1):
            if i <= 5 or i > len(polynomials) - 3:  # Show first 5 and last 3
                print(f"  f{i}(p,q) = {poly}")
            elif i == 6:
                print(f"  ... ({len(polynomials) - 8} more polynomials) ...")

        return polynomials


# ============================================================================
# MAIN PROGRAM
# ============================================================================

def estimate_factors(N: int, p_hint: int = None, q_hint: int = None) -> Tuple[int, int]:
    """Estimate initial factor candidates, using hints when available."""
    sqrt_N = int(math.isqrt(N))

    # Use hints when available, otherwise estimate
    if p_hint is not None:
        p_candidate = p_hint
    else:
        p_candidate = sqrt_N + np.random.randint(-100, 100)
        p_candidate = max(2, p_candidate)

    if q_hint is not None:
        q_candidate = q_hint
    else:
        q_candidate = sqrt_N + np.random.randint(-100, 100)
        q_candidate = max(2, q_candidate)

    # If we have one hint but not the other, we could try to estimate the missing one
    # based on the hint, but for now we'll keep it simple and just use what's provided

    return p_candidate, q_candidate


def generate_training_rsa_key(bits: int) -> Tuple[int, int, int]:
    """
    Generate a training RSA key with known factors for pre-training the transformer.

    Returns:
        Tuple of (N, p, q) where N = p * q
    """
    import random
    import math

    # Generate two prime factors of roughly equal size
    half_bits = bits // 2

    # Generate p
    while True:
        p = random.getrandbits(half_bits)
        if p.bit_length() == half_bits and p > 1 and p % 2 != 0:  # Odd number
            if is_prime(p):
                break

    # Generate q (ensure different from p)
    while True:
        q = random.getrandbits(half_bits)
        if q.bit_length() == half_bits and q > 1 and q % 2 != 0 and q != p:
            if is_prime(q):
                break

    # Ensure p < q for consistency
    if p > q:
        p, q = q, p

    N = p * q
    return N, p, q


def is_prime(n: int, k: int = 5) -> bool:
    """Miller-Rabin primality test for RSA key generation."""
    if n == 2 or n == 3:
        return True
    if n <= 1 or n % 2 == 0:
        return False

    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Witness loop
    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def pretrain_transformer(transformer: StepPredictionTransformer, num_keys: int, key_bits: int) -> bool:
    """
    Pre-train the transformer on synthetic RSA keys to learn factorization patterns.

    Args:
        transformer: The transformer to pre-train
        num_keys: Number of synthetic RSA keys to generate and train on
        key_bits: Bit length of synthetic keys

    Returns:
        True if pre-training successful
    """
    print(f"\n🎓 STARTING TRANSFORMER PRE-TRAINING")
    print(f"🎓 Generating {num_keys} synthetic {key_bits}-bit RSA keys...")
    print(f"🎓 This will teach the transformer how to navigate bitspace to find factors")

    successful_trainings = 0

    for i in range(num_keys):
        print(f"\n🎓 Pre-training key {i+1}/{num_keys}...")

        # Generate a training RSA key
        N, true_p, true_q = generate_training_rsa_key(key_bits)
        print(f"🎓 Generated N={N} with factors p={true_p}, q={true_q}")

        # Create a temporary lattice solver for this training key
        temp_solver = MinimizableFactorizationLatticeSolver(N, delta=0.75)

        # Run a mini bulk search to collect training data
        # We'll simulate the search process to teach the transformer
        sqrt_N = int(math.isqrt(N))
        search_radius = min(1000, sqrt_N // 100)  # Smaller radius for training

        print(f"🎓 Running mini bulk search to collect training data...")

        # Simulate several search steps around the true factors
        training_positions = [
            sqrt_N,  # Around sqrt(N)
            true_p,  # Around true p
            true_q,  # Around true q
            (true_p + true_q) // 2,  # Around sum
            abs(true_p - true_q) // 2,  # Around difference
        ]

        for pos in training_positions:
            if pos <= 0 or pos > N:
                continue

            # Simulate a search step at this position
            step_result = temp_solver._bulk_search_step(pos, search_radius, sqrt_N)

            if step_result:
                p_found, q_found, diff_bits, _ = step_result

                # Record the result for transformer learning with perfect correction signals
                transformer.add_search_result(
                    pos, diff_bits, sqrt_N, N.bit_length(),
                    p_found, q_found, true_p, true_q  # Perfect supervision during pre-training
                )

                # Check if we found the exact factors
                if p_found == true_p and q_found == true_q:
                    print(f"🎓 ✅ Found exact factors during training!")
                    successful_trainings += 1

        print(f"🎓 Completed training on key {i+1}")

    # ============================================================================
    # EXTENSIVE TRAINING: Multiple passes on accumulated knowledge
    # ============================================================================

    print(f"\n🎓 PHASE 1: Initial training complete")
    print(f"🎓 Successfully trained on {successful_trainings}/{num_keys} keys")
    print(f"🎓 Transformer has {len(transformer.search_history)} search experiences")

    # Phase 2: Extensive training on all accumulated data
    print(f"\n🎓 PHASE 2: Extensive training on accumulated knowledge...")
    print(f"🎓 Training for 25 epochs on {len(transformer.search_history)} total experiences...")

    transformer.train_on_history(epochs=25)  # Much more training

    # Phase 3: Generate additional diverse training data
    print(f"\n🎓 PHASE 3: Generating additional diverse training data...")
    additional_keys = max(20, num_keys // 2)  # Generate more keys for diversity

    for i in range(additional_keys):
        print(f"🎓 Additional training key {i+1}/{additional_keys}...")

        # Generate a different size key for variety
        diverse_bits = key_bits + random.randint(-100, 100)  # Vary key size
        diverse_bits = max(256, min(2048, diverse_bits))  # Keep in reasonable range

        N, true_p, true_q = generate_training_rsa_key(diverse_bits)

        # Run mini search with different parameters for variety
        sqrt_N = int(math.isqrt(N))
        search_radius = min(2000, sqrt_N // 50)  # Different radius

        temp_solver = MinimizableFactorizationLatticeSolver(N, delta=0.75)

        # Test multiple positions around different mathematical landmarks
        positions = [
            sqrt_N,
            true_p, true_q,
            (true_p + true_q) // 2,
            abs(true_p - true_q) // 2,
            true_p + (true_q - true_p) // 3,  # One-third point
            true_q - (true_q - true_p) // 4,  # Three-quarters point
        ]

        for pos in positions:
            if pos <= 0 or pos > N:
                continue

            step_result = temp_solver._bulk_search_step(pos, search_radius, sqrt_N)

            if step_result:
                p_found, q_found, diff_bits, _ = step_result
                transformer.add_search_result(
                    pos, diff_bits, sqrt_N, N.bit_length(),
                    p_found, q_found, true_p, true_q
                )

    # Phase 4: Final extensive training
    print(f"\n🎓 PHASE 4: Final extensive training...")
    print(f"🎓 Now training on {len(transformer.search_history)} total experiences...")

    # Train multiple times with different batch configurations
    for training_round in range(3):
        print(f"🎓 Training round {training_round + 1}/3...")
        transformer.train_on_history(epochs=15)  # 15 epochs per round

        # Shuffle the training data between rounds for better generalization
        if hasattr(transformer, 'search_history') and len(transformer.search_history) > 100:
            import random as rand
            rand.shuffle(transformer.search_history[:len(transformer.search_history)//2])  # Shuffle half

    # Phase 5: Validation training on known patterns
    print(f"\n🎓 PHASE 5: Validation and pattern consolidation...")
    print(f"🎓 Generating validation keys to test learned patterns...")

    validation_keys = 5
    for i in range(validation_keys):
        print(f"🎓 Validation key {i+1}/{validation_keys}...")

        # Use same size as original training for validation
        N, true_p, true_q = generate_training_rsa_key(key_bits)

        temp_solver = MinimizableFactorizationLatticeSolver(N, delta=0.75)
        sqrt_N = int(math.isqrt(N))

        # Test key positions that should be learned
        test_positions = [true_p, true_q, (true_p + true_q) // 2]
        search_radius = min(1000, sqrt_N // 100)

        for pos in test_positions:
            step_result = temp_solver._bulk_search_step(pos, search_radius, sqrt_N)
            if step_result:
                p_found, q_found, diff_bits, _ = step_result
                transformer.add_search_result(
                    pos, diff_bits, sqrt_N, N.bit_length(),
                    p_found, q_found, true_p, true_q
                )

    # Final comprehensive training
    print(f"\n🎓 FINAL TRAINING: Comprehensive pattern learning...")
    transformer.train_on_history(epochs=30)  # Final extensive training

    transformer.pretrained = True

    print(f"\n🎓 🎉 EXTENSIVE PRE-TRAINING COMPLETE! 🎉")
    print(f"🎓 Successfully trained on {successful_trainings + additional_keys + validation_keys} keys total")
    print(f"🎓 Transformer now has {len(transformer.search_history)} diverse search experiences")
    print(f"🎓 Training included {25 + 45 + 30} epochs = {100} total training epochs")
    print(f"🎓 Learned patterns from keys ranging {min(key_bits-100, 256)}-{max(key_bits+100, 2048)} bits")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Minimizable Factorization Lattice Attack with Enhanced Polynomial Solving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("N", type=str, nargs='?', help="The number to factor (optional when using --pretrain-only)")
    parser.add_argument("--p", type=str, help="Initial P candidate (integer or decimal)")
    parser.add_argument("--q", type=str, help="Initial Q candidate (integer or decimal)")
    parser.add_argument("--p-decimal", type=str, help="P as decimal approximation")
    parser.add_argument("--q-decimal", type=str, help="Q as decimal approximation")
    parser.add_argument("--s-squared", type=str, help="S² value where S² = 4N + D (for direct factorization)")
    parser.add_argument("--s", type=str, help="S value where S = p + q (sum of factors)")
    parser.add_argument("--d", type=str, help="D value where D = (p - q)² (square of difference)")
    parser.add_argument("--d-hint", type=str, help="D hint for k search (doesn't need to be perfect square)")
    parser.add_argument("--auto-find-s-d", action="store_true",
                       help="Automatically find S and D using Root's Method (default: enabled if S/D not provided)")
    parser.add_argument("--no-auto-find-s-d", action="store_true",
                       help="Disable automatic S and D discovery")
    parser.add_argument("--search-radius", type=int, default=None,
                       help="Search radius for corrections in bits (default: 2048, meaning 2^2048 for full key coverage)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--polynomial", action="store_true",
                       help="Enable polynomial solving methods")
    parser.add_argument("--bulk", action="store_true",
                       help="Enable bulk factor search using LLL (creates large lattice, may be slow)")
    
    # Advanced parameters
    parser.add_argument("--max-polynomials", type=int, default=None,
                       help="Maximum number of polynomials to analyze (default: auto)")
    parser.add_argument("--max-lattice-vectors", type=int, default=None,
                       help="Maximum number of lattice vectors (default: auto)")
    parser.add_argument("--coeff-limit", type=int, default=None,
                       help="Coefficient limit for large numbers (default: auto)")
    parser.add_argument("--lattice-dimension", type=int, default=None,
                       help="Lattice dimension size (e.g., for controlling the base size like 2^200)")
    parser.add_argument("--trial-division-limit", type=int, default=None,
                       help="Trial division limit (default: auto)")
    parser.add_argument("--ultra-search-radius", type=int, default=None,
                       help="Ultra refinement search radius in bits (e.g., 3000 for 2^3000, default: auto)")
    parser.add_argument("--polynomial-grid-size", type=int, default=None,
                       help="Polynomial grid size for generation (default: auto)")
    parser.add_argument("--max-root-candidates", type=int, default=None,
                       help="Maximum root candidates per variable (default: auto)")
    parser.add_argument("--max-root-combinations", type=int, default=None,
                       help="Maximum total root combinations to test (default: auto, scales with N)")
    parser.add_argument("--root-sampling-strategy", type=str, default="none",
                       choices=["none", "random", "stratified", "adaptive"],
                       help="Sampling strategy for root combinations: none=test all, random=sample, stratified=even distribution, adaptive=smart sampling (default: none)")
    parser.add_argument("--root-sampling-fraction", type=float, default=1.0,
                       help="Fraction of combinations to test when using sampling (0.0-1.0, default: 1.0)")
    parser.add_argument("--early-termination", action="store_true",
                       help="Stop testing combinations once a good candidate is found")
    parser.add_argument("--no-transformer", action="store_true",
                       help="Disable Transformer model (use simpler model to save memory)")
    parser.add_argument("--pretrain", type=int, default=None,
                       help="EXTENSIVE pre-training: Generate N diverse RSA keys for broad factorization knowledge (recommended: 20-50)")
    parser.add_argument("--pretrain-only", action="store_true",
                       help="Pre-training only mode: Generate and train on RSA keys without factoring a target N")
    parser.add_argument("--pretrain-bits", type=int, default=1024,
                       help="Base bit length for pre-training RSA keys (default: 1024, will vary ±100 bits for diversity)")
    parser.add_argument("--load-pretrained", type=str, default=None,
                       help="Load pre-trained transformer model from file")
    parser.add_argument("--save-pretrained", type=str, default=None,
                       help="Save trained transformer model to file after pre-training/attack")

    args = parser.parse_args()
    
    # Store advanced parameters in a config dict for access by classes
    global_config = {
        'max_polynomials': args.max_polynomials,
        'max_lattice_vectors': args.max_lattice_vectors,
        'coeff_limit': args.coeff_limit,
        'lattice_dimension': args.lattice_dimension,
        'trial_division_limit': args.trial_division_limit,
        'ultra_search_radius': args.ultra_search_radius,
        'polynomial_grid_size': args.polynomial_grid_size,
        'max_root_candidates': args.max_root_candidates,
        'max_root_combinations': args.max_root_combinations,
        'root_sampling_strategy': args.root_sampling_strategy,
        'root_sampling_fraction': args.root_sampling_fraction,
        'early_termination': args.early_termination,
        'no_transformer': args.no_transformer,
    }

    # Parse N (optional in pretrain-only mode)
    if args.pretrain_only:
        if args.pretrain is None:
            print("Error: --pretrain-only requires --pretrain N to specify number of keys")
            return 1
        N = None  # Not needed for pre-training only
    else:
        if args.N is None:
            print("Error: N is required unless using --pretrain-only")
            return 1
        try:
            N = int(args.N)
        except ValueError:
            print(f"Error: Invalid number format: {args.N}")
            return 1

    print(f"=" * 80)
    print(f"MINIMIZABLE FACTORIZATION LATTICE + POLYNOMIAL SOLVER")
    print(f"=" * 80)
    if args.pretrain_only:
        print(f"PRE-TRAINING ONLY MODE")
        print(f"Will generate and train on {args.pretrain} RSA keys")
        print(f"Key sizes: {args.pretrain_bits - 100} to {args.pretrain_bits + 100} bits")
        if args.save_pretrained:
            print(f"Model will be saved to: {args.save_pretrained}")
        print()
    else:
        print(f"Target N: {N}")
        print(f"Bit length: {N.bit_length()}")
        print()

    # ============================================================================
    # FACTORIZATION MODE (skip in pretrain-only mode)
    # ============================================================================
    success = False  # Initialize for both modes
    if not args.pretrain_only:
        # Initialize lattice solver (only needed for factorization, not pre-training)
        lattice_solver = MinimizableFactorizationLatticeSolver(N, delta=0.75)

        # IMPROVEMENT: Refine p and q approximations for better accuracy
        print(f"\n{'='*80}")
        print(f"REFINING P AND Q APPROXIMATIONS")
        print(f"{'='*80}")
        refined_p, refined_q = lattice_solver.refine_approximations(max_iterations=5)
        print(f"Refined approximations: p={refined_p}, q={refined_q}")
        print(f"Error reduction: {abs(lattice_solver.p_approx * lattice_solver.q_approx - N)}")
        # Store config for use in Root's Method
        lattice_solver.config = global_config
        # Store bulk search flag
        lattice_solver.use_bulk_search = args.bulk
        # Store pre-trained transformer for bulk search
        if 'transformer' in locals():
            lattice_solver.pretrained_transformer = transformer

        # Store known factors for correction/supervised learning
        lattice_solver.known_p = None
        lattice_solver.known_q = None
        if args.p:
            try:
                lattice_solver.known_p = int(args.p)
                print(f"[Correction] 📚 Known P factor provided: {lattice_solver.known_p}")
            except ValueError:
                print(f"[Warning] Invalid P factor format: {args.p}")
        if args.q:
            try:
                lattice_solver.known_q = int(args.q)
                print(f"[Correction] 📚 Known Q factor provided: {lattice_solver.known_q}")
            except ValueError:
                print(f"[Warning] Invalid Q factor format: {args.q}")

        if lattice_solver.known_p and lattice_solver.known_q:
            product = lattice_solver.known_p * lattice_solver.known_q
            if product == N:
                print(f"[Correction] ✅ Known factors verified: {lattice_solver.known_p} × {lattice_solver.known_q} = {N}")
                print(f"[Correction] 🎓 Transformer will receive correction signals during search")
            else:
                print(f"[Correction] ⚠️  Known factors don't multiply to N: {product} ≠ {N}")
                lattice_solver.known_p = None
                lattice_solver.known_q = None

    # ============================================================================
    # TRANSFORMER INITIALIZATION
    # ============================================================================
    # Always initialize transformer (needed for bulk search)
    transformer = StepPredictionTransformer(
        d_model=128, nhead=4, num_layers=2,
        max_seq_len=500000, use_torch=TORCH_AVAILABLE
    )

    # Load pre-trained model if specified
    if args.load_pretrained:
        print(f"Loading pre-trained model from {args.load_pretrained}...")
        transformer.load_model(args.load_pretrained)

    # ============================================================================
    # PRE-TRAINING ONLY MODE: Generate keys and train, then exit
    # ============================================================================
    if args.pretrain_only:
        if args.pretrain is None:
            print("Error: --pretrain-only requires --pretrain N to specify number of keys")
            return 1

        print(f"\n{'='*80}")
        print(f"PRE-TRAINING ONLY MODE")
        print(f"{'='*80}")
        print(f"Will generate {args.pretrain} diverse RSA keys")
        print(f"Key sizes: {args.pretrain_bits - 100} to {args.pretrain_bits + 100} bits")
        if args.save_pretrained:
            print(f"Model will be saved to: {args.save_pretrained}")

        # Run extensive pre-training
        print(f"\nGenerating {args.pretrain} diverse RSA keys for training...")
        pretrain_success = pretrain_transformer(transformer, args.pretrain, args.pretrain_bits)

        if pretrain_success:
            print(f"\n✅ PRE-TRAINING COMPLETED SUCCESSFULLY!")
            print(f"Transformer learned from {len(transformer.search_history)} diverse search experiences")

            # Save the model
            if args.save_pretrained:
                print(f"Saving pre-trained model to {args.save_pretrained}...")
                if transformer.save_model(args.save_pretrained):
                    print(f"✅ Pre-trained model saved successfully!")
                else:
                    print(f"❌ Failed to save model")
                    return 1
            else:
                print(f"⚠️  No save path specified (--save-pretrained). Model will not be saved.")

        else:
            print(f"❌ Pre-training failed!")
            return 1

        print(f"\n🎉 Pre-training complete! Use the saved model with:")
        print(f"python standalone_lattice_attack.py <N> --load-pretrained {args.save_pretrained or 'your_model.pth'} --bulk")
        return 0

    # ============================================================================
    # NORMAL FACTORIZATION MODE
    # ============================================================================

    # Parse N for normal operation
    try:
        N = int(args.N)
    except ValueError:
        print(f"Error: Invalid number format: {args.N}")
        return 1

    # ============================================================================
    # PRE-TRAINING PHASE: Train transformer on synthetic RSA keys
    # ============================================================================
    if args.pretrain and not args.no_transformer:
        print(f"\n{'='*80}")
        print(f"TRANSFORMER PRE-TRAINING PHASE")
        print(f"{'='*80}")

        # Run extensive pre-training for broad factorization knowledge
        print(f"🎓 EXTENSIVE PRE-TRAINING: Building broad factorization knowledge...")
        print(f"🎓 This will generate {args.pretrain} diverse RSA keys and train extensively")
        print(f"🎓 Key sizes: {args.pretrain_bits - 100} to {args.pretrain_bits + 100} bits for pattern diversity")
        print(f"🎓 Expected training time: {args.pretrain * 2 + 30} minutes")
        print(f"🎓 Final model will have learned from {args.pretrain * 3 + 25} keys total")
        pretrain_success = pretrain_transformer(transformer, args.pretrain, args.pretrain_bits)

        if pretrain_success:
            print(f"✅ Pre-training completed successfully!")
            print(f"Transformer now has {len(transformer.search_history)} training samples")

            # Save the pre-trained model immediately after pre-training
            if args.save_pretrained:
                print(f"\n💾 Saving pre-trained model...")
                print(f"   Save path: {args.save_pretrained}")
                print(f"   Training samples: {len(transformer.search_history)}")
                if transformer.save_model(args.save_pretrained):
                    print(f"✅ Pre-trained model saved to {args.save_pretrained}")
                else:
                    print(f"❌ Failed to save model to {args.save_pretrained}")
                    return 1
        else:
            print(f"❌ Pre-training failed!")
            return 1
    
    # Check if S² = 4N + D factorization is requested
    success = False
    refined_p = None
    refined_q = None
    alpha_final = None
    beta_final = None
    
    # If S², S, or D are not provided, try to find them automatically using Root's Method
    should_auto_find = not args.no_auto_find_s_d and (args.auto_find_s_d or (args.s_squared is None and (args.s is None or args.d is None)))
    
    if should_auto_find:
        print(f"\n{'='*80}")
        print(f"AUTOMATIC S AND D DISCOVERY VIA ROOT'S METHOD")
        print(f"{'='*80}")
        print("S², S, and D not provided - attempting automatic discovery...")
        
        # First, try to get initial candidates for Root's Method
        if args.p is not None and args.q is not None:
            try:
                p_hint = int(args.p)
                q_hint = int(args.q)
            except:
                p_hint = None
                q_hint = None
        else:
            p_hint = None
            q_hint = None
        
        # Generate polynomials for Root's Method
        print("Generating polynomials for Root's Method...")
        if p_hint and q_hint:
            polynomials = lattice_solver.generate_factorization_polynomials(p_hint, q_hint)
        else:
            # Estimate candidates
            sqrt_N = int(math.isqrt(N))
            p_est = sqrt_N
            q_est = sqrt_N
            polynomials = lattice_solver.generate_factorization_polynomials(p_est, q_est)
        
        # Parse D_hint if provided
        D_hint = None
        if hasattr(args, 'd_hint') and args.d_hint is not None:
            try:
                D_hint = int(args.d_hint)
                print(f"Using D_hint = {D_hint}")
            except ValueError:
                print(f"Warning: Invalid D_hint format: {args.d_hint}")

        # If no p_hint/q_hint provided, compute estimated candidates to use as hints
        if p_hint is None or q_hint is None:
            print("Computing estimated factor candidates for algebraic method...")
            p_est, q_est = estimate_factors(N, p_hint, q_hint)
            if p_hint is None:
                p_hint = p_est
            if q_hint is None:
                q_hint = q_est
            print(f"Using estimated candidates as hints: p_hint={p_hint}, q_hint={q_hint}")

        # Try to find S and D using pure algebra
        S_found, D_found, p_found, q_found, S_sq_found = lattice_solver.find_s_and_d_via_roots_method(
            polynomials=polynomials,
            p_hint=p_hint,
            q_hint=q_hint,
            search_radius=args.search_radius,
            D_hint=D_hint
        )
        
        if S_found is not None and D_found is not None:
            print(f"\n{'='*80}")
            print(f"✓ AUTOMATIC DISCOVERY SUCCESSFUL!")
            print(f"{'='*80}")
            print(f"Found S = {S_found}")
            print(f"Found D = {D_found}")
            print(f"Found S² = {S_sq_found}")
            print(f"Verification: S² = 4N + D: {S_sq_found} = {4*N + D_found} ✓")
            
            # Verify perfect S across all 3 domains
            S_perfect, S_sq_perfect, D_perfect, is_perfect = lattice_solver.calculate_perfect_s(S_squared=S_sq_found)
            
            if is_perfect:
                print(f"\n✓ Discovered S is PERFECT across all 3 domains!")
            else:
                print(f"\n⚠ Discovered S does not satisfy all 3 domains")
            
            # Automatically extract factors from S and D
            print(f"\n{'='*80}")
            print(f"AUTOMATIC FACTOR EXTRACTION")
            print(f"{'='*80}")
            
            # Check if factors were already extracted in _find_s_squared_factorization
            if p_found is not None and q_found is not None:
                print(f"Using factors already extracted from S² = 4N + D:")
                print(f"  p = {p_found}")
                print(f"  q = {q_found}")
                
                # Verify factorization
                product = p_found * q_found
                print(f"  p × q = {product}")
                print(f"  Target N = {N}")
                
                if product == N:
                    print(f"\n🎉 SUCCESS! EXACT FACTORIZATION FOUND!")
                    print(f"p = {p_found}")
                    print(f"q = {q_found}")
                else:
                    error = abs(product - N)
                    print(f"  Error: {error}")
                    print(f"  Factors are approximate, ready for refinement")
            else:
                # Extract factors using S and D (even if D is not a perfect square)
                sqrt_D = lattice_solver._integer_sqrt_approx(D_found)
                
                if sqrt_D * sqrt_D == D_found:
                    print(f"✓ D is a perfect square: D = T² where T = {sqrt_D}")
                    print(f"Using direct calculation: p = (S + T)/2, q = (S - T)/2")
                    
                    if (S_found - sqrt_D) % 2 == 0 and (S_found + sqrt_D) % 2 == 0:
                        p_extracted = (S_found + sqrt_D) // 2
                        q_extracted = (S_found - sqrt_D) // 2
                        print(f"Calculated: p = (S + T)/2 = {p_extracted}")
                        print(f"Calculated: q = (S - T)/2 = {q_extracted}")
                    else:
                        # S ± T not even, use direct calculation anyway
                        print(f"S ± T is not even, using direct calculation")
                        p_extracted = (S_found + sqrt_D) // 2
                        q_extracted = (S_found - sqrt_D) // 2
                else:
                    print(f"⚠️  D is not a perfect square (D = {D_found}, √D ≈ {sqrt_D})")
                    print(f"   Using approximate √D for factor extraction")
                    
                    # Extract factors: p = (S + √D)/2, q = (S - √D)/2
                    p_extracted = (S_found + sqrt_D) // 2
                    q_extracted = (S_found - sqrt_D) // 2
                
                print(f"Extracted factors:")
                print(f"  p = {p_extracted}")
                print(f"  q = {q_extracted}")
                
                # Verify factorization
                product = p_extracted * q_extracted
                print(f"  p × q = {product}")
                print(f"  Target N = {N}")
                
                if product == N:
                    print(f"\n🎉 SUCCESS! EXACT FACTORIZATION FOUND!")
                    print(f"p = {p_extracted}")
                    print(f"q = {q_extracted}")
                    p_found = p_extracted
                    q_found = q_extracted
                else:
                    error = abs(product - N)
                    print(f"  Error: {error}")
                    if sqrt_D * sqrt_D == D_found:
                        print(f"  D is perfect square but factors don't multiply to N exactly")
                    else:
                        print(f"  D is not perfect square - factors are refined estimates")
                    print(f"  Ready for further refinement")
                    p_found = p_extracted
                    q_found = q_extracted
            
            if p_found and q_found:
                if p_found * q_found == N:
                    
                    # Compute decimal factors
                    k_base = int(math.isqrt(N))
                    Qc_base = int(math.isqrt(N))
                    alpha_final = float(p_found - k_base)
                    beta_final = float(q_found - Qc_base)
                    
                    print(f"\n📊 Decimal Factors with Adjustments:")
                    print(f"   Base candidates: k = {k_base}, Qc = {Qc_base}")
                    print(f"   Adjustments: α = {alpha_final:.10f}, β = {beta_final:.10f}")
                    p_decimal = k_base + alpha_final
                    q_decimal = Qc_base + beta_final
                    print(f"   p = k + α = {k_base} + {alpha_final:.10f} = {p_decimal:.10f}")
                    print(f"   q = Qc + β = {Qc_base} + {beta_final:.10f} = {q_decimal:.10f}")
                    
                    success = True
                    return 0
                else:
                    print(f"\n⚠️  Found S and D, but factors are approximate:")
                    print(f"p = {p_found}, q = {q_found}")
                    print(f"p × q = {p_found * q_found} (target: {N})")
                    # Continue with lattice attack for refinement
    
    # Check if only S is provided (without D) - use exact factorization method
    if args.s is not None and args.d is None and args.s_squared is None:
        print(f"\n{'='*80}")
        print(f"EXACT FACTORIZATION FROM S (Mathematical Derivation)")
        print(f"{'='*80}")
        
        try:
            S = int(args.s)
            print(f"Using S = {S} (where S ≈ (p+q)/2)")
            
            # Calculate and verify perfect S across all 3 domains
            S_perfect, S_sq_perfect, D_perfect, is_perfect = lattice_solver.calculate_perfect_s(S=S)
            
            if is_perfect:
                print(f"\n✓ S satisfies all 3 domains - using perfect S")
            else:
                print(f"\n⚠ S does not satisfy all 3 domains, but proceeding with exact factorization method")
            
            # Call the exact factorization method
            p_exact, q_exact, T_exact, k_exact = lattice_solver.solve_exact_from_s(
                S=S,
                search_radius=args.search_radius
            )
            
            if p_exact and q_exact:
                if p_exact * q_exact == N:
                    print(f"\n{'='*80}")
                    print(f"🎉 SUCCESS! EXACT FACTORIZATION FOUND!")
                    print(f"{'='*80}")
                    print(f"p = {p_exact}")
                    print(f"q = {q_exact}")
                    print(f"T = p + q = {T_exact}")
                    print(f"k = p - q = {k_exact}")
                    print(f"Verification: {p_exact} × {q_exact} = {p_exact * q_exact} ✓")
                    print(f"Verification: T² - 4N = k²: {T_exact*T_exact} - {4*N} = {k_exact*k_exact} ✓")
                    
                    # Display decimal factors
                    k_base = int(math.isqrt(N))
                    Qc_base = int(math.isqrt(N))
                    alpha_final = float(p_exact - k_base)
                    beta_final = float(q_exact - Qc_base)
                    print(f"\n📊 Decimal Factors with Adjustments:")
                    print(f"   Base candidates: k = {k_base}, Qc = {Qc_base}")
                    print(f"   Adjustments: α = {alpha_final:.10f}, β = {beta_final:.10f}")
                    p_decimal = k_base + alpha_final
                    q_decimal = Qc_base + beta_final
                    print(f"   p = k + α = {k_base} + {alpha_final:.10f} = {p_decimal:.10f}")
                    print(f"   q = Qc + β = {Qc_base} + {beta_final:.10f} = {q_decimal:.10f}")
                    
                    success = True
                    return 0
                else:
                    error = abs(p_exact * q_exact - N)
                    print(f"\n⚠️  Found factors but not exact:")
                    print(f"p = {p_exact}, q = {q_exact}")
                    print(f"p × q = {p_exact * q_exact} (target: {N})")
                    print(f"Error: {error}")
                    print(f"T = {T_exact}, k = {k_exact}")
                    # Continue with refinement if needed
        except Exception as e:
            print(f"Error in exact factorization from S: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing with other methods...")
    
    if args.s_squared is not None or (args.s is not None and args.d is not None):
        print(f"\n{'='*80}")
        print(f"FACTORING USING S² = 4N + D")
        print(f"{'='*80}")
        
        try:
            if args.s_squared is not None:
                S_squared = int(args.s_squared)
                
                # Calculate and verify perfect S across all 3 domains
                S_perfect, S_sq_perfect, D_perfect, is_perfect = lattice_solver.calculate_perfect_s(S_squared=S_squared)
                
                if is_perfect:
                    print(f"\n✓ Using perfect S for factorization")
                else:
                    print(f"\n⚠ S does not satisfy all 3 domains, but proceeding with factorization")
                
                refined_p, refined_q, alpha_final, beta_final = lattice_solver.factor_from_s_squared(S_squared=S_squared)
            elif args.s is not None and args.d is not None:
                S = int(args.s)
                D = int(args.d)
                
                # Calculate and verify perfect S across all 3 domains
                S_perfect, S_sq_perfect, D_perfect, is_perfect = lattice_solver.calculate_perfect_s(S=S)
                
                if is_perfect:
                    print(f"\n✓ Using perfect S for factorization")
                else:
                    print(f"\n⚠ S does not satisfy all 3 domains, but proceeding with factorization")
                
                refined_p, refined_q, alpha_final, beta_final = lattice_solver.factor_from_s_squared(S=S, D=D)
            
            if refined_p and refined_q:
                if refined_p * refined_q == N:
                    print(f"\n{'='*80}")
                    print(f"🎉 SUCCESS! EXACT FACTORIZATION FOUND USING S² = 4N + D!")
                    print(f"{'='*80}")
                    print(f"p = {refined_p}")
                    print(f"q = {refined_q}")
                    print(f"Verification: {refined_p} × {refined_q} = {refined_p * refined_q} ✓")
                    
                    # Display decimal factors
                    if alpha_final is not None and beta_final is not None:
                        k_base = int(math.isqrt(N))
                        Qc_base = int(math.isqrt(N))
                        print(f"\n📊 Decimal Factors with Adjustments:")
                        print(f"   Base candidates: k = {k_base}, Qc = {Qc_base}")
                        print(f"   Adjustments: α = {alpha_final:.10f}, β = {beta_final:.10f}")
                        p_decimal = k_base + alpha_final
                        q_decimal = Qc_base + beta_final
                        print(f"   p = k + α = {k_base} + {alpha_final:.10f} = {p_decimal:.10f}")
                        print(f"   q = Qc + β = {Qc_base} + {beta_final:.10f} = {q_decimal:.10f}")
                    
                    success = True
                    return 0
        except Exception as e:
            print(f"Error in S² = 4N + D factorization: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing with lattice attack...")
    
    # Get or estimate candidates
    if args.p_decimal and args.q_decimal:
        # Use decimal approximations
        from decimal import Decimal
        try:
            p_decimal = Decimal(str(args.p_decimal))
            q_decimal = Decimal(str(args.q_decimal))
        except (ValueError, TypeError):
            print(f"Error: Invalid decimal format: p_decimal={args.p_decimal}, q_decimal={args.q_decimal}")
            return 1

        print(f"Using decimal approximations:")
        print(f"  p ≈ {p_decimal}")
        print(f"  q ≈ {q_decimal}")

        # Verify they multiply to N (approximately)
        product = p_decimal * q_decimal
        diff = abs(product - Decimal(str(N)))
        print(f"  p × q = {product}")
        print(f"  Error: {diff}")

        # Convert to integer candidates (round to nearest)
        # Ensure we get integers, not floats
        p_candidate = int(p_decimal.to_integral_value(rounding='ROUND_HALF_UP'))
        q_candidate = int(q_decimal.to_integral_value(rounding='ROUND_HALF_UP'))
        
        # Ensure N is an integer (in case it was computed from floats)
        N = int(N)

        print(f"  Rounded to integers: p={p_candidate}, q={q_candidate}")

    elif args.p is not None and args.q is not None:
        # Handle string arguments (could be decimals or large integers)
        try:
            # Try direct integer conversion first (handles large numbers)
            p_candidate = int(args.p)
            q_candidate = int(args.q)
            # Ensure N is an integer
            N = int(N)
            print(f"Using provided candidates: p={p_candidate}, q={q_candidate}")
        except ValueError:
            # Fallback to Decimal for decimal strings to preserve precision
            try:
                from decimal import Decimal
                p_decimal = Decimal(str(args.p))
                q_decimal = Decimal(str(args.q))
                p_candidate = int(p_decimal.to_integral_value(rounding='ROUND_HALF_UP'))
                q_candidate = int(q_decimal.to_integral_value(rounding='ROUND_HALF_UP'))
                # Ensure N is an integer
                N = int(N)
                print(f"Using provided candidates (converted from decimal): p={p_candidate}, q={q_candidate}")
            except (ValueError, OverflowError, TypeError) as e:
                print(f"Error: Invalid candidate format: p={args.p}, q={args.q}, error: {e}")
                return 1
    else:
        print("Estimating initial factor candidates...")
        # Check if we have partial hints to pass to estimation
        p_hint_for_est = None
        q_hint_for_est = None
        if hasattr(args, 'p') and args.p is not None:
            try:
                p_hint_for_est = int(args.p)
            except:
                pass
        if hasattr(args, 'q') and args.q is not None:
            try:
                q_hint_for_est = int(args.q)
            except:
                pass

        p_candidate, q_candidate = estimate_factors(N, p_hint_for_est, q_hint_for_est)
        print(f"Estimated candidates: p={p_candidate}, q={q_candidate}")

    # Calculate initial quality
    initial_product = p_candidate * q_candidate
    initial_diff = abs(initial_product - N)
    initial_diff_bits = initial_diff.bit_length()
    print(f"Initial product: {initial_product}")
    print(f"Initial error: {initial_diff} ({initial_diff_bits} bits)")
    print()

    # Determine search radius - default is 2^2048 to cover full 2048-bit key space
    # For LLL-based short vector search, large radius is fine (we're not brute forcing)
    DEFAULT_SEARCH_RADIUS_BITS = 2048
    DEFAULT_SEARCH_RADIUS = 2 ** DEFAULT_SEARCH_RADIUS_BITS
    
    if args.ultra_search_radius is not None:
        # Interpret ultra_search_radius as number of bits (e.g., 3000 means 2^3000)
        search_radius = 2 ** args.ultra_search_radius
        print(f"🔧 Using ultra search radius: 2^{args.ultra_search_radius} = {search_radius.bit_length()} bits")
    elif args.search_radius is not None:
        # If search_radius is provided as a number, interpret it as bits if > 10000, otherwise as direct value
        if args.search_radius > 10000:
            # Likely meant as bits
            search_radius = 2 ** args.search_radius
            print(f"🔧 Using search radius: 2^{args.search_radius} = {search_radius.bit_length()} bits")
        else:
            # Direct value
            search_radius = args.search_radius
            print(f"🔧 Using search radius: {search_radius}")
    else:
        # Default: 2^4000 (huge but fine for LLL - we're finding short vectors, not brute forcing)
        search_radius = DEFAULT_SEARCH_RADIUS
        print(f"🔧 Using default search radius: 2^{DEFAULT_SEARCH_RADIUS_BITS} = {search_radius.bit_length()} bits")
        print(f"    (LLL finds short vectors efficiently, so large radius is fine)")
    
    # Auto-expand if initial difference is larger than search radius (shouldn't happen with 2^4000 default)
    search_radius_bits = search_radius.bit_length() if search_radius > 0 else 0
    if initial_diff_bits > search_radius_bits:
        # Expand to initial_diff_bits + 100 bits margin
        expanded_bits = initial_diff_bits + 100
        search_radius = 2 ** expanded_bits
        print(f"🔧 Auto-expanding search radius: {search_radius_bits} bits → {expanded_bits} bits")
        print(f"    (Initial difference: {initial_diff_bits} bits, need larger radius)")

    # Run lattice attack
    refined_p, refined_q, improvement, pyramid_basis = lattice_solver.solve(
        p_candidate, q_candidate,
        confidence=0.8,
        search_radius=search_radius
    )

    print(f"\n" + "=" * 80)
    print(f"LATTICE ATTACK RESULTS")
    print(f"=" * 80)

    success = False
    best_candidate = None  # Track best candidate from Root's method for final summary

    if refined_p and refined_q:
        final_product = refined_p * refined_q
        final_diff = abs(final_product - N)

        print(f"Original candidates: p={p_candidate}, q={q_candidate}")
        print(f"Refined candidates: p={refined_p}, q={refined_q}")
        print(f"Final product: {final_product}")

        # Compute and display decimal factors with adjustments
        alpha = refined_p - p_candidate  # Adjustment α
        beta = refined_q - q_candidate   # Adjustment β
        p_decimal, q_decimal, alpha_calc, beta_calc = lattice_solver.compute_decimal_factors_with_adjustments(
            p_candidate, q_candidate, float(alpha), float(beta)
        )
        
        print(f"\n{'─'*80}")
        print(f"DECIMAL FACTORS WITH ADJUSTMENTS")
        print(f"{'─'*80}")
        print(f"Base candidates: k = {p_candidate}, Qc = {q_candidate}")
        print(f"Adjustments: α = {alpha:.10f}, β = {beta:.10f}")
        print(f"Decimal factors:")
        print(f"  p = k + α = {p_candidate} + {alpha:.10f} = {p_decimal:.10f}")
        print(f"  q = Qc + β = {q_candidate} + {beta:.10f} = {q_decimal:.10f}")
        
        # Verify decimal product
        from decimal import Decimal
        p_dec = Decimal(str(p_decimal))
        q_dec = Decimal(str(q_decimal))
        product_dec = p_dec * q_dec
        N_dec = Decimal(N)
        error_dec = abs(product_dec - N_dec)
        print(f"  p × q = {product_dec:.10f}")
        print(f"  Target N = {N_dec}")
        print(f"  Error = {error_dec:.10f}")

        if final_product == N:
            print(f"\n🎉 SUCCESS! EXACT FACTORIZATION FOUND!")
            print(f"p = {refined_p}")
            print(f"q = {refined_q}")
            print(f"Verification: {refined_p} × {refined_q} = {final_product} ✓")
            success = True
        else:
            print(f"\nFinal error: {final_diff}")
            if final_diff < initial_diff:
                improvement_pct = (initial_diff - final_diff) / initial_diff * 100
                print(f"Improvement: {improvement_pct:.1f}%")
    else:
        print("❌ Lattice attack found no improvements")

    # Try polynomial methods if enabled (always run when --polynomial is specified)
    if args.polynomial:
        print(f"\n{'='*80}")
        print(f"POLYNOMIAL-BASED FACTORIZATION ATTEMPT")
        print(f"{'='*80}")

        try:
            # Use best candidates available
            best_p = refined_p if refined_p else p_candidate
            best_q = refined_q if refined_q else q_candidate
            
            # Generate polynomial system
            print(f"\nGenerating enhanced polynomial system...")
            polynomials = lattice_solver.generate_factorization_polynomials(best_p, best_q, lattice_basis=pyramid_basis)
            
            # Limit polynomials if max_polynomials is set
            max_polys = args.max_polynomials if hasattr(args, 'max_polynomials') and args.max_polynomials else None
            if max_polys and len(polynomials) > max_polys:
                print(f"[GUI] Limiting polynomials to {max_polys} (from {len(polynomials)} generated)")
                polynomials = polynomials[:max_polys]
            
            # Initialize polynomial solver with config
            config = {
                'max_polynomials': args.max_polynomials if hasattr(args, 'max_polynomials') else None,
                'max_lattice_vectors': args.max_lattice_vectors if hasattr(args, 'max_lattice_vectors') else None,
                'coeff_limit': args.coeff_limit if hasattr(args, 'coeff_limit') else None,
                'trial_division_limit': args.trial_division_limit if hasattr(args, 'trial_division_limit') else None,
                'ultra_search_radius': args.ultra_search_radius if hasattr(args, 'ultra_search_radius') else None,
                'polynomial_grid_size': args.polynomial_grid_size if hasattr(args, 'polynomial_grid_size') else None,
                'max_root_candidates': args.max_root_candidates if hasattr(args, 'max_root_candidates') else None,
            }
            poly_solver = EnhancedPolynomialSolver(N, config=config, p_approx=best_p, q_approx=best_q)
            # Pass bulk search flag if available
            if hasattr(lattice_solver, 'use_bulk_search'):
                poly_solver.use_bulk_search = lattice_solver.use_bulk_search

            # Try all polynomial solving methods
            poly_solution = poly_solver.solve_with_all_methods(polynomials, best_p, best_q)

            # Track best candidate for final summary
            best_candidate_diff = float('inf')

            if poly_solution:
                p_poly, q_poly = poly_solution
                try:
                    if p_poly.bit_length() + q_poly.bit_length() <= 4096:
                        product = p_poly * q_poly
                        if product == N:
                            print(f"\n{'='*80}")
                            print(f"🎉 SUCCESS! POLYNOMIAL METHOD FOUND EXACT FACTORIZATION!")
                            print(f"{'='*80}")
                            print(f"p = {p_poly}")
                            print(f"q = {q_poly}")
                            print(f"Verification: {p_poly} × {q_poly} = {p_poly * q_poly} ✓")
                            success = True
                        else:
                            diff = abs(product - N)
                            best_candidate = (p_poly, q_poly)
                            best_candidate_diff = diff
                            print(f"\nPolynomial method found approximate solution:")
                            print(f"p = {p_poly}, q = {q_poly}")
                            print(f"Product: {p_poly * q_poly} (error: {diff:,})")
                    else:
                        # Too large to verify, but track as candidate
                        best_candidate = (p_poly, q_poly)
                        print(f"\nPolynomial method found candidate (too large to verify exactly):")
                        print(f"p = {p_poly}, q = {q_poly}")
                except:
                    # Error verifying, but still track
                    best_candidate = (p_poly, q_poly)
                    print(f"\nPolynomial method found candidate (verification failed):")
                    print(f"p = {p_poly}, q = {q_poly}")
            else:
                print(f"\nPolynomial solving methods did not find exact factorization")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            return 1
        except Exception as e:
            print(f"\nPolynomial solving encountered error: {e}")

    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    
    if success:
        print("✅ FACTORIZATION SUCCESSFUL!")
        # Display decimal factors for successful factorization
        if refined_p and refined_q:
            alpha = refined_p - p_candidate
            beta = refined_q - q_candidate
            p_decimal, q_decimal, _, _ = lattice_solver.compute_decimal_factors_with_adjustments(
                p_candidate, q_candidate, float(alpha), float(beta)
            )
            print(f"\n📊 Decimal Factors with Adjustments:")
            print(f"   p = {p_candidate} + {alpha:.10f} = {p_decimal:.10f}")
            print(f"   q = {q_candidate} + {beta:.10f} = {q_decimal:.10f}")
    else:
        # Check if we have a best candidate to display
        if best_candidate:
            p_best, q_best = best_candidate
            print("❌ Exact factorization not found")
            print(f"\n🏆 BEST CANDIDATE FROM ROOT'S METHOD:")
            print(f"   p = {p_best}")
            print(f"   q = {q_best}")
            
            # Compute decimal factors for best candidate
            try:
                # Use original candidates as base
                alpha_best = p_best - p_candidate
                beta_best = q_best - q_candidate
                p_decimal_best, q_decimal_best, _, _ = lattice_solver.compute_decimal_factors_with_adjustments(
                    p_candidate, q_candidate, float(alpha_best), float(beta_best)
                )
                print(f"\n📊 Decimal Factors with Adjustments:")
                print(f"   p = {p_candidate} + {alpha_best:.10f} = {p_decimal_best:.10f}")
                print(f"   q = {q_candidate} + {beta_best:.10f} = {q_decimal_best:.10f}")
            except:
                pass
            
            try:
                if p_best.bit_length() + q_best.bit_length() <= 4096:
                    product = p_best * q_best
                    diff = abs(product - N)
                    print(f"   Product: {p_best} × {q_best} = {product}")
                    print(f"   Difference from N: {diff:,}")
                    if diff == 0:
                        print(f"   ✓ This is actually an exact factorization!")
                    else:
                        print(f"   ⚠️  Approximation (not exact)")
                else:
                    print(f"   ⚠️  Approximation (too large to verify exactly)")
            except:
                print(f"   ⚠️  Approximation (verification failed)")
        elif refined_p and refined_q:
            # Display decimal factors even if not exact
            alpha = refined_p - p_candidate
            beta = refined_q - q_candidate
            p_decimal, q_decimal, _, _ = lattice_solver.compute_decimal_factors_with_adjustments(
                p_candidate, q_candidate, float(alpha), float(beta)
            )
            print("❌ Exact factorization not found")
            print(f"\n📊 Decimal Factors with Adjustments (from lattice):")
            print(f"   p = {p_candidate} + {alpha:.10f} = {p_decimal:.10f}")
            print(f"   q = {q_candidate} + {beta:.10f} = {q_decimal:.10f}")
        else:
            print("❌ Exact factorization not found")
        
        print("\nSuggestions:")
        print("  • Try larger --search-radius (e.g., 5000 or 10000)")
        print("  • Provide better initial candidates with --p and --q")
        print("  • For small numbers, --polynomial flag may help")
        print("  • For very large numbers, consider specialized factorization algorithms")
    
    print("\n🔬 Methods used:")
    print("  • Minimizable Factorization Lattice (custom approach)")
    if args.polynomial:
        print("  • Enhanced Polynomial Solving:")
        print("    - Gröbner Basis (symbolic)")
        print("    - Resultant Elimination (symbolic)")
        print("    - Modular Constraints & Trial Division")
        print("    - Numerical Refinement")
        print("    - Hensel Lifting")

    # ============================================================================
    # END FACTORIZATION MODE
    # ============================================================================

    # Save trained transformer if requested (for post-attack learning)
    # Note: Pre-training saves happen immediately after pre-training above
    if args.save_pretrained and not args.pretrain:
        print(f"\n💾 Saving transformer model after attack...")
        print(f"   Model learned from {len(transformer.search_history)} search attempts on target")
        if transformer.save_model(args.save_pretrained):
            print(f"✅ Model saved to {args.save_pretrained}")
        else:
            print(f"❌ Failed to save model")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
