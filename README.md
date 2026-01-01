# LuckyLattice 
**Advanced Integer Factorization Tool using Lattice Reduction and Polynomial Methods**

**Usage**
```
**python3 standalone_lattice_attack.py** TargetNumber **--lattice-dimension** 114813069527425452423283320117768198402231770208869520047764273682576626139237031385665948631650626991844596463898746277344711896086305533142593135616665318539129989145312280000688779148240044871428926990063486244781615463646388363947317026040466353970904996558162398808944629605623311649536164221970332681344168908984458505602379484807914058900934776500429002716706625830522008132236281291761267883317206598995396418127021779858404042159853183251540889433902091920554957783589672039160081957216630582755380425583726015528348786419432054508915275783882625175435528800822842770817965453762184851149029376 **--polynomial --max-root-combinations** 20 **--search-radius** 10 **--verbose**
```
**Note:** Lattice Dimension is set to be really, really, really high but it will still run very well.

LuckyLattice is a powerful factorization tool that combines lattice-based attacks (LLL reduction) with advanced polynomial solving techniques (Gröbner bases, resultants, Hensel lifting) to factor large integers. It's particularly effective when you have approximate knowledge of the factors.

## Features ✨

- **Lattice-Based Factorization**: Custom minimizable factorization lattice solver using LLL reduction
- **Integer-Based LLL**: Handles arbitrarily large integers without float conversion errors (supports 2048+ bit numbers)
- **Polynomial Methods**: Multiple polynomial solving techniques:
  - Gröbner Basis elimination
  - Resultant elimination
  - Hensel lifting
  - Modular constraints & trial division
- **Approximate Factor Support**: Works with approximate `p` and `q` values when exact factors are unknown
- **Flexible Search**: Configurable search radius (up to 2^2000 for huge coefficients)
- **Automatic S/D Discovery**: Automatically finds sum and difference of factors using Root's Method
- **Extensive Configuration**: Fine-tune all parameters for optimal performance

## Installation 📦

### Requirements

- Python 3.7+
- NumPy
- SymPy

### Install Dependencies

```bash
pip install numpy sympy
```

### Optional Dependencies

- `fpylll`: For faster LLL reduction (optional, custom integer-based implementation included)

## Usage 🚀

### Basic Usage

```bash
python standalone_lattice_attack.py <N>
```

### With Approximate Factors

```bash
python standalone_lattice_attack.py <N> --p <p_approx> --q <q_approx>
```

### With Polynomial Methods

```bash
python standalone_lattice_attack.py <N> --polynomial
```

### Advanced Example

```bash
python standalone_lattice_attack.py 2021 --polynomial --verbose --search-radius 5000
```

## Command-Line Options 📋

### Required Arguments

- `N`: The number to factor (integer)

### Optional Arguments

#### Factor Hints
- `--p P`: Initial P candidate (integer or decimal)
- `--q Q`: Initial Q candidate (integer or decimal)
- `--p-decimal P_DECIMAL`: P as decimal approximation
- `--q-decimal Q_DECIMAL`: Q as decimal approximation
- `--s S`: S value where S = p + q (sum of factors)
- `--d D`: D value where D = (p - q)² (square of difference)
- `--s-squared S_SQUARED`: S² value where S² = 4N + D

#### Search Parameters
- `--search-radius SEARCH_RADIUS`: Search radius for corrections (default: 1000)
- `--ultra-search-radius ULTRA_SEARCH_RADius`: Ultra-high search radius for huge coefficients
- `--lattice-dimension LATTICE_DIMENSION`: Lattice dimension parameter

#### Method Selection
- `--polynomial`: Enable polynomial solving methods
- `--auto-find-s-d`: Automatically find S and D using Root's Method (default: enabled)
- `--no-auto-find-s-d`: Disable automatic S and D discovery

#### Polynomial Options
- `--max-polynomials MAX_POLYNOMIALS`: Maximum number of polynomials to analyze
- `--coeff-limit COEFF_LIMIT`: Coefficient size limit
- `--trial-division-limit TRIAL_DIVISION_LIMIT`: Limit for trial division
- `--polynomial-grid-size POLYNOMIAL_GRID_SIZE`: Grid size for polynomial search
- `--max-root-candidates MAX_ROOT_CANDIDATES`: Maximum root candidates to check
- `--root-sampling-strategy {none,random,stratified,adaptive}`: Strategy for root sampling
- `--early-termination`: Stop after finding first solution

#### Output
- `--verbose`: Enable verbose output

## Examples 📚

### Example 1: Factor Small Number

```bash
python standalone_lattice_attack.py 2021 --polynomial --verbose
```

Output:
```
🎉 SUCCESS! POLYNOMIAL METHOD FOUND EXACT FACTORIZATION!
p = 43
q = 47
Verification: 43 × 47 = 2021 ✓
```

### Example 2: Factor with Approximate Hints

```bash
python standalone_lattice_attack.py <LARGE_N> \
    --p <p_approx> \
    --q <q_approx> \
    --polynomial \
    --search-radius 10000
```

### Example 3: Very Large Number (2048+ bits)

```bash
python standalone_lattice_attack.py <HUGE_N> \
    --p <p_approx> \
    --q <q_approx> \
    --polynomial \
    --ultra-search-radius 2000000000 \
    --verbose
```

## How It Works 🔬

### 1. Lattice-Based Method

The `MinimizableFactorizationLatticeSolver` constructs a pyramid-shaped lattice basis representing factorization relations. It uses LLL reduction to find short vectors that correspond to optimal corrections to approximate factors.

**Key Features:**
- Pyramid lattice structure for large numbers
- Integer-based LLL reduction (no float overflow)
- Configurable search radius up to 2^2000

### 2. Polynomial Methods

The `EnhancedPolynomialSolver` uses multiple algebraic techniques:

#### Gröbner Basis Elimination
- Uses lexicographic ordering to eliminate variables
- Constructs univariate polynomials in target variables
- Solves for integer roots directly

#### Resultant Elimination
- Optimized for ABCD fused polynomials
- Efficiently eliminates variables through resultants

#### Hensel Lifting
- Lifts modular solutions to full integers
- Works with modular constraints

#### Modular Constraints & Trial Division
- Uses modular arithmetic to constrain solutions
- Efficient trial division for small factors

### 3. Integer-Based LLL

The `fpylll_wrapper.py` module provides an integer-based LLL implementation that:
- Uses Python's arbitrary-precision integers
- Avoids float conversion errors for large numbers
- Provides fpylll-compatible interface
- Handles numbers with 2048+ bits

## Architecture 🏗️

```
LuckyLattice/
├── standalone_lattice_attack.py    # Main script with all solvers
├── fpylll_wrapper.py               # Integer-based LLL implementation
└── README.md                        # This file
```

### Main Classes

- **`MinimizableFactorizationLatticeSolver`**: Lattice-based factorization using LLL reduction
- **`EnhancedPolynomialSolver`**: Polynomial solving methods (Gröbner, resultants, Hensel)
- **`IntegerMatrix`**: Matrix class for large integers
- **`LLL`**: Integer-based LLL reduction function

## Performance Considerations ⚡

- **Small numbers (< 100 bits)**: Polynomial methods are fast and efficient
- **Medium numbers (100-1000 bits)**: Lattice method works well with good approximations
- **Large numbers (1000+ bits)**: Requires approximate factors and larger search radius
- **Very large numbers (2048+ bits)**: Use `--ultra-search-radius` and integer-based LLL

### Tips for Best Performance

1. Provide the best possible approximate `p` and `q` values
2. Use `--polynomial` flag for numbers < 500 bits
3. Increase `--search-radius` for larger numbers
4. Use `--verbose` to monitor progress
5. For huge numbers, consider using `--ultra-search-radius`

## Limitations ⚠️

- Factorization of cryptographically secure RSA numbers (without side channels) is computationally infeasible
- Very large numbers require good approximate factors to succeed
- Running time increases significantly with number size and search radius
- Some methods may not find solutions even if they exist (NP-hard problem)

## Use Cases 🎯

- **Educational purposes**: Understanding factorization algorithms
- **CTF challenges**: When approximate factors are known
- **Research**: Testing lattice-based factorization methods
- **Side-channel analysis**: When partial information about factors is available
- **Weak key analysis**: Finding factors when keys were generated poorly

## Contributing 🤝

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License 📄

MIT

## Acknowledgments 🙏

- Uses SymPy for symbolic mathematics
- Uses NumPy for numerical computations
- Implements LLL algorithm (Lenstra–Lenstra–Lovász)
- Gröbner basis computation inspired by SageMath

## References 📖

- LLL Algorithm: Lenstra, A. K., Lenstra, H. W., & Lovász, L. (1982). Factoring polynomials with rational coefficients.
- Gröbner Bases: Cox, D., Little, J., & O'Shea, D. (2015). Ideals, Varieties, and Algorithms.
- Factorization: Crandall, R., & Pomerance, C. (2005). Prime Numbers: A Computational Perspective.

### Comparison with Coppersmith's Method

LuckyLattice differs significantly from Coppersmith's attack in several key ways:

| Aspect | Coppersmith's Method | LuckyLattice |
|--------|---------------------|--------------|
| **Domain** | Works modulo N (or divisors of N) | Works over integers (no modular constraint) |
| **Root Size** | Requires roots < N^(1/d) where d is polynomial degree | Can find larger corrections with approximate factors |
| **Lattice Structure** | Specific construction for small roots mod N | Pyramid-shaped lattice for factorization relations |
| **Input Requirements** | Partial information (e.g., MSB/LSB of factors) | Approximate factors (p_approx, q_approx) |
| **Polynomial Approach** | Direct LLL on constructed lattice | Gröbner basis elimination + direct solving |
| **Method** | Finds small integer solutions to f(x) ≡ 0 (mod N) | Solves Diophantine equation f(x,v) = 0 over integers |
| **Use Case** | RSA with partial key exposure | Factorization with approximate knowledge |

**Key Differences:**

1. **Modular vs. Integer Arithmetic**
   - **Coppersmith**: Finds solutions to f(x) ≡ 0 (mod N), requiring roots to be small relative to N
   - **LuckyLattice**: Solves f(x,v) = 0 over integers, allowing larger corrections when approximate factors are known

2. **Lattice Construction**
   - **Coppersmith**: Constructs a specific lattice from polynomial powers (x, x², ..., x^m) and N powers
   - **LuckyLattice**: Uses a pyramid-shaped lattice representing factorization relations (p, q, p*q-N, p+q, p-q, etc.)

3. **Polynomial Solving**
   - **Coppersmith**: Uses LLL to find short vectors that correspond to small roots
   - **LuckyLattice**: Uses Gröbner basis elimination to obtain univariate polynomials, then solves directly

4. **Correction Size**
   - **Coppersmith**: Limited by N^(1/d) bound - roots must be very small
   - **LuckyLattice**: Can handle corrections up to 2^2000 with good approximations

5. **When Each Works Best**
   - **Coppersmith**: Best when you know partial bits (e.g., half the bits of p or q)
   - **LuckyLattice**: Best when you have approximate values (e.g., p ≈ p_approx with small error)

**Example Scenario:**

For an RSA modulus N = p·q:
- **Coppersmith**: If you know the top 50% of bits of p, it can recover the rest
- **LuckyLattice**: If you know p ≈ p_approx (within search radius), it can find exact p and q

### 1. Lattice-Based Method

The `MinimizableFactorizationLatticeSolver` constructs a pyramid-shaped lattice basis representing factorization relations. It uses LLL reduction to find short vectors that correspond to optimal corrections to approximate factors.

**Key Features:**
- Pyramid lattice structure for large numbers
- Integer-based LLL reduction (no float overflow)
- Configurable search radius up to 2^2000

### 2. Polynomial Methods

The `EnhancedPolynomialSolver` uses multiple algebraic techniques:

#### Gröbner Basis Elimination
- Uses lexicographic ordering to eliminate variables
- Constructs univariate polynomials in target variables
- Solves for integer roots directly

#### Resultant Elimination
- Optimized for ABCD fused polynomials
- Efficiently eliminates variables through resultants

#### Hensel Lifting
- Lifts modular solutions to full integers
- Works with modular constraints

#### Modular Constraints & Trial Division
- Uses modular arithmetic to constrain solutions
- Efficient trial division for small factors

### 3. Integer-Based LLL

The `fpylll_wrapper.py` module provides an integer-based LLL implementation that:
- Uses Python's arbitrary-precision integers
- Avoids float conversion errors for large numbers
- Provides fpylll-compatible interface
- Handles numbers with 2048+ bits

## Architecture 🏗️

```
LuckyLattice/
├── standalone_lattice_attack.py    # Main script with all solvers
├── fpylll_wrapper.py               # Integer-based LLL implementation
└── README.md                        # This file
```

### Main Classes

- **`MinimizableFactorizationLatticeSolver`**: Lattice-based factorization using LLL reduction
- **`EnhancedPolynomialSolver`**: Polynomial solving methods (Gröbner, resultants, Hensel)
- **`IntegerMatrix`**: Matrix class for large integers
- **`LLL`**: Integer-based LLL reduction function

## Performance Considerations ⚡

- **Small numbers (< 100 bits)**: Polynomial methods are fast and efficient
- **Medium numbers (100-1000 bits)**: Lattice method works well with good approximations
- **Large numbers (1000+ bits)**: Requires approximate factors and larger search radius
- **Very large numbers (2048+ bits)**: Use `--ultra-search-radius` and integer-based LLL

### Tips for Best Performance

1. Provide the best possible approximate `p` and `q` values
2. Use `--polynomial` flag for numbers < 500 bits
3. Increase `--search-radius` for larger numbers
4. Use `--verbose` to monitor progress
5. For huge numbers, consider using `--ultra-search-radius`

---

**Note**: This tool is for educational and research purposes. Do not use for illegal activities.

