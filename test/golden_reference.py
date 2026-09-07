def to_int8(value):
    """Saturate and cast to int8 range [-128, 127]"""
    if value < -128:
        return -128
    if value > 127:
        return 127
    return int(value)

def to_int16(value):
    """Cast to int16 range for accumulator [-32768, 32767]"""
    if value < -32768:
        return -32768
    if value > 32767:
        return 32767
    return int(value)

def leaky_relu_int8(value, beta_shift=2):
    """
    Leaky ReLU with beta = 0.25 (arithmetic right shift by 2)
    For positive values: pass through unchanged
    For negative values: arithmetic right shift by beta_shift positions
    beta_shift=2 gives beta=0.25, beta_shift=1 gives beta=0.5
    """
    if value >= 0:
        return value
    else:
        # Arithmetic right shift (preserves sign in Python)
        return value >> beta_shift

def pe_compute(weight, activation, bias, beta_shift=2):
    """
    Complete PE pipeline:
    Stage 1: MAC (multiply-accumulate)
    Stage 2: Bias add
    Stage 3: Leaky ReLU
    Stage 4: Output saturation to int8
    
    All inputs are int8 signed integers [-128, 127]
    Accumulator is int16 to handle overflow
    """
    # Input validation
    assert -128 <= weight <= 127, f"Weight {weight} out of int8 range"
    assert -128 <= activation <= 127, f"Activation {activation} out of int8 range"
    assert -128 <= bias <= 127, f"Bias {bias} out of int8 range"
    
    # Stage 1: Multiply-accumulate
    # int8 * int8 = up to 16-bit result
    mac_result = int(weight) * int(activation)
    mac_result = to_int16(mac_result)
    
    # Stage 2: Bias add (in int16 space)
    biased = to_int16(mac_result + int(bias))
    
    # Stage 3: Leaky ReLU (in int16 space)
    activated = leaky_relu_int8(biased, beta_shift)
    
    # Stage 4: Saturate output to int8
    output = to_int8(activated)
    
    return {
        'mac_result': mac_result,
        'biased': biased,
        'activated': activated,
        'output': output
    }

def generate_test_vectors(n=1000, seed=42):
    """Generate random test vectors using a zero-import Linear Congruential Generator"""
    # LCG constants (Numerical Recipes parameters)
    state = seed
    a = 1664525
    c = 1013904223
    m = 2**32
    
    vectors = []
    for _ in range(n):
        vals = []
        for _ in range(3):  # We need 3 random values per iteration
            state = (a * state + c) % m
            # Scale 32-bit unsigned int to signed int8 [-128, 127]
            val = (state % 256) - 128
            vals.append(val)
            
        weight, activation, bias = vals
        result = pe_compute(weight, activation, bias)
        vectors.append({
            'weight': weight,
            'activation': activation,
            'bias': bias,
            'expected_output': result['output']
        })
    return vectors

# Corner cases — these MUST all pass
CORNER_CASES = [
    # (weight, activation, bias, description)
    (0,    0,    0,    "all zeros"),
    (127,  127,  127,  "max positive — tests saturation"),
    (-128, -128, -128, "max negative — tests saturation"),
    (127,  -128, 0,    "max negative product"),
    (-128, 127,  0,    "max negative product reversed"),
    (1,    1,    0,    "minimal positive"),
    (-1,   1,    0,    "minimal negative output before ReLU"),
    (0,    127,  0,    "zero weight"),
    (127,  0,    0,    "zero activation"),
    (-1,   -1,   0,    "negative*negative=positive"),
    (64,   64,   0,    "overflow test"),
    (-64,  64,   0,    "negative overflow test"),
    (0,    0,    127,  "bias only positive"),
    (0,    0,    -128, "bias only negative — tests ReLU on bias"),
]

if __name__ == "__main__":
    print("Corner case verification:")
    print(f"{'Weight':>8} {'Activ':>8} {'Bias':>6} {'MAC':>8} "
          f"{'Biased':>8} {'Post-ReLU':>10} {'Output':>8} Description")
    print("-" * 80)
    for w, a, b, desc in CORNER_CASES:
        r = pe_compute(w, a, b)
        print(f"{w:>8} {a:>8} {b:>6} {r['mac_result']:>8} "
              f"{r['biased']:>8} {r['activated']:>10} "
              f"{r['output']:>8}  {desc}")
    
    print(f"\nGenerating {1000} random test vectors...")
    vectors = generate_test_vectors(1000)
    print(f"Generated {len(vectors)} test vectors successfully")
    print("Golden reference model complete.")
