# SPDX-FileCopyrightText: © 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from golden_reference import pe_compute, generate_test_vectors, CORNER_CASES

async def reset_dut(dut):
    """Apply reset sequence"""
    dut.rst_n.value = 0
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

async def load_weight(dut, weight):
    """Load a weight into the weight register"""
    dut.ui_in.value = weight & 0xFF   # mask to 8 bits for unsigned wire
    dut.uio_in.value = 0b00000001     # load_weight = 1
    await RisingEdge(dut.clk)
    dut.uio_in.value = 0b00000000     # release
    await RisingEdge(dut.clk)

async def load_bias(dut, bias):
    """Load a bias into the bias register"""
    dut.ui_in.value = bias & 0xFF
    dut.uio_in.value = 0b00000010     # load_bias = 1
    await RisingEdge(dut.clk)
    dut.uio_in.value = 0b00000000
    await RisingEdge(dut.clk)

async def reset_accumulator(dut):
    """Reset the accumulator"""
    dut.uio_in.value = 0b00001000     # reset_acc = 1
    await RisingEdge(dut.clk)
    dut.uio_in.value = 0b00000000
    await RisingEdge(dut.clk)

async def compute(dut, activation):
    """Feed one activation and wait for output (3 cycle pipeline)"""
    dut.ui_in.value  = activation & 0xFF
    dut.uio_in.value = 0b00000100     # accumulate = 1
    await RisingEdge(dut.clk)
    dut.uio_in.value = 0b00000000
    # Wait for 3-cycle pipeline
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    # Read output (convert unsigned cocotb value back to signed int8)
    raw = int(dut.uo_out.value)
    return raw if raw < 128 else raw - 256

@cocotb.test()
async def test_reset(dut):
    """Verify output is zero after reset"""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)
    assert int(dut.uo_out.value) == 0, "Output should be 0 after reset"
    assert int(dut.uio_out.value) & 0x10 == 0, "Valid flag should be 0 after reset"
    cocotb.log.info("PASS: reset test")

@cocotb.test()
async def test_corner_cases(dut):
    """Verify all corner cases against golden reference"""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    
    for weight, activation, bias, description in CORNER_CASES:
        await reset_dut(dut)
        await reset_accumulator(dut)
        await load_weight(dut, weight)
        await load_bias(dut, bias)
        
        actual = await compute(dut, activation)
        expected = pe_compute(weight, activation, bias)['output']
        
        assert actual == expected, (
            f"FAIL: {description}\n"
            f"  weight={weight}, activation={activation}, bias={bias}\n"
            f"  Expected: {expected}, Got: {actual}"
        )
        cocotb.log.info(f"PASS: {description}: output={actual}")

@cocotb.test()
async def test_random_vectors(dut):
    """Verify 1000 random test vectors against golden reference"""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    vectors = generate_test_vectors(1000, seed=42)
    failures = []
    
    for i, v in enumerate(vectors):
        await reset_dut(dut)
        await reset_accumulator(dut)
        await load_weight(dut, v['weight'])
        await load_bias(dut, v['bias'])
        
        actual = await compute(dut, v['activation'])
        expected = v['expected_output']
        
        if actual != expected:
            failures.append({
                'index': i,
                'weight': v['weight'],
                'activation': v['activation'],
                'bias': v['bias'],
                'expected': expected,
                'actual': actual
            })
    
    if failures:
        msg = f"{len(failures)} failures:\n"
        for f in failures[:5]:  # show first 5
            msg += (f"  [{f['index']}] w={f['weight']} a={f['activation']} "
                   f"b={f['bias']}: expected {f['expected']}, got {f['actual']}\n")
        assert False, msg
    
    cocotb.log.info(f"PASS: all 1000 random vectors match golden reference")

@cocotb.test()
async def test_weight_stationary(dut):
    """
    Verify weight-stationary behavior:
    load weight once, run multiple activations,
    results should match golden reference for each
    """
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)
    
    weight = 42
    bias = 10
    activations = [10, 20, -15, 100, -128, 0, 64, -64]
    
    await load_weight(dut, weight)
    await load_bias(dut, bias)
    
    for act in activations:
        await reset_accumulator(dut)
        actual = await compute(dut, act)
        expected = pe_compute(weight, act, bias)['output']
        assert actual == expected, (
            f"Weight-stationary fail: w={weight} a={act} b={bias}: "
            f"expected {expected}, got {actual}"
        )
        cocotb.log.info(
            f"PASS: weight-stationary: w={weight} a={act} b={bias} → {actual}"
        )

@cocotb.test()
async def test_valid_flag_timing(dut):
    """Verify output_valid flag appears exactly 3 cycles after accumulate"""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)
    await load_weight(dut, 1)
    await load_bias(dut, 0)
    
    # Assert accumulate
    dut.ui_in.value  = 1
    dut.uio_in.value = 0b00000100
    await RisingEdge(dut.clk)
    dut.uio_in.value = 0b00000000
    
    # Cycle 1 after accumulate
    await RisingEdge(dut.clk)
    valid_cycle1 = int(dut.uio_out.value) & 0x10
    
    # Cycle 2
    await RisingEdge(dut.clk)
    valid_cycle2 = int(dut.uio_out.value) & 0x10
    
    # Cycle 3 — valid should appear here
    await RisingEdge(dut.clk)
    valid_cycle3 = int(dut.uio_out.value) & 0x10
    
    assert valid_cycle1 == 0, "Valid flag should not appear at cycle 1"
    assert valid_cycle2 == 0, "Valid flag should not appear at cycle 2"
    assert valid_cycle3 != 0, "Valid flag should appear at cycle 3"
    cocotb.log.info("PASS: valid flag timing correct (3 cycle latency confirmed)")

'''
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles


@cocotb.test()
async def test_project(dut):
    dut._log.info("Start")

    # Set the clock period to 10 us (100 KHz)
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())

    # Reset
    dut._log.info("Reset")
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    dut._log.info("Test project behavior")

    # Set the input values you want to test
    dut.ui_in.value = 20
    dut.uio_in.value = 30

    # Wait for one clock cycle to see the output values
    await ClockCycles(dut.clk, 1)

    # The following assersion is just an example of how to check the output values.
    # Change it to match the actual expected output of your module:
    assert dut.uo_out.value == 50

    # Keep testing the module by changing the input values, waiting for
    # one or more clock cycles, and asserting the expected output values.
