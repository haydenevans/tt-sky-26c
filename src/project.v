/*
 * Copyright (c) 2026 Hayden Evans
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none // Treat undeclared signals as fatal errors instead of one-bit wires
`timescale 1ns/1ps // Set simulation resolution - 1ns time unit, 1ps precision

module tt_um_haydenevans_top (
    input  wire [7:0] ui_in,    // Dedicated inputs - used to deliver weight values, bias values, and activation values at different times
    output wire [7:0] uo_out,   // Dedicated outputs - 8-bit signed output (computation result)
    input  wire [7:0] uio_in,   // IOs: Input path - Bidirectional inputs used as control inputs (load_weight, load_bias, accumulate, reset_acc)
    output wire [7:0] uio_out,  // IOs: Output path - Bidirection outputs used as status outputs (output_valid, overflow, two spare bits)
    output wire [7:0] uio_oe,   // IOs: Enable path (active high: 0=input, 1=output)
    input  wire       ena,      // always 1 when the design is powered, so you can ignore it
    input  wire       clk,      // clock
    input  wire       rst_n     // reset_n - low to reset
);
    
    // Pin Map:
    // ui_in[7:0]  : shared data bus (time-multiplexed: weight / bias / activation)
    // uo_out[7:0] : int8 saturated output after Leaky ReLU (beta = 0.25)
    // uio[0]      : load_weight  (input) — strobe high one cycle to load weight
    // uio[1]      : load_bias    (input) — strobe high one cycle to load bias
    // uio[2]      : accumulate   (input) — strobe high one cycle to compute MAC
    // uio[3]      : reset_acc    (input) — strobe high one cycle to clear accumulator
    // uio[4]      : output_valid (output) — high exactly 3 cycles after accumulate
    // uio[5]      : overflow     (output) — high when uo_out was saturated (clamped)
    // uio[6]      : spare        (output, driven 0)
    // uio[7]      : spare        (output, driven 0)

    wire load_weight = uio_in[0];
    wire load_bias   = uio_in[1];
    wire accumulate  = uio_in[2];
    wire reset_acc   = uio_in[3];
    
    // Usage protocol:
    // 1. Assert load_weight (uio[0]), place weight on ui_in[7:0], clock once, release
    // 2. Assert load_bias, place bias on ui_in, clock once, release
    // 3. Assert reset_acc, clock once, release
    // 4. Assert accumulate, place activation on ui_in, clock once, release
    // 5. After 3 more clock cycles: read uo_out when output_valid is high

    reg signed [7:0] weight_reg; // Holds stationary weight
    reg signed [7:0] bias_reg; // Holds bias term

    always @(posedge clk) begin
        if (!rst_n) begin
            weight_reg <= 8'h00;
            bias_reg <= 8'h00;
        end else begin
            if (load_weight) weight_reg <= $signed(ui_in);
            if (load_bias) bias_reg <= $signed(ui_in);
        end
    end

    // Stage 1 - Multiply-Accumulate (MAC)
    reg signed [15:0] accumulator; // Holds running sum
    reg signed [15:0] stage1_mac; // Snapshot of accumulator result flowing forward to Stage 2
    reg stage1_valid; // Tracks whether data in stage1_mac is a valid computation result

    wire signed [15:0] product = $signed(weight_reg) * $signed(ui_in);
    wire signed [16:0] acc_next = $signed({accumulator[15], accumulator}) + $signed({product[15], product});

    always @(posedge clk) begin
        if (!rst_n || reset_acc) begin
            accumulator  <= 16'h0000; // Reset accumulator
            stage1_mac   <= 16'h0000; // Reset stage1_mac
            stage1_valid <= 1'b0; // Reset stage1_valid
        end else if (accumulate) begin
            if (acc_next > 17'sh07FFF) begin //Positive int16 saturation
                accumulator <= 16'h7FFF; // Clamp accumulator
                stage1_mac  <= 16'h7FFF; // Clamp stage1_mac
            end else if (acc_next < -17'sh08000) begin // Negative int16 saturation
                accumulator <= 16'h8000; // Clamp accumulator
                stage1_mac  <= 16'h8000; // Clamp stage1_mac
            end else begin
                accumulator <= acc_next[15:0];
                stage1_mac  <= acc_next[15:0];
            end
            stage1_valid <= 1'b1;
        end else begin
            stage1_valid <= 1'b0;
        end
    end

    // Stage 2 - Bias Addition
    reg signed [15:0] stage2_biased; // Snapshot of biased result flowing forward to Stage 3
    reg stage2_valid; // Tracks whether data in stage2_biased is a valid computation result

    wire signed [16:0] biased_next = $signed({stage1_mac[15], stage1_mac}) + $signed({{9{bias_reg[7]}}, bias_reg}); //Add sign extended bias to stage1_mac value from Stage 1

    always @(posedge clk) begin
        if (!rst_n) begin
            stage2_biased <= 16'h0000; // Reset stage2_biased
            stage2_valid  <= 1'b0; // Reset stage2_valid
        end else begin
            stage2_valid <= stage1_valid;
            if (stage1_valid) begin
                if (biased_next > 17'sh07FFF) // Positive int16 saturation
                    stage2_biased <= 16'h7FFF; // Clamp stage2_biased
                else if (biased_next < -17'sh08000) // Negative int16 saturation
                    stage2_biased <= 16'h8000; // Clamp stage2_biased
                else
                    stage2_biased <= biased_next[15:0];
            end
        end
    end

    //Stage 3 - Leaky ReLU and int8 Saturation
    reg signed [7:0] stage3_output; // Stores output from Stage 3
    reg stage3_valid; // Tracks whether stage3_output data is valid
    reg overflow_out; // Overflow flag thrown if Leaky ReLU result lies outside of int8 saturation

    wire signed [15:0] relu_result = stage2_biased[15] ? (stage2_biased >>> 2) : stage2_biased; // Test to see if stage2_biased is negative, if so, apply Leaky ReLU with arithmetic right shift of 2 (*0.25)

    always @(posedge clk) begin
        if (!rst_n) begin
            stage3_output <= 8'h00; // Reset stage3_output
            stage3_valid  <= 1'b0; // Reset stage3_valid
            overflow_out  <= 1'b0; // Reset overflow_out
        end else begin
            stage3_valid <= stage2_valid;
            if (stage2_valid) begin
                if (relu_result > 16'sh007F) begin // Positive int8 saturation
                    stage3_output <= 8'h7F; // Clamp stage3_output
                    overflow_out  <= 1'b1; // Throw overflow flag
                end else if (relu_result < 16'shFF80) begin // Negative int8 saturation
                    stage3_output <= 8'h80; // Clamp stage3_output
                    overflow_out  <= 1'b1; // Throw overflow flag
                end else begin
                    stage3_output <= relu_result[7:0];
                    overflow_out  <= 1'b0;
                end
            end else begin
                overflow_out <= 1'b0;
            end
        end
    end

    // All output pins must be assigned. If not used, assign to 0.
    assign uo_out  = stage3_output; // Current value of stage3_output
    assign uio_out = {2'b00, overflow_out, stage3_valid, 4'b0000};
    assign uio_oe  = 8'b11110000; // Enable uio[7:4] as outputs by asserting active high

    // List all unused inputs to prevent warnings
    wire _unused = &{ena, uio_in[7:4], 1'b0};

endmodule
