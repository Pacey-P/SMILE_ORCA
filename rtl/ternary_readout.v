`timescale 1ns/1ps
//============================================================================
// ternary_readout.v -- few-bit range-matched readout (single-slope counter)
//
// Converts a signed partial sum P into an R-bit code via a constant-step
// single-slope ramp + comparator (a counter/time-domain ADC). The behavioral
// study found R ~ 4-5 bits preserves transformer quality for ternary tiles
// (partial sums are concentrated), so this readout is small and cheap.
//
//   code = sign(P) * round(|P| / step)   clamped to +/-(2^(R-1)-1)
// Round-to-nearest is exact (compares 2*|P| against (2k-1)*step, no half-LSB
// truncation). Latency <= 2^(R-1) cycles. `step` is the per-column LSB
// (= range / (2^(R-1)-1)), supplied by offline calibration.
//============================================================================
module ternary_readout #(
    parameter integer PW = 20,        // partial-sum width (signed)
    parameter integer R  = 5          // readout bits (signed code)
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 start,
    input  wire signed [PW-1:0] P,        // partial sum to convert
    input  wire        [PW-1:0] step,     // LSB (>=1)
    output reg                  done,
    output reg signed [R-1:0]   code      // signed R-bit code
);
    localparam integer QMAX = (1 << (R-1)) - 1;
    localparam [1:0] S_IDLE=2'd0, S_RUN=2'd1, S_DONE=2'd2;
    reg [1:0]        state;
    reg              sgn;
    reg [PW:0]       mag2;             // 2*|P|
    reg [PW+1:0]     thr;             // (2k-1)*step threshold, grows by 2*step
    reg [R-1:0]      cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state<=S_IDLE; done<=1'b0; code<={R{1'b0}}; cnt<={R{1'b0}};
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: if (start) begin
                    sgn  <= P[PW-1];
                    mag2 <= P[PW-1] ? ({1'b0,(~P + 1'b1)} <<< 1) : ({1'b0,P} <<< 1);
                    thr  <= {1'b0, step};          // (2*0+1)*step
                    cnt  <= {R{1'b0}};
                    state<= S_RUN;
                end
                S_RUN: begin
                    if ((mag2 >= thr) && (cnt != QMAX[R-1:0])) begin
                        cnt <= cnt + 1'b1;
                        thr <= thr + ({1'b0, step} <<< 1);   // += 2*step
                    end else begin
                        code  <= sgn ? (-$signed({1'b0,cnt})) : $signed({1'b0,cnt});
                        done  <= 1'b1;
                        state <= S_IDLE;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
