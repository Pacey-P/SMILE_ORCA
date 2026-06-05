`timescale 1ns/1ps
//============================================================================
// dualslope_readout.v -- drift-immune dual-slope counter readout (synth-style)
//
// Two-phase ratiometric conversion:
//   RUN-UP   : integrate |P| for a FIXED N1 clock cycles -> acc = N1*|P|
//   RUN-DOWN : subtract a reference `ref` each cycle, counting until acc < ref
//              -> code = floor(N1*|P| / ref)
// Because BOTH phases are timed by the SAME clock and use the SAME integrator,
// clock-period and ramp-current drift cancel (the result is a ratio N1*|P|/ref,
// independent of absolute timing) -- the classic dual-slope advantage that a
// single-slope readout lacks. Latency = N1 + up to (N1*|P|max/ref) cycles
// (~2x single-slope). code is signed R-bit, clamped to +/-(2^(R-1)-1).
//============================================================================
module dualslope_readout #(
    parameter integer PW   = 20,
    parameter integer R    = 5,
    parameter integer N1   = (1<<(R-1)) - 1,             // run-up cycles
    parameter integer ACCW = PW + 6
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 start,
    input  wire signed [PW-1:0] P,
    input  wire        [PW-1:0] ref,        // reference (= range), design constant
    output reg                  done,
    output reg signed [R-1:0]   code
);
    localparam integer QMAX = (1<<(R-1)) - 1;
    localparam [1:0] S_IDLE=2'd0, S_UP=2'd1, S_DN=2'd2;
    reg [1:0]         state;
    reg               sgn;
    reg [PW-1:0]      mag;
    reg [ACCW-1:0]    acc;
    reg [$clog2(N1+1)-1:0] upc;
    reg [R-1:0]       cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state<=S_IDLE; done<=1'b0; code<={R{1'b0}}; acc<={ACCW{1'b0}};
        end else begin
            done<=1'b0;
            case (state)
                S_IDLE: if (start) begin
                    sgn <= P[PW-1];
                    mag <= P[PW-1] ? (~P + 1'b1) : P;
                    acc <= {ACCW{1'b0}};
                    upc <= {($clog2(N1+1)){1'b0}};
                    cnt <= {R{1'b0}};
                    state <= S_UP;
                end
                S_UP: begin                                   // integrate |P|, N1 cycles
                    acc <= acc + {{(ACCW-PW){1'b0}}, mag};
                    upc <= upc + 1'b1;
                    if (upc == N1-1) state <= S_DN;
                end
                S_DN: begin                                   // de-integrate by ref
                    if ((acc >= {{(ACCW-PW){1'b0}}, ref}) && (cnt != QMAX[R-1:0])) begin
                        acc <= acc - {{(ACCW-PW){1'b0}}, ref};
                        cnt <= cnt + 1'b1;
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
