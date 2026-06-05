`timescale 1ns/1ps
//============================================================================
// ternary_mac_tile.v -- ternary-weight CIM MAC tile (add/sub/skip), synth-style
//
// y[m] = sum_n W[m][n] * x[n],  W[m][n] in {-1,0,+1}
//
// The ternary primitive is ADD / SUBTRACT / SKIP -- no multiplier:
//   weight encoded as 2 bits {nz, sign}: nz=0 -> SKIP (contributes 0, no add),
//   nz=1 -> add (+x) if sign=0, subtract (-x) if sign=1.
// This is exactly what a ternary in-memory array does; here it is a digital
// add/sub/skip reduction (combinational adder tree). Bit-exact integer result;
// the lossy few-bit readout is a separate module (ternary_readout.v).
//============================================================================
module ternary_mac_tile #(
    parameter integer M    = 8,
    parameter integer N    = 16,
    parameter integer XB   = 8,                          // signed activation bits
    parameter integer ACCW = XB + $clog2(N) + 1          // exact accumulator width
)(
    input  wire [M*N*2-1:0]  w_flat,   // 2 bits/weight: bit0=nz, bit1=sign
    input  wire [N*XB-1:0]   x_flat,   // signed XB activations
    output reg  [M*ACCW-1:0] y_flat    // signed exact partial sums
);
    integer mm, nn;
    reg signed [ACCW-1:0] acc;
    reg signed [XB-1:0]   xv;
    reg nz, sgn;
    always @(*) begin
        for (mm = 0; mm < M; mm = mm + 1) begin
            acc = {ACCW{1'b0}};
            for (nn = 0; nn < N; nn = nn + 1) begin
                nz  = w_flat[(mm*N+nn)*2 + 0];
                sgn = w_flat[(mm*N+nn)*2 + 1];
                xv  = x_flat[nn*XB +: XB];
                if (nz) begin
                    if (sgn) acc = acc - xv;             // SUBTRACT
                    else     acc = acc + xv;             // ADD
                end                                      // else SKIP
            end
            y_flat[mm*ACCW +: ACCW] = acc;
        end
    end
endmodule
