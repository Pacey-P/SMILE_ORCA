`timescale 1ns/1ps
//============================================================================
// cim_mac_tile.v  --  Bit-sliced digital CIM MAC tile (synthesizable-style)
//
// Computes   y[m] = sum_n W[m][n] * x[n]    for m in [0,M), n in [0,N)
//
// "CIM-like" structure (digital / behavioral model, NOT analog):
//   * Weights are stored in the tile (weight-stationary), each weight sliced
//     into WB bit-planes. Inputs are streamed BIT-SERIALLY: one input
//     bit-plane per clock cycle (XB cycles total) -- like real CIM macros.
//   * Core primitive = the BITLINE POPCOUNT:
//        B[m,i,j] = sum_n ( w[m][n][i] & x[n][j] )
//     count of cells in output column m whose weight-bit-plane i AND the
//     current input bit-plane j are both 1. In analog CIM this is bitline
//     current; in DIGITAL CIM (what we model) it is a popcount / adder tree.
//   * Partial sum for input bit j:
//        P[m,j] = sum_i s_i * 2^i * B[m,i,j]      (s_i = -1 for weight MSB)
//   * Bit-serial accumulation over input bits:
//        y[m]   = sum_j s_j * 2^j * P[m,j]        (s_j = -1 for input MSB)
//   Algebraically identical to exact signed two's-complement integer matmul.
//
// Latency: XB cycles (one per input bit-plane) + handshake.
//============================================================================
module cim_mac_tile #(
    parameter integer M    = 8,                       // output rows
    parameter integer N    = 16,                      // input cols
    parameter integer WB   = 8,                       // weight bits (signed)
    parameter integer XB   = 8,                       // input  bits (signed)
    parameter integer ACCW = WB + XB + $clog2(N) + 2  // accumulator width
)(
    input  wire                clk,
    input  wire                rst_n,
    input  wire                start,    // pulse: latch operands, begin
    input  wire [M*N*WB-1:0]   w_flat,   // weights, row-major w[m][n], WB each
    input  wire [N*XB-1:0]     x_flat,   // inputs   x[n], XB each
    output reg                 done,     // pulses high for 1 cyc when y valid
    output reg  [M*ACCW-1:0]   y_flat    // outputs  y[m], ACCW each (signed)
);

    localparam integer JW = (XB > 1) ? $clog2(XB) : 1;

    // ---- weight-stationary storage + latched input vector -----------------
    reg [WB-1:0] W [0:M*N-1];
    reg [XB-1:0] X [0:N-1];

    reg [JW:0]                jcur;       // current input bit-plane index
    reg signed [ACCW-1:0]     acc [0:M-1];

    // ---- combinational: partial sums P[m] for the current input plane -----
    // P[m] = sum_i s_i*2^i * popcount_n( W[m][n][i] & X[n][jcur] )
    reg signed [ACCW-1:0] Pcomb [0:M-1];
    integer mm, nn, ii;
    reg [15:0] pc;   // bitline popcount accumulator (N <= 65535)
    always @(*) begin
        for (mm = 0; mm < M; mm = mm + 1) begin
            Pcomb[mm] = {ACCW{1'b0}};
            for (ii = 0; ii < WB; ii = ii + 1) begin
                pc = 16'd0;
                for (nn = 0; nn < N; nn = nn + 1)
                    pc = pc + (W[mm*N+nn][ii] & X[nn][jcur]); // CIM bitline
                if (ii == WB-1)   // weight sign bit -> negative weight
                    Pcomb[mm] = Pcomb[mm] - ($signed({1'b0,pc}) <<< ii);
                else
                    Pcomb[mm] = Pcomb[mm] + ($signed({1'b0,pc}) <<< ii);
            end
        end
    end

    // ---- FSM: load -> stream XB input planes -> done ----------------------
    localparam [1:0] S_IDLE = 2'd0, S_RUN = 2'd1, S_DONE = 2'd2;
    reg [1:0] state;
    integer k;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; done <= 1'b0; jcur <= 0;
            for (k = 0; k < M; k = k + 1) acc[k] <= {ACCW{1'b0}};
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: begin
                    if (start) begin
                        for (k = 0; k < M*N; k = k + 1)
                            W[k] <= w_flat[k*WB +: WB];
                        for (k = 0; k < N; k = k + 1)
                            X[k] <= x_flat[k*XB +: XB];
                        for (k = 0; k < M; k = k + 1) acc[k] <= {ACCW{1'b0}};
                        jcur  <= 0;
                        state <= S_RUN;
                    end
                end
                S_RUN: begin
                    // accumulate this input plane with weight s_j * 2^jcur
                    for (mm = 0; mm < M; mm = mm + 1) begin
                        if (jcur == XB-1)   // input sign bit -> negative
                            acc[mm] <= acc[mm] - (Pcomb[mm] <<< jcur);
                        else
                            acc[mm] <= acc[mm] + (Pcomb[mm] <<< jcur);
                    end
                    if (jcur == XB-1) state <= S_DONE;
                    else              jcur  <= jcur + 1'b1;
                end
                S_DONE: begin
                    for (mm = 0; mm < M; mm = mm + 1)
                        y_flat[mm*ACCW +: ACCW] <= acc[mm];
                    done  <= 1'b1;
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
