`timescale 1ns/1ps
//============================================================================
// td_readout.v -- Linear time-domain readout: counter + comparator (synth-style)
//
// Models a CONSTANT-CURRENT single-slope conversion of a CIM partial sum:
//   * `value` = |partial sum| as accumulated charge (digital proxy).
//   * `step`  = charge added per clock = constant current * T_clk = the LSB.
//   * A ramp accumulates `step` each cycle; a comparator fires when ramp >=
//     value; a counter records the number of cycles => the digital code.
//
//   code = ceil(value / step)      <-- LINEAR in value (constant step),
//                                      NOT exponential RC decay (~ln(value)).
//
// This is the digital primitive behind the numpy TD model in py/readout_model.py
// (that model uses round(); hardware single-slope naturally yields ceil(); both
// are 1-LSB-accurate UNIFORM/LINEAR quantizers). Latency = code cycles.
//============================================================================
module td_readout #(
    parameter integer VW = 24,   // value / ramp width
    parameter integer CW = 16    // counter (code) width
)(
    input  wire           clk,
    input  wire           rst_n,
    input  wire           start,
    input  wire [VW-1:0]  value,   // |partial sum| (charge units)
    input  wire [VW-1:0]  step,    // charge per clock = constant current = LSB
    output reg            done,    // pulses high for 1 cycle when code valid
    output reg            ovf,     // set if code saturates (value/step too big)
    output reg [CW-1:0]   code     // = ceil(value/step)
);
    localparam [1:0] S_IDLE = 2'd0, S_RUN = 2'd1, S_DONE = 2'd2;
    reg [1:0]    state;
    reg [VW:0]   ramp;             // +1 bit headroom for ramp + step
    reg [VW-1:0] val_l, step_l;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; done <= 1'b0; ovf <= 1'b0; code <= {CW{1'b0}};
            ramp  <= {(VW+1){1'b0}};
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: if (start) begin
                    val_l <= value; step_l <= step;
                    ramp  <= {(VW+1){1'b0}}; code <= {CW{1'b0}}; ovf <= 1'b0;
                    state <= S_RUN;
                end
                S_RUN: begin
                    if (ramp >= {1'b0, val_l}) begin       // comparator fires
                        state <= S_DONE;
                    end else if (code == {CW{1'b1}}) begin // saturate guard
                        ovf <= 1'b1; state <= S_DONE;
                    end else begin
                        code <= code + 1'b1;               // count one clock
                        ramp <= ramp + {1'b0, step_l};     // constant-current ramp
                    end
                end
                S_DONE: begin done <= 1'b1; state <= S_IDLE; end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
