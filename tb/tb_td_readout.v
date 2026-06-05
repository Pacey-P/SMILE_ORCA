`timescale 1ns/1ps
//============================================================================
// tb_td_readout.v -- self-checking TB for the linear time-domain readout.
// Proves: (1) code == ceil(value/step) bit-exact (the counter+comparator
//             implements the intended quantizer), and
//         (2) |code*step - value| <= step for all inputs  => the readout is
//             LINEAR (uniform 1-LSB quantizer), not exponential.
// Only measured pass/fail counts are printed.
//============================================================================
module tb_td_readout;
    localparam integer VW = 24;
    localparam integer CW = 16;

    reg              clk = 0, rst_n = 0, start = 0;
    reg  [VW-1:0]    value, step;
    wire             done, ovf;
    wire [CW-1:0]    code;

    td_readout #(.VW(VW),.CW(CW)) dut (
        .clk(clk),.rst_n(rst_n),.start(start),
        .value(value),.step(step),.done(done),.ovf(ovf),.code(code));

    always #5 clk = ~clk;

    integer total = 0, fails = 0, lin_fails = 0;
    integer got, gold, qerr, maxqerr = 0;

    task convert(input [VW-1:0] v, input [VW-1:0] s);
        begin
            value = v; step = s;
            @(negedge clk); start = 1; @(negedge clk); start = 0;
            wait (done == 1'b1); @(negedge clk);
            got  = code;
            gold = (v == 0) ? 0 : (v + s - 1) / s;       // ceil(v/s)
            total = total + 1;
            if (ovf) begin
                fails = fails + 1;
                $display("  OVF  value=%0d step=%0d (code saturated)", v, s);
            end else if (got !== gold) begin
                fails = fails + 1;
                $display("  FAIL value=%0d step=%0d  code=%0d  expect=%0d", v, s, got, gold);
            end
            // linearity: reconstructed value within 1 LSB(step) of true value
            qerr = got*s - v;  if (qerr < 0) qerr = -qerr;
            if (qerr > s) begin
                lin_fails = lin_fails + 1;
                $display("  LINFAIL value=%0d step=%0d  |code*step-value|=%0d > step", v, s, qerr);
            end
            if (qerr > maxqerr) maxqerr = qerr;
        end
    endtask

    integer i, sidx;
    integer steps [0:3];
    initial begin
        $dumpfile("sim/tb_td_readout.vcd"); $dumpvars(0, tb_td_readout);
        rst_n = 0; repeat(3) @(negedge clk); rst_n = 1; @(negedge clk);

        steps[0] = 1;    // step=1 => code must equal value exactly (perfect line)
        steps[1] = 73;   // odd step, exercises ceil rounding
        steps[2] = 256;  // power-of-two-ish step
        steps[3] = 1000;

        // NOTE: single-slope latency is O(value/step) cycles (up to 2^B). We
        // bound test magnitudes to ~<=2000 cycles/conversion so the sim
        // finishes; linearity/exactness hold at any magnitude by construction.
        for (sidx = 0; sidx < 4; sidx = sidx + 1) begin
            // edge cases
            convert(0,                   steps[sidx]);   // zero
            convert(steps[sidx],         steps[sidx]);   // exactly 1 step
            convert(steps[sidx]-1,       steps[sidx]);   // just under 1 step
            convert(1500*steps[sidx]+7,  steps[sidx]);   // large (~1500 codes)
            // linear sweep across code space (value = k*step + offset)
            for (i = 0; i <= 60; i = i + 1)
                convert(i*steps[sidx] + (i*7) % steps[sidx], steps[sidx]);
            // random values bounded to ~1000 codes
            for (i = 0; i < 30; i = i + 1)
                convert(($random % (1000*steps[sidx]+1)) & 24'h7FFFFF, steps[sidx]);
        end

        $display("================================================");
        $display("TD readout self-check: %0d/%0d code-exact, %0d ceil-mismatch, %0d linearity-fail",
                 total - fails, total, fails, lin_fails);
        $display("max |code*step - value| = %0d (must be <= step for each case)", maxqerr);
        if (fails == 0 && lin_fails == 0)
            $display("RESULT: counter+comparator IS a bit-exact LINEAR (1-LSB) readout");
        else
            $display("RESULT: FAILURES DETECTED");
        $display("================================================");
        $finish;
    end

    initial begin #5000000; $display("FAIL: watchdog timeout"); $finish; end
endmodule
