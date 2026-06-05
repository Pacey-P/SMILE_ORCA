`timescale 1ns/1ps
// tb_dualslope.v -- self-check: dual-slope counter computes floor(N1*|P|/ref),
// clamped to +/-(2^(R-1)-1), with sign. Proves the ratiometric two-phase counter.
module tb_dualslope;
    localparam integer PW=16, R=5, N1=(1<<(R-1))-1, QMAX=(1<<(R-1))-1;
    reg clk=0, rst_n=0, start=0;
    reg signed [PW-1:0] P; reg [PW-1:0] ref;
    wire done; wire signed [R-1:0] code;
    dualslope_readout #(.PW(PW),.R(R)) dut(.clk(clk),.rst_n(rst_n),.start(start),
        .P(P),.ref(ref),.done(done),.code(code));
    always #5 clk=~clk;
    integer tot=0, fail=0, i, absP, gc, gcode, got;
    task conv(input signed [PW-1:0] pv, input [PW-1:0] rv); begin
        P=pv; ref=rv;
        @(negedge clk); start=1; @(negedge clk); start=0;
        wait(done==1); @(negedge clk);
        got=code; absP=(pv<0)?-pv:pv;
        gc=(N1*absP)/rv; if(gc>QMAX) gc=QMAX; gcode=(pv<0)?-gc:gc;
        tot=tot+1;
        if(got!==gcode) begin fail=fail+1;
            $display("  FAIL P=%0d ref=%0d code=%0d expect=%0d",pv,rv,got,gcode); end
    end endtask
    initial begin
        rst_n=0; repeat(3) @(negedge clk); rst_n=1; @(negedge clk);
        conv(0,100); conv(100,100); conv(50,100); conv(1500,100);
        for(i=0;i<120;i=i+1) conv($random%2000, 50+($random%400));
        $display("================================================");
        $display("dual-slope readout: %0d/%0d == floor(N1*|P|/ref) (%0d fail)",tot-fail,tot,fail);
        if(fail==0) $display("RESULT: dual-slope ratiometric counter CORRECT"); else $display("RESULT: FAIL");
        $display("================================================");
        $finish;
    end
    initial begin #3000000; $display("FAIL: watchdog"); $finish; end
endmodule
