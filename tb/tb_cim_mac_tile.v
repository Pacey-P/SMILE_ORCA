`timescale 1ns/1ps
//============================================================================
// tb_cim_mac_tile.v -- self-checking TB: DUT must match EXACT integer matmul
//   * many random matrices
//   * edge cases: all zeros, all max-positive, all min-negative, mixed extremes
// Prints only measured pass/fail counts. No result is asserted unless the
// DUT output was compared bit-exact to a golden integer computation here.
//============================================================================
module tb_cim_mac_tile;
    localparam integer M    = 8;
    localparam integer N    = 16;
    localparam integer WB   = 8;
    localparam integer XB   = 8;
    localparam integer ACCW = WB + XB + 4 + 2;   // = $clog2(16)=4
    localparam integer WMAX =  (1<<(WB-1)) - 1;   //  127
    localparam integer WMIN = -(1<<(WB-1));       // -128
    localparam integer XMAXV =  (1<<(XB-1)) - 1;
    localparam integer XMINV = -(1<<(XB-1));

    reg                clk = 0, rst_n = 0, start = 0;
    reg  [M*N*WB-1:0]  w_flat;
    reg  [N*XB-1:0]    x_flat;
    wire               done;
    wire [M*ACCW-1:0]  y_flat;

    cim_mac_tile #(.M(M),.N(N),.WB(WB),.XB(XB)) dut (
        .clk(clk),.rst_n(rst_n),.start(start),
        .w_flat(w_flat),.x_flat(x_flat),.done(done),.y_flat(y_flat));

    always #5 clk = ~clk;   // 100 MHz

    integer total = 0, fails = 0;
    integer mm, nn;
    integer gold [0:M-1];
    integer wv, xv, got;

    // run one vector: drive current w_flat/x_flat, check vs golden
    task run_and_check(input [255:0] label);
        begin
            // golden: exact signed integer matmul
            for (mm = 0; mm < M; mm = mm + 1) begin
                gold[mm] = 0;
                for (nn = 0; nn < N; nn = nn + 1) begin
                    wv = $signed(w_flat[(mm*N+nn)*WB +: WB]);
                    xv = $signed(x_flat[(nn)*XB +: XB]);
                    gold[mm] = gold[mm] + wv*xv;
                end
            end
            @(negedge clk); start = 1; @(negedge clk); start = 0;
            wait (done == 1'b1); @(negedge clk);
            for (mm = 0; mm < M; mm = mm + 1) begin
                got   = $signed(y_flat[mm*ACCW +: ACCW]);
                total = total + 1;
                if (got !== gold[mm]) begin
                    fails = fails + 1;
                    $display("  FAIL [%0s] m=%0d  dut=%0d  exact=%0d",
                             label, mm, got, gold[mm]);
                end
            end
        end
    endtask

    integer t, idx;
    initial begin
        $dumpfile("sim/tb_cim_mac_tile.vcd"); $dumpvars(0, tb_cim_mac_tile);
        rst_n = 0; repeat (3) @(negedge clk); rst_n = 1; @(negedge clk);

        // ---- edge case 1: all zeros --------------------------------------
        w_flat = 0; x_flat = 0; run_and_check("zeros");

        // ---- edge case 2: all weights & inputs = max positive ------------
        for (idx=0; idx<M*N; idx=idx+1) w_flat[idx*WB +: WB] = WMAX[WB-1:0];
        for (idx=0; idx<N;   idx=idx+1) x_flat[idx*XB +: XB] = XMAXV[XB-1:0];
        run_and_check("maxpos");

        // ---- edge case 3: all = min negative -----------------------------
        for (idx=0; idx<M*N; idx=idx+1) w_flat[idx*WB +: WB] = WMIN[WB-1:0];
        for (idx=0; idx<N;   idx=idx+1) x_flat[idx*XB +: XB] = XMINV[XB-1:0];
        run_and_check("minneg");

        // ---- edge case 4: mixed extremes (checkerboard signs) ------------
        for (idx=0; idx<M*N; idx=idx+1)
            w_flat[idx*WB +: WB] = (idx[0] ? WMIN[WB-1:0] : WMAX[WB-1:0]);
        for (idx=0; idx<N; idx=idx+1)
            x_flat[idx*XB +: XB] = (idx[0] ? XMAXV[XB-1:0] : XMINV[XB-1:0]);
        run_and_check("mixedext");

        // ---- random trials ------------------------------------------------
        for (t = 0; t < 200; t = t + 1) begin
            for (idx=0; idx<M*N; idx=idx+1) w_flat[idx*WB +: WB] = $random;
            for (idx=0; idx<N;   idx=idx+1) x_flat[idx*XB +: XB] = $random;
            run_and_check("random");
        end

        $display("================================================");
        $display("CIM MAC tile self-check: %0d/%0d checks passed, %0d failed",
                 total - fails, total, fails);
        if (fails == 0) $display("RESULT: ALL CHECKS MATCH EXACT INTEGER MATMUL");
        else            $display("RESULT: MISMATCHES DETECTED");
        $display("================================================");
        $finish;
    end

    // watchdog
    initial begin #500000; $display("FAIL: watchdog timeout"); $finish; end
endmodule
