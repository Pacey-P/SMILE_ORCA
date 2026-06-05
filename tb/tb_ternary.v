`timescale 1ns/1ps
//============================================================================
// tb_ternary.v -- self-checking TB for the ternary CIM tile + few-bit readout.
//  (1) ternary add/sub/skip array == EXACT ternary integer matmul (bit-exact).
//  (2) ternary_readout (5-bit single-slope) == range-matched round quantizer,
//      and reconstruction error <= 1 LSB.
// Only measured pass/fail counts are printed.
//============================================================================
module tb_ternary;
    localparam integer M    = 8;
    localparam integer N    = 16;
    localparam integer XB   = 8;
    localparam integer ACCW = XB + 4 + 1;          // $clog2(16)=4 -> 13
    localparam integer R    = 5;
    localparam integer QMAX = (1<<(R-1)) - 1;      // 15

    reg  [M*N*2-1:0]  w_flat;
    reg  [N*XB-1:0]   x_flat;
    wire [M*ACCW-1:0] y_flat;

    ternary_mac_tile #(.M(M),.N(N),.XB(XB)) tile (
        .w_flat(w_flat), .x_flat(x_flat), .y_flat(y_flat));

    // readout DUT
    reg                  clk=0, rst_n=0, start=0;
    reg  signed [ACCW-1:0] P;
    reg         [ACCW-1:0] step;
    wire                 done;
    wire signed [R-1:0]  code;
    ternary_readout #(.PW(ACCW),.R(R)) ro (
        .clk(clk),.rst_n(rst_n),.start(start),.P(P),.step(step),.done(done),.code(code));
    always #5 clk=~clk;

    integer mac_tot=0, mac_fail=0, rd_tot=0, rd_fail=0, lin_fail=0;
    integer mm, nn, t, idx;
    integer gold [0:M-1];
    integer wv, xv, got, absP, gcnt, gcode, recon, maxabs;

    task compute_golden;
        begin
            maxabs=0;
            for (mm=0; mm<M; mm=mm+1) begin
                gold[mm]=0;
                for (nn=0; nn<N; nn=nn+1) begin
                    wv = w_flat[(mm*N+nn)*2+0] ? (w_flat[(mm*N+nn)*2+1] ? -1 : 1) : 0;
                    xv = $signed(x_flat[nn*XB +: XB]);
                    gold[mm] = gold[mm] + wv*xv;
                end
                got = $signed(y_flat[mm*ACCW +: ACCW]);
                mac_tot = mac_tot + 1;
                if (got !== gold[mm]) begin
                    mac_fail = mac_fail + 1;
                    $display("  MAC FAIL m=%0d dut=%0d exact=%0d", mm, got, gold[mm]);
                end
                if ((gold[mm]>=0?gold[mm]:-gold[mm]) > maxabs) maxabs = (gold[mm]>=0?gold[mm]:-gold[mm]);
            end
        end
    endtask

    task check_readout;
        begin
            step = (maxabs+QMAX-1)/QMAX; if (step==0) step=1;   // range-matched LSB
            for (mm=0; mm<M; mm=mm+1) begin
                P = gold[mm];
                @(negedge clk); start=1; @(negedge clk); start=0;
                wait(done==1'b1); @(negedge clk);
                got = code;
                absP = (gold[mm]>=0)?gold[mm]:-gold[mm];
                gcnt = (2*absP + step)/(2*step); if (gcnt>QMAX) gcnt=QMAX;   // round-half-up, clamp
                gcode = (gold[mm]<0) ? -gcnt : gcnt;
                rd_tot = rd_tot + 1;
                if (got !== gcode) begin
                    rd_fail = rd_fail + 1;
                    $display("  RD FAIL m=%0d P=%0d step=%0d code=%0d expect=%0d", mm, gold[mm], step, got, gcode);
                end
                recon = got*step - gold[mm]; if (recon<0) recon=-recon;
                if (recon > step) begin   // within 1 LSB (saturation excepted)
                    if ((got!==QMAX) && (got!==-QMAX)) begin
                        lin_fail = lin_fail + 1;
                        $display("  LIN FAIL m=%0d |code*step-P|=%0d > step=%0d", mm, recon, step);
                    end
                end
            end
        end
    endtask

    task run_vec; begin #1; compute_golden; check_readout; end endtask

    initial begin
        rst_n=0; repeat(3) @(negedge clk); rst_n=1; @(negedge clk);
        // edge: all-zero weights
        w_flat=0; for(idx=0;idx<N;idx=idx+1) x_flat[idx*XB +: XB]=$random; run_vec;
        // edge: all +1
        for(idx=0;idx<M*N;idx=idx+1) w_flat[idx*2 +: 2]=2'b01; run_vec;
        // edge: all -1
        for(idx=0;idx<M*N;idx=idx+1) w_flat[idx*2 +: 2]=2'b11; run_vec;
        // edge: max activations, mixed weights
        for(idx=0;idx<N;idx=idx+1) x_flat[idx*XB +: XB]=8'sd127;
        for(idx=0;idx<M*N;idx=idx+1) w_flat[idx*2 +: 2]=(idx[0]?2'b11:2'b01); run_vec;
        // random trials (~30% zero weights)
        for(t=0;t<150;t=t+1) begin
            for(idx=0;idx<M*N;idx=idx+1) begin
                if (($random%10) < 3) w_flat[idx*2 +: 2]=2'b00;          // ~30% skip
                else w_flat[idx*2 +: 2]={($random&1)?1'b1:1'b0,1'b1};    // nz=1, random sign
            end
            for(idx=0;idx<N;idx=idx+1) x_flat[idx*XB +: XB]=$random;
            run_vec;
        end
        $display("================================================");
        $display("MAC: %0d/%0d bit-exact vs ternary matmul (%0d fail)", mac_tot-mac_fail, mac_tot, mac_fail);
        $display("RD : %0d/%0d match round-quantizer (%0d fail), %0d linearity-fail",
                 rd_tot-rd_fail, rd_tot, rd_fail, lin_fail);
        if (mac_fail==0 && rd_fail==0 && lin_fail==0)
            $display("RESULT: ternary add/sub/skip EXACT + %0d-bit readout CORRECT", R);
        else $display("RESULT: FAILURES DETECTED");
        $display("================================================");
        $finish;
    end
    initial begin #2000000; $display("FAIL: watchdog"); $finish; end
endmodule
