# SMILE_ORCA -- digital CIM matmul + time-domain readout research prototype
# Digital/behavioral simulation only (NOT analog circuit design, NOT silicon).
IV    = iverilog -g2005
VVP   = vvp
SIM   = sim

.PHONY: all p1 clean
all: p1

# ---- Piece 1: bit-sliced digital CIM MAC tile -----------------------------
p1: $(SIM)/tb_cim_mac_tile.vvp
	$(VVP) $<

$(SIM)/tb_cim_mac_tile.vvp: rtl/cim_mac_tile.v tb/tb_cim_mac_tile.v
	$(IV) -o $@ $^

clean:
	rm -f $(SIM)/*.vvp $(SIM)/*.vcd
