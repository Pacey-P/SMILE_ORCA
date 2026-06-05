# SMILE_ORCA -- digital CIM matmul + time-domain readout research prototype
# Digital/behavioral simulation only (NOT analog circuit design, NOT silicon).
IV    = iverilog -g2005
VVP   = vvp
SIM   = sim

.PHONY: all p1 p2 p2_rtl p2_py clean
all: p1 p2

# ---- Piece 1: bit-sliced digital CIM MAC tile -----------------------------
p1: $(SIM)/tb_cim_mac_tile.vvp
	$(VVP) $<

$(SIM)/tb_cim_mac_tile.vvp: rtl/cim_mac_tile.v tb/tb_cim_mac_tile.v
	$(IV) -o $@ $^

# ---- Piece 2: linear time-domain readout vs ADC ---------------------------
p2: p2_py p2_rtl

p2_py:
	python3 py/readout_model.py

p2_rtl: $(SIM)/tb_td_readout.vvp
	$(VVP) $<

$(SIM)/tb_td_readout.vvp: rtl/td_readout.v tb/tb_td_readout.v
	$(IV) -o $@ $^

clean:
	rm -f $(SIM)/*.vvp $(SIM)/*.vcd
