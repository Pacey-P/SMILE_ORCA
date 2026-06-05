#!/usr/bin/env python3
"""energy_ternary.py -- gate-level energy estimate for the ternary CIM tile,
HONEST about which regime the readout-precision finding actually helps.

Key distinction (this is the whole point):
  * DIGITAL CIM tile (an adder tree, = rtl/ternary_mac_tile.v): the add/sub/skip
    COMPUTE dominates; the readout is trivial. Ternary's win here is "no
    multiplier + skip zeros" (known, BitNet-style). The readout-bit finding is
    ~MOOT in this regime.
  * ANALOG CIM tile: the MAC is ~free (charge accumulation on bitlines); the
    per-column ADC/readout DOMINATES. THIS is where the measured partial-sum
    concentration -> R*=4-5 bit readout pays off (~8x on the dominant cost),
    and it is the analog+ternary pairing the design thesis highlighted.

The RTL we built+verified is the DIGITAL functional model (proves the ternary
arithmetic and the few-bit readout are bit-exact). The readout-ENERGY win is an
ANALOG-CIM claim. ALL constants are labeled assumptions (45nm Horowitz regime +
piece-2 ADC FoM); behavioral estimate, not silicon.
"""
pJ=1.0; fJ=1e-3
# ASSUMPTIONS -----------------------------------------------------------------
E_ADD8   = 0.03*pJ     # 8-bit int add/sub (Horowitz'14). ASSUMPTION.
E_MUL8   = 0.20*pJ     # 8-bit int multiply. ASSUMPTION.
E_ANALOG_MAC = 0.0005*pJ  # per-input analog accumulate (~free). ASSUMPTION.
FOM_ADC  = 10*fJ       # ADC energy = FoM*2^B per conversion (piece-2). ASSUMPTION.
SKIP     = 0.31        # MEASURED ternary weight zero-fraction.

def E_adc(B): return FOM_ADC*(2**B)

def main():
    M,N=256,1024
    print("="*74); print("TERNARY CIM TILE energy -- digital vs analog regime (per tile-eval)")
    print("="*74)
    print(f"tile M={M} cols, N={N} fan-in | skip={SKIP:.0%} (measured) | ADC FoM={FOM_ADC/fJ:.0f} fJ\n")

    # ---- DIGITAL regime: compute-bound -------------------------------------
    dense_comp  = M*N*(E_MUL8+E_ADD8)
    tern_comp   = M*N*(1-SKIP)*E_ADD8
    print("(1) DIGITAL CIM (adder tree = the RTL we verified): COMPUTE-bound")
    print(f"    dense int8 compute   = {dense_comp:9.1f} pJ")
    print(f"    ternary a/s/skip     = {tern_comp:9.1f} pJ   ({dense_comp/tern_comp:.1f}x  <- no mult + skip)")
    print(f"    readout (any 5-8b)   ~ a few pJ  -> NEGLIGIBLE here; readout-bit finding MOOT")
    print(f"    => ternary win in digital regime = {dense_comp/tern_comp:.1f}x, from no-multiplier+skip (known).")

    # ---- ANALOG regime: readout/ADC-bound (where the finding lands) --------
    analog_mac = M*N*E_ANALOG_MAC          # ~free
    ro8 = M*E_adc(8); ro5 = M*E_adc(5)
    print("\n(2) ANALOG CIM (MAC ~free; per-column ADC dominates): READOUT-bound")
    print(f"    analog MAC (~free)   = {analog_mac:9.2f} pJ")
    print(f"    8-bit ADC readout    = {ro8:9.2f} pJ   (generic CIM)")
    print(f"    5-bit ADC readout    = {ro5:9.2f} pJ   (R*, from ternary concentration)")
    t8=analog_mac+ro8; t5=analog_mac+ro5
    print(f"    tile total 8b -> 5b  = {t8:9.2f} -> {t5:9.2f} pJ  ({t8/t5:.1f}x)")
    print(f"    => HERE the readout-bit finding pays off: ~{ro8/ro5:.0f}x on the DOMINANT cost.")
    print("       (ratio is 2^(8-5)=8x; range-matching + concentration is what makes 5b safe.)")

    print("\nHONEST BOTTOM LINE:")
    print("  - 'no multiplier + 31% skip' (~7x compute) is real but KNOWN (BitNet-class).")
    print("  - The NOVEL bit -- few-bit readout from measured partial-sum concentration --")
    print("    matters specifically for ANALOG ternary CIM, where the ADC is the bottleneck,")
    print("    giving ~8x on that dominant term. It is ~moot for a digital adder-tree tile.")
    print("  - Weight movement = 0 (CIM premise). Constants are assumptions; ratios robust.")
    print("="*74)

if __name__=="__main__": main()
