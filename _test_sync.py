import sys; from pathlib import Path
HERE=Path.cwd(); sys.path.insert(0,str(HERE.parent/"labeling_toolkit"))
import features as F
base="anchors_fixed_figs"
# the 5 FIS/FSW review candidates DON'T all have fixed renders; test the anchors we DO have + note expected
cases=[("seed003__sequential_H","0027","review: 8ch,cross44 -> expect FSW-ish >4"),
       ("seed001__sequential_H","0099","GSW: 28ch broad"),
       ("seed000__sequential_noH","0071","seizure spread"),
       ("seed001__sequential_noH","0047","seizure spikewave"),
       ("seed002__sequential_noH","0051","sustained ictal"),
       ("seed002__sequential_noH","0063","sustained ictal")]
for run,ep,note in cases:
    lfp=f"{base}/{run}/epoch_{ep}/lfp.png"
    f=F.lfp_features(lfp)
    print(f"{run} ep{ep}: sync_extent={f['sync_spike_extent']:2d} cross={f['cross_cols']:4d} "
          f"n_active={f['n_active_channels']:2d} fast_dur={f['fast_activity_duration_s']:.2f}  [{note}]")
