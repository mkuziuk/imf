import numpy as np
import pandas as pd

from robustness.analysis import did_vector, specs
from robustness.contamination import DATASETS


def test_family_sizes_and_synthetic_separation():
    from collections import Counter
    real = Counter(s['family'] for d in DATASETS[1:] for s in specs(d))
    assert real == {'Primary':18, 'S1':18, 'S2':27, 'S3':54, 'S4':108, 'S5':36}
    assert len(list(specs('synthetic'))) == 87


def test_did_uses_same_split_clean_gap():
    rows = []
    for sid in range(15):
        for method, contaminated, clean in [('irmf_h2_a1.41_conv', .6, .8), ('imf_linear_a1.41', .4, .9)]:
            rows.append(dict(dataset='synthetic', classifier='logreg', feature_set='comp', scenario='shift',
                             condition='impulse2_0.04', method=method, split_id=sid,
                             balanced_accuracy=contaminated+sid*.001,
                             clean_balanced_accuracy=clean+sid*.001, n_train=100, n_test=25))
    # Deliberately shuffle rows to detect positional, rather than split-key, pairing.
    table = pd.DataFrame(rows).sample(frac=1,random_state=22)
    d, train, test = did_vector(table,'synthetic','logreg','comp','shift','impulse2_0.04','imf_linear_a1.41')
    np.testing.assert_allclose(d,.3)
    np.testing.assert_array_equal(test/train, np.repeat(.25,15))
