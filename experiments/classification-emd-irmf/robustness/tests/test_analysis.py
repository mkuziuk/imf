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


def test_common_frame_rescoring_handles_equal_counts_different_ids(monkeypatch, tmp_path):
    from robustness.analysis import common_impulse_gauss_vector
    truth=np.array(['a','b','a','b','a','b'])
    ids=np.arange(6)
    for method in ('irmf_h2_a1.41_conv','imf_linear_a1.41'):
        folder=tmp_path/method
        folder.mkdir()
        robust=method.startswith('irmf')
        np.savez(folder/'predictions.npz',
                 clean__ids=ids, clean__pred=truth,
                 **{'impulse2_0.04__ids':ids[:4],
                    'impulse2_0.04__pred':truth[:4] if robust else truth[:4][::-1],
                    'gauss_0.04__ids':ids[2:],
                    'gauss_0.04__pred':truth[2:][::-1] if robust else truth[2:]})
    monkeypatch.setattr('robustness.classification.checkpoint',lambda d,s,c,m,f,i:tmp_path/m)
    monkeypatch.setattr('robustness.analysis.load_info',lambda d:{'primary_task':'signal','tasks':{'signal':'y_signal'}})
    monkeypatch.setattr('robustness.analysis.load_meta',lambda d:pd.DataFrame({'y_signal':truth}))
    rows=[dict(dataset='synthetic',classifier='logreg',feature_set='comp',scenario='shift',
          condition='impulse2_0.04',method='irmf_h2_a1.41_conv',split_id=i,n_train=100) for i in range(15)]
    d,train,test=common_impulse_gauss_vector(pd.DataFrame(rows),'synthetic','logreg','comp',
                                            'impulse2_0.04','imf_linear_a1.41')
    np.testing.assert_array_equal(d,np.full(15,2.))
    np.testing.assert_array_equal(test,np.full(15,2))
