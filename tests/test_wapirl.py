"""Scientific contract and integration checks for the F whole-wafer baseline."""
import importlib.util
import sys
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods.wapirl import wapirl as w
from shared.evaluation_adapters import PreparedMethod
from shared.evaluation import _check_request

class WaPIRLTests(unittest.TestCase):
    def test_whole_input_no_filter_or_region_crop(self):
        raw = np.ones((52,52), dtype=np.uint8)
        raw[0,0] = 2
        raw[-1,-1] = 2
        raw[2,2] = 3
        x = w.preprocess(raw,52)
        np.testing.assert_array_equal(x[0],raw==2)
        np.testing.assert_array_equal(x[0],x[1])
        np.testing.assert_array_equal(x[2],(raw==1)|(raw==2))
        self.assertEqual(x[0].sum(),2)  # isolated pixels must survive

    def test_rotation_channels_and_reproducibility(self):
        raw = np.ones((1,52,52), dtype=np.uint8);raw[0,0,1]=2
        ds=w.WholeViews(raw,[0],2026)
        a,b=ds[0][1:3]
        self.assertTrue(torch.equal(ds[0][2],b))
        self.assertTrue(any(torch.equal(torch.rot90(a,k,(-2,-1)),b) for k in range(4)))
        self.assertTrue(torch.equal(b[0],b[1]))

    def test_f_model_initialization_exact(self):
        froot=ROOT.parent/'wbm-region-retrieval'
        if not froot.exists():
            self.skipTest('F repository not available')
        sys.path.insert(0,str(froot))
        from experiments.proposed._shared.ef_suite import build_model
        c=w.load_config(ROOT/'methods/wapirl/config.json',require_ks=False)
        w.seed_everything(c['seed']); actual=w.model_for(c)
        w.seed_everything(c['seed']); expected=build_model(c,c['parameters'])
        for key,value in actual.state_dict().items():
            self.assertTrue(torch.equal(value,expected.state_dict()[key]),key)
        self.assertEqual(actual['encoder'].stem[0].stride,(1,1))

    def test_clipped_cosine_and_no_labels(self):
        features=np.array([[1.,0.],[-1.,0.],[.6,.8],[1.,0.]],dtype=np.float32)
        ranker=PreparedMethod('wapirl',np.arange(4),features,{}, {},tie_seed=2026)
        result=ranker.rank(0,[1,2,3])
        np.testing.assert_array_equal(result.ids,[3,2,1])
        np.testing.assert_allclose(result.scores,[1,.6,0])
        self.assertEqual(result.scored_count,3)

    def test_smoke_test_rejected(self):
        with self.assertRaisesRegex(ValueError,'Test requires'):
            _check_request({'method':'wapirl','smoke':True},'test',[1],'global',1,2026)

    def test_training_resume_and_tamper(self):
        # Real optimizer/memory/checkpoint pipeline; tiny model isolates resume semantics.
        from unittest.mock import patch
        torch.set_num_threads(2)
        maps=np.ones((6,8,8),dtype=np.uint8)
        for i in range(6):maps[i,i,1:4]=2
        bundle=SimpleNamespace(maps=maps,train_ids=np.arange(4),valid_ids=np.arange(4,6),signature='test')
        c=w.load_config(ROOT/'methods/wapirl/config.json',require_ks=False)
        c.update(device='cpu',num_workers=0,batch_size=2,input_size=8,smoke=True)
        c['local'].update(epochs=2,projector_dim=4,num_negatives=5)
        def tiny(config):
            return torch.nn.ModuleDict({'encoder':torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(192,8)), 'projector':torch.nn.Linear(8,4)})
        with tempfile.TemporaryDirectory() as tmp, patch.object(w,'model_for',tiny):
            c['output_dir']=str(Path(tmp)/'full');Path(c['output_dir']).mkdir()
            w.train(bundle,c,'run')
            full=w.load_checkpoint(Path(c['output_dir'])/'last.pt')
            w.validate_source(bundle,c,'run')
            c['output_dir']=str(Path(tmp)/'resumed');Path(c['output_dir']).mkdir()
            save=w.save_checkpoint
            def stop_after_one(path,*args,**kwargs):
                save(path,*args,**kwargs)
                if path.name=='last.pt':raise InterruptedError()
            with patch.object(w,'save_checkpoint',stop_after_one), self.assertRaises(InterruptedError):
                w.train(bundle,c,'run')
            w.train(bundle,c,'run',resume=True)
            resumed=w.load_checkpoint(Path(c['output_dir'])/'last.pt')
            for key in full['model']:
                self.assertTrue(torch.equal(full['model'][key],resumed['model'][key]),key)
            self.assertTrue(torch.equal(full['memory'],resumed['memory']))
            w.validate_source(bundle,c,'run')
            with (Path(c['output_dir'])/'best_nce.pt').open('ab') as f:f.write(b'x')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):w.validate_source(bundle,c,'run')
