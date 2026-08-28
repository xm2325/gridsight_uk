import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from component_mask_metrics import pole_end_candidate,mask_matches,decode_masks
from resume_component_masks import training_audit


class ComponentMaskTests(unittest.TestCase):
    def test_duplicate_final_callback_is_not_an_extra_training_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'training').mkdir()
            events=[{'epoch':i,'losses':[1.]} for i in range(1,21)]
            (root/'results.json').write_text(json.dumps({'status':'FAILED','predictions':[],'epoch_losses':events+[events[-1]]}))
            (root/'training/results.csv').write_text('epoch,time,train/seg_loss\n'+''.join(f'{i},{i},1\n' for i in range(1,21)))
            _,rows=training_audit(root);self.assertEqual(len(rows),20)
            (root/'training/results.csv').write_text('epoch,time,train/seg_loss\n1,1,1\n')
            with self.assertRaises(AssertionError):training_audit(root)

    def numpy(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('Optional raster checks require NumPy')
        return np

    def test_pole_end_is_unscored_and_abstains_without_context(self):
        np=self.numpy();mask=np.zeros((110,110),bool);mask[10:100,45:55]=True
        self.assertEqual(pole_end_candidate(mask,[])['status'],'unknown')
        r=pole_end_candidate(mask,[[20,12,80,20]])
        self.assertEqual(r['status'],'geometry_candidate');self.assertLess(r['point'][1],20)
        self.assertIsNone(r['score']);self.assertFalse(r['supervised_pole_top'])
        self.assertEqual(pole_end_candidate(mask,[[20,12,80,20],[20,90,80,98]])['status'],'unknown')
        mask=np.zeros((110,110),bool);mask[45:55,10:100]=True
        r=pole_end_candidate(mask,[[12,20,20,80]])
        self.assertLess(r['point'][0],20)

    def test_clipped_pole_end_is_not_accepted(self):
        np=self.numpy();mask=np.zeros((100,100),bool);mask[:,45:55]=True
        self.assertIn('truncated',pole_end_candidate(mask,[[20,0,80,10]])['reason'])

    def test_mask_matching_counts_duplicates_and_missing_classes(self):
        np=self.numpy();mask=np.ones((10,10),bool)
        predictions=[{'prediction_index':i,'class_id':0,'score':.9-i*.1} for i in range(2)]
        refs=[{'class_id':0},{'class_id':1}]
        result=mask_matches(predictions,[mask,mask],refs,[mask,mask])
        self.assertEqual((result['tp'],result['fp'],result['fn']),(1,1,1))
        raw={'mask_shape':np.array([0,10,10]),'mask_bits':np.zeros((0,13),np.uint8)}
        self.assertEqual(decode_masks(raw).shape,(0,10,10))


if __name__=='__main__':unittest.main()
