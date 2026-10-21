"""No training, pretrained downloads or optimizer steps; synthetic data/state only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torchvision import models

from services.c3_quality_assessment.experiment2_training import (
    CLASSES, VERSION, checkpoint_payload, class_weights, configure_model, digest,
    image_views, require_colab_gpu, scores, validate_dataset, validate_partition, compare_results,
)
from services.c3_quality_assessment.build_experiment2_colab import notebook


class Experiment2Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def fixture(self):
        release={'dataset_version':VERSION,'workbook_sha256':'','tasks':{}}
        workbook=self.root/'source/quality_dataset_v2.xlsx'
        workbook.parent.mkdir();workbook.write_bytes(b'synthetic provenance')
        release['workbook_sha256']=digest(workbook)
        for task,classes in CLASSES.items():
            rows=[]
            for i in range(12):
                image=self.root/f'images/P02/P02_S{i+1:03d}_{task}.png'
                image.parent.mkdir(parents=True,exist_ok=True)
                Image.new('RGB',(10,20),(i*10,20,30)).save(image)
                rows.append({'Plant_ID':'P02','Sample_ID':f'S{i+1:03d}','sample_key':f'P02:S{i+1:03d}',
                    'Image_ID':f'{task}{i}','sha256':digest(image),'image_path':image.relative_to(self.root).as_posix(),
                    'label':classes[i%2],'view_type':'Root' if task=='root' else 'Front','status':'usable','exclusion_reason':''})
            frame=pd.DataFrame(rows);frame.to_csv(self.root/f'{task}_usable.csv',index=False)
            release['tasks'][task]={'usable_images':12,'class_counts':frame.label.value_counts().to_dict()}
            folder=self.root/'splits'/task;folder.mkdir(parents=True)
            for fold in range(3):
                split=frame.copy();split['split']=[['test','validation','train'][((i//2)-fold)%3] for i in range(12)]
                split.to_csv(folder/f'fold_{fold+1}_split.csv',index=False)
        (self.root/'release.json').write_text(json.dumps(release))
        self.rehash()

    def rehash(self):
        checks={p.relative_to(self.root).as_posix():digest(p) for p in self.root.rglob('*') if p.is_file() and p.name!='checksums.json'}
        (self.root/'checksums.json').write_text(json.dumps(checks))

    def test_valid_fixed_partitions_and_exact_coverage(self):
        self.fixture();_,frames,splits=validate_dataset(self.root)
        for task in CLASSES:
            self.assertEqual(len(frames[task]),12)
            self.assertEqual(len(splits[task]),3)

    def test_tampered_split_checksum_and_membership_rejected(self):
        self.fixture();p=self.root/'splits/root/fold_1_split.csv'
        d=pd.read_csv(p,keep_default_na=False);d.loc[0,'split']='train';d.to_csv(p,index=False)
        with self.assertRaisesRegex(ValueError,'Checksum'):validate_dataset(self.root)
        self.rehash()
        with self.assertRaisesRegex(ValueError,'OOF coverage'):validate_dataset(self.root)

    def test_duplicate_image_leakage_and_namespace_mismatch(self):
        self.fixture();frame=pd.read_csv(self.root/'root_usable.csv',keep_default_na=False)
        split=pd.read_csv(self.root/'splits/root/fold_1_split.csv',keep_default_na=False)
        frame.loc[2,'sha256']=frame.loc[0,'sha256'];split.loc[2,'sha256']=split.loc[0,'sha256']
        with self.assertRaisesRegex(ValueError,'leakage'):validate_partition(frame,split)
        split.loc[0,'Plant_ID']='P03'
        with self.assertRaisesRegex(ValueError,'mismatch'):validate_partition(frame,split)

    def test_only_intended_parameters_trainable_and_bn_frozen(self):
        model=models.resnet18(weights=None);model.fc=nn.Linear(512,2)
        for architecture in ['A','B']:
            configure_model(model,architecture)
            for name,parameter in model.named_parameters():
                self.assertEqual(parameter.requires_grad,name.startswith('fc.') or (architecture=='B' and name.startswith('layer4.')))
            self.assertTrue(all(not m.training for m in model.modules() if isinstance(m,nn.BatchNorm2d)))
        with self.assertRaises(ValueError):configure_model(model,'C')

    def test_augmentation_deterministic_and_original_view_fixed(self):
        path=self.root/'image.png';Image.new('RGB',(50,100),'green').save(path)
        a,b,c=image_views(path,42),image_views(path,42),image_views(path,43)
        self.assertEqual(tuple(a.shape),(3,3,224,224));self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(a[0],c[0]));self.assertFalse(torch.equal(a[1:],c[1:]))

    def test_weights_and_absent_class_metrics(self):
        weights=class_weights([0,0,0,1]);self.assertTrue(torch.allclose(weights,torch.tensor([2/3,2.])))
        with self.assertRaises(ValueError):class_weights([0,0])
        result=scores(np.array([0,0]),np.array([0,1]),CLASSES['root'])
        self.assertIsNone(result['minority_recall']);self.assertIsNone(result['balanced_accuracy'])

    def test_checkpoint_loads_through_existing_inference_contract(self):
        # Random synthetic weights: no training, downloads or inference forward pass.
        from services.c3_quality_assessment import model_inference
        model=models.resnet18(weights=None);model.fc=nn.Linear(512,2)
        path=self.root/'synthetic.pth'
        torch.save(checkpoint_payload(model,'leaf','B',{'fold':1}),path)
        with patch.object(model_inference,'MODEL_PATH',path),patch.object(model_inference,'DEVICE',torch.device('cpu')):
            loaded,classes=model_inference.load_quality_model()
        self.assertEqual(classes,CLASSES['leaf']);self.assertEqual(loaded.fc.out_features,2)
        self.assertTrue(torch.equal(loaded.fc.weight,model.fc.weight))

    def test_local_training_guard(self):
        with patch('platform.system',return_value='Windows'):
            with self.assertRaisesRegex(RuntimeError,'disabled'):require_colab_gpu()

    def test_notebook_unexecuted_and_compiles(self):
        doc=notebook('a'*64,'b'*64)
        for cell in doc['cells']:
            if cell['cell_type']=='code':
                self.assertIsNone(cell['execution_count']);self.assertEqual(cell['outputs'],[])
                compile(''.join(cell['source']),'cell','exec')

    def test_comparison_rejects_different_splits(self):
        for arch in ['A','B']:
            path=self.root/arch/'root';path.mkdir(parents=True)
            report={'dataset_version':VERSION,'architecture':arch,'workbook_sha256':'same','class_names':CLASSES['root'],
                'preprocessing':{},'config':{},'folds':[{'split_sha256':arch}]}
            (path/'evaluation.json').write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'split hashes'):compare_results(self.root)


if __name__=='__main__':unittest.main()
