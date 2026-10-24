#!/usr/bin/env python3
"""CPU semantic checks for FT/AdaLN and fair, resumable SFM training."""
import ast
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from turbovla.models.action_head import FTHistoryEncoder, TurboVLAActionHead
from turbovla.models.configuration import ActionHeadConfig, FTHistoryConfig, TurboVLAConfig
from turbovla.training.sfm_utils import TaskBalancedSampler, masked_l1_sum, normalize_images


class TrainingSemantics(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(5)

    def head(self, width=13, gru=True, adaln=False):
        return TurboVLAActionHead(ActionHeadConfig(state_dim=width, horizon=5, num_layers=2,
            dropout=0, initialization_seed=42), 32, 4, 64,
            FTHistoryConfig(enabled=gru, adaln=adaln))

    def test_gru_matches_original_without_double_normalizing(self):
        source = ROOT / 'experiments/default_sfm/preprocessing_audit/turbovla_training_source_snapshot/training_snapshot/jack/turbovla/models/action_head.py'
        node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'FTHistoryEncoder')
        scope = dict(torch=torch, nn=nn, FTHistoryConfig=FTHistoryConfig)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
        mean = torch.tensor([.2, -.1, .3, .001, -.002, .04])
        std = torch.tensor([.4, .5, 3., .006, .003, .06])
        old = scope['FTHistoryEncoder'](SimpleNamespace(history_length=50,input_dim=6,hidden_dim=128,
            num_layers=1,mean=mean.tolist(),std=std.tolist(),std_floor=.01),32)
        new = FTHistoryEncoder(FTHistoryConfig(enabled=True),32)
        new.load_state_dict({k:v for k,v in old.state_dict().items() if k not in ('mean','scale')})
        raw = torch.randn(3,50,6)
        torch.testing.assert_close(new((raw-mean)/std.clamp_min(.01)), old(raw), rtol=0,atol=0)
        # Calling another window must not carry recurrent state into the first.
        first = new(torch.zeros(3,50,6))
        new(torch.randn(3,50,6))
        torch.testing.assert_close(first,new(torch.zeros(3,50,6)),rtol=0,atol=0)

    def test_zero_adaln_matches_gru_and_has_gradients(self):
        base, adaptive = self.head(), self.head(adaln=True)
        base.eval(); adaptive.eval()
        for key,value in base.state_dict().items():
            torch.testing.assert_close(value,adaptive.state_dict()[key],rtol=0,atol=0)
        tokens,state,ft = torch.randn(3,9,32),torch.randn(3,13),torch.randn(3,50,6)
        torch.testing.assert_close(base(tokens,state,ft),adaptive(tokens,state,ft),rtol=1e-5,atol=1e-6)
        output=adaptive(tokens,state,ft)
        output.square().mean().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in adaptive.ft_history_encoder.parameters()),0)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in adaptive.decoder.adaln_modulations.parameters()),0)

    def test_common_decoder_initialization_independent_of_input_width(self):
        heads=[self.head(width=w,gru=g,adaln=a) for w,g,a in [(313,False,False),(13,True,False),(9,True,True)]]
        for key,value in heads[0].decoder.state_dict().items():
            for head in heads[1:]: torch.testing.assert_close(value,head.decoder.state_dict()[key],rtol=0,atol=0)
        for key,value in heads[1].ft_history_encoder.state_dict().items():
            torch.testing.assert_close(value,heads[2].ft_history_encoder.state_dict()[key],rtol=0,atol=0)

    def test_task_sampling_not_frame_length_weighted_and_resume_exact(self):
        shards=[dict(task='long',frames=900),dict(task='short',frames=100)]
        sampler=TaskBalancedSampler(shards,64,42,64000)
        count=sum(index>=900 for batch in sampler for index in batch)
        self.assertLess(abs(count/64000-.5),.015)
        iterator=iter(sampler); first=next(iterator); next(iterator)  # prefetched, not consumed
        sampler.consume(len(first))
        resumed=TaskBalancedSampler(shards,64,42,64000)
        resumed.load_state_dict(sampler.state_dict())
        self.assertEqual(next(iter(resumed)),sampler.batch_at(64))
        self.assertEqual(resumed.batch_at(128),TaskBalancedSampler(shards,64,42,64000).batch_at(128))

    def test_accumulation_uses_full_batch_valid_element_denominator(self):
        model=nn.Linear(4,7); other=copy.deepcopy(model)
        x=torch.randn(5,5,4); target=torch.randn(5,5,7)
        mask=torch.ones(5,5,dtype=torch.bool); mask[1,1:]=False;mask[4,3:]=False
        target[~mask]=float('nan')
        denominator=int(mask.sum())*7
        (masked_l1_sum(model(x),target,mask)/denominator).backward()
        for start in range(0,5,2):
            (masked_l1_sum(other(x[start:start+2]),target[start:start+2],mask[start:start+2])/denominator).backward()
        for p,q in zip(model.parameters(),other.parameters()): torch.testing.assert_close(p.grad,q.grad)

    def test_augmentation_preserves_black_padding_and_shared_views(self):
        images=torch.zeros(3,2,3,224,224,dtype=torch.uint8)
        images[:,:,:,28:196]=120
        settings=dict(mean=[0,0,0],std=[1,1,1],color_jitter=dict(brightness=.2,contrast=.2,saturation=.2,hue=.02,padding_tblr=[28,28,0,0]))
        out=normalize_images(images,settings,True)
        self.assertTrue(torch.equal(out[:,0],out[:,1]))
        self.assertEqual(float(out[...,:28,:].abs().sum()+out[...,196:,:].abs().sum()),0)

    def test_ft_contract_rejects_double_normalization(self):
        with self.assertRaises(ValueError):
            TurboVLAConfig(ft_history=FTHistoryConfig(enabled=True,input_normalization='standard'))
        with self.assertRaises(ValueError):
            TurboVLAConfig(ft_history=FTHistoryConfig(enabled=False,adaln=True))


if __name__=='__main__': unittest.main()
