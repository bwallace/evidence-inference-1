# -*- coding: utf-8 -*-
"""
Utilities for extracting/processing sections for attention.
"""
import sys
import copy
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath
from sklearn.metrics import roc_auc_score


# this monstrosity produces the module directory in an environment where this is unpacked
sys.path.insert(0, abspath(join(dirname(abspath(__file__)), '..', '..')))

import torch
import numpy as np
from evidence_inference.models.utils import PaddedSequence
from evidence_inference.preprocess.preprocessor import SimpleInferenceVectorizer as SimpleInferenceVectorizer

USE_CUDA = True

def gen_histogram(titles, preds, labels, name = 0):
    """ 
    Generates a histogram. 
    """
    ### PRINT AUC...
    flat_preds = []
    flat_labels = []
    for i in range(len(preds)):
        flat_preds += list(preds[i][0].squeeze().cpu().numpy())
        flat_labels += labels[i]
        
    auc = roc_auc_score(flat_labels, flat_preds)
    print("AUC SCORE: {}".format(auc))
    
    d_label = {}
    d_pred  = {}
    for i in range(len(titles)):
        batch = titles[i][0]
        for j in range(len(batch)):
            t = ".".join(batch[j].split(".")[:2]).lower()
            pred = preds[i][0].squeeze()[j].data.tolist()
            label = labels[i][j]
            if t in d_label:
                d_label[t] += label
                d_pred[t] += pred
            else:
                d_label[t] = label
                d_pred[t]  = pred
                
    # DICTIONARY TO ARRAYS            
    x_axis  = []
    y_label = []
    y_pred  = []
    for t in d_label.keys():
        if d_pred[t] >= 20 or d_label[t] >= 20:
            x_axis.append(t)
            y_pred.append(d_pred[t])
            y_label.append(d_label[t])
    
    if len(x_axis) == 0:
        for t in d_label.keys():
            if d_pred[t] >= 5 or d_label[t] >= 5:
                x_axis.append(t)
                y_pred.append(d_pred[t])
                y_label.append(d_label[t])
        
     
    # PLOT 
    fig, ax = plt.subplots()
    bar_width = 0.35
    index     = np.asarray(range(len(x_axis)))
    
    ax.bar(index, y_pred, bar_width, color='b', label='Preds')
    ax.bar(index + bar_width, y_label, bar_width, color='r', label='Labels')
    
    ax.set_xlabel('Section Type')
    ax.set_ylabel('Overall Score')
    ax.set_title('Section Attention Weights')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(x_axis, rotation='vertical')
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(str(name) + "_" + 'epochs.png')
   

def gen_recursive_encoding(instances, inference_vectorizer, section_titles):
    """
    Gives us a dict that allows us the information to do attention from 
    subsections -> sections. 
    """
    big_articles, big_indices, _, _, _, _, _ = split_sections(instances, inference_vectorizer, big_sections=True)
    sec_aggr = []
    old_titles = section_titles[0]
    agg_st, agg_end = 0, 1
        
    last_mask = ".".join(old_titles[0].split(".")[:2])

    for i in range(1, len(old_titles)):
        this_mask = ".".join(old_titles[i].split(".")[:2])
        if this_mask == last_mask:
            agg_end  += 1
        else:
            sec_aggr.append(agg_end - agg_st)
            
            # reset
            last_mask = this_mask
            agg_st  = 0
            agg_end = 1 
            
    sec_aggr.append(agg_end - agg_st)
    
    return {'big_sections': big_articles, 'section_splits': sec_aggr}

def interval_overlap(a, b):    
    return max(0, min(a[1], b[1]) - max(a[0], b[0])) != 0

def gen_big_sections(info):
    """ Aggregate subsections into big sections. """
    new_titles = []
    new_ss     = []
    old_titles = info['section_titles']
    old_ss     = info['section_splits']
    if len(old_titles) == 0:
        return info 
    
    last_mask = ".".join(old_titles[0].split(".")[:2])
    agg_st    = 0
    agg_end   = old_ss[0]
    for i in range(1, len(old_titles)):
        this_mask = ".".join(old_titles[i].split(".")[:2])
        if this_mask == last_mask:
            agg_end  += old_ss[i]
        else:
            new_ss.append(agg_end - agg_st)
            new_titles.append(last_mask)
            
            # reset
            last_mask = this_mask
            agg_st  = agg_end
            agg_end = agg_st + old_ss[i] # the last end + length of this array
            
    # last one
    new_ss.append(agg_end - agg_st)
    new_titles.append(this_mask)
        
    info['section_titles'] = new_titles
    info['section_splits'] = new_ss 
    return info

def split_sections(instances, inference_vectorizer, big_sections = False):
    """ Split into sections. If big_sections = False, use subsections, else use big sections. """
    unk_idx = int(inference_vectorizer.str_to_idx[SimpleInferenceVectorizer.PAD])
    Is, Cs, Os = [PaddedSequence.autopad([torch.LongTensor(inst[x]) for inst in instances], batch_first=True, padding_value=unk_idx) for x in ['I', 'C', 'O']]
    indices  = []
    sections = [] 
    section_titles = []
    for i in range(len(instances)):
        info = instances[i]
        if big_sections:
            info = gen_big_sections(info)
        
        ss  = info['section_splits']
        art = info['article']
        evidence_labels = info['evidence_spans']
        section_labels = []
        section_titles.append(info['section_titles'])
        start = 0
        new_added = 0
        
        for s in ss:
            tmp = art[s:start + s]
            is_evid = False
            for labels in evidence_labels:
                is_evid = is_evid or interval_overlap([start, start+s], labels)
            
            if is_evid: 
                section_labels.append(1)
            else:
                section_labels.append(0)
                
            if len(tmp) == 0:
                tmp = [unk_idx]
            sections.append(tmp)
            start += s
            new_added += 1
        
        indices.append(new_added)
    
    
    # cap number of sections...
    inst = [torch.LongTensor(inst) for inst in sections]
    pad_sections = PaddedSequence.autopad(inst, batch_first=True, padding_value=unk_idx)
    return pad_sections, indices, section_labels, section_titles, Is, Cs, Os