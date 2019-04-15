# -*- coding: utf-8 -*-
"""
Module for training sectional attention variants of InferenceNet
"""

from sklearn.metrics import roc_auc_score
import sys 
import copy
import random
from os.path import join, dirname, abspath

# this monstrosity produces the module directory in an environment where this is unpacked
sys.path.insert(0, abspath(join(dirname(abspath(__file__)), '..', '..')))
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

import torch
import numpy as np
import torch.nn as nn
from torch import optim
from evidence_inference.models.model_0 import InferenceNet, _get_y_vec
from evidence_inference.models.section_attn_helper import split_sections, gen_recursive_encoding, gen_histogram
from evidence_inference.models.attention_distributions import pretrain_attention, get_article_attention_weights, prepare_article_attention_target

USE_CUDA = True

    

def pretrain_section_attention(train_Xy, val_Xy, inference_vectorizer, model, criterion, epochs=100, batch_size=1, patience=5):
    best_score = float('-inf'); best_model = None
    epochs_since_improvement = 0
    print("Pre-training attention distribution with {} training examples, {} validation examples".format(len(train_Xy), len(val_Xy)))

    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(train_Xy), batch_size):
            instances = train_Xy[i:i+batch_size]
            articles, indices, labels, titles, Is, Cs, Os = split_sections(instances, inference_vectorizer, False)
            rec = gen_recursive_encoding(instances, inference_vectorizer, titles)
            target = torch.tensor(labels).float()

            if USE_CUDA:
                articles, Is, Cs, Os, target = articles.cuda(), Is.cuda(), Cs.cuda(), Os.cuda(), target.cuda()

            _, attn_weights = model(articles, indices, Is, Cs, Os, batch_size=len(instances), h_dropout_rate=0, recursive_encoding = rec)
            
            attn_weights = attn_weights[0].squeeze()
            optimizer.zero_grad()
            if not (torch.min(attn_weights >= 0).item() == 1 and torch.min(attn_weights <= 1.) == 1):
                print("Error in weights")
            if not (torch.min(target.data >= 0).item() == 1 and torch.min(target.data <= 1.) == 1):
                print("Error in weights")
                
            loss = criterion(attn_weights, target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("Epoch {} Loss for pretrained attention (training set):".format(epoch), epoch_loss)
        with torch.no_grad():
            overall_val_auc, val_loss = 0, 0
            for instances in val_Xy:
                instances = [instances]
                articles, indices, labels, titles, Is, Cs, Os = split_sections(instances, inference_vectorizer, False)
                rec = gen_recursive_encoding(instances, inference_vectorizer, titles)
                target = torch.tensor(labels).float()
    
                if USE_CUDA:
                    articles, Is, Cs, Os = articles.cuda(), Is.cuda(), Cs.cuda(), Os.cuda()
    
                _, attn_weights = model(articles, indices, Is, Cs, Os, batch_size=len(instances), h_dropout_rate=0, recursive_encoding = rec)
                attn_weights = attn_weights[0].squeeze()
                preds = attn_weights.cpu().numpy()
                val_auc, v_loss = roc_auc_score(target, preds), criterion(attn_weights, target.cuda())
                overall_val_auc += val_auc / len(val_Xy)
                val_loss += v_loss / len(val_Xy)
                
            print("Pretraining attention validation loss: {:.3F}, auc: {:.3F}".format(val_loss, overall_val_auc))

            epochs_since_improvement += 1
           
            if overall_val_auc > best_score:
                print("new best model at epoch {}".format(epoch))
                best_score = overall_val_auc
                best_model = copy.deepcopy(model)
                epochs_since_improvement = 0

            if epochs_since_improvement > patience:
                print("Exiting early due to no improvement on validation after {} epochs.".format(patience))
                break
                
    return best_model

def sec_attn_make_preds(nnet, instances,
                        batch_size, inference_vectorizer, 
                        verbose_attn_if_first_batch=False, 
                        gen_big_sections = False,
                        recursive_encoding = False):
    """
    Make preds for the above model.
    """
    y_vec = torch.cat([_get_y_vec(inst['y'], as_vec=False) for inst in instances]).squeeze()
    y_hat_vec = []
    all_sec_weights = []
    all_labels = []
    all_titles = []
    # we batch this so the GPU doesn't run out of memory
    for i in range(0, len(instances), batch_size):
        batch_instances = instances[i:i+batch_size]
        articles, indices, labels, titles, Is, Cs, Os = split_sections(batch_instances, inference_vectorizer, gen_big_sections)
        rec = None
        if recursive_encoding:
            rec = gen_recursive_encoding(instances, inference_vectorizer, titles)
 
        articles, Is, Cs, Os = articles.cuda(), Is.cuda(), Cs.cuda(), Os.cuda()
        y_hat_batch, section_weights = nnet(articles, indices, Is, Cs, Os, batch_size=len(batch_instances), h_dropout_rate=0, recursive_encoding = rec)
        y_hat_vec.append(y_hat_batch)
        all_sec_weights.append(section_weights)
        all_labels.append(labels)
        all_titles.append(titles)

    return y_vec, torch.cat(y_hat_vec, dim=0), all_sec_weights, all_labels, all_titles  

def train_section_attn(ev_inf: InferenceNet, 
                       train_Xy, val_Xy, test_Xy, inference_vectorizer, 
                       epochs=50, batch_size=16, shuffle=True, 
                       gen_big_sections = False, recursive_encoding = True,
                       pretrain_tokens = False, pretrain_sections = False):
    """ 
    Train the section attention model.
    """
    if not shuffle:
        train_Xy.sort(key=lambda x: len(x['article']))
        val_Xy.sort(key=lambda x: len(x['article']))
        test_Xy.sort(key=lambda x: len(x['article']))
        
    print("Using {} training examples, {} validation examples, {} testing examples".format(len(train_Xy), len(val_Xy), len(test_Xy)))

    best_val_model = None
    best_val_f1 = float('-inf')
    if USE_CUDA:
        ev_inf = ev_inf.cuda()
        
    if pretrain_tokens:
        ev_inf = pretrain_attention(train_Xy, val_Xy, ev_inf, 
                                    prepare_article_attention_target, 
                                    get_article_attention_weights, 
                                    criterion = torch.nn.BCELoss(reduction='sum'), 
                                    epochs=100, batch_size=16, 
                                    cuda=True, tokenwise_attention=False,
                                    patience=5, attention_acceptance='auc')
        
    if pretrain_sections:
         ev_inf = pretrain_section_attention(train_Xy, 
                                             val_Xy, 
                                             inference_vectorizer,
                                             ev_inf,
                                             nn.BCELoss(reduction='sum'))
                                   

    optimizer = optim.Adam(ev_inf.parameters())
    criterion = nn.CrossEntropyLoss(reduction='sum')  # sum (not average) of the batch losses.

    val_metrics = {"val_acc": [], "val_p": [], "val_r": [], "val_f1": [], "val_loss": [], 'train_loss': []}
    for epoch in range(epochs):
        if shuffle:
            random.shuffle(train_Xy)

        epoch_loss = 0
        for i in range(0, len(train_Xy), batch_size):
            instances = train_Xy[i:i+batch_size]
            ys = torch.cat([_get_y_vec(inst['y'], as_vec=False) for inst in instances], dim=0)
            articles, indices, labels, titles, Is, Cs, Os = split_sections(instances, inference_vectorizer, gen_big_sections)
            rec = None
            if recursive_encoding:
                rec = gen_recursive_encoding(instances, inference_vectorizer, titles)

            optimizer.zero_grad()
            
            if USE_CUDA:
                articles, Is, Cs, Os = articles.cuda(), Is.cuda(), Cs.cuda(), Os.cuda()
                ys = ys.cuda()
                                
            tags, _ = ev_inf(articles, indices, Is, Cs, Os, batch_size=len(instances), recursive_encoding = rec)
            loss = criterion(tags, ys)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        val_metrics['train_loss'].append(epoch_loss)

        with torch.no_grad():
            val_y, val_y_hat, section_weights, sec_labels, sec_text = sec_attn_make_preds(ev_inf, 
                                                                                          val_Xy, 
                                                                                          batch_size, 
                                                                                          inference_vectorizer, 
                                                                                          verbose_attn_if_first_batch=False, 
                                                                                          gen_big_sections = gen_big_sections, 
                                                                                          recursive_encoding = recursive_encoding)
            
            #import pdb; pdb.set_trace() # Try to get statistics for things. 
            gen_histogram(sec_text, section_weights, sec_labels, epoch)
            
            val_loss = criterion(val_y_hat, val_y.squeeze())
            y_hat = [int(np.argmax(y_i.cpu())) for y_i in val_y_hat]
            
            if USE_CUDA:
                val_y = val_y.cpu()
                val_loss = val_loss.cpu().item()
        
            acc = accuracy_score(val_y, y_hat)
            val_metrics["val_acc"].append(acc)
            val_metrics["val_loss"].append(val_loss)
           
            p, r, f1, _ = precision_recall_fscore_support(val_y, y_hat, labels=None, beta=1, average='macro', pos_label=1, warn_for=('f-score',), sample_weight=None)
            val_metrics["val_f1"].append(f1)
            val_metrics["val_p"].append(p)
            val_metrics["val_r"].append(r)

            if f1 > best_val_f1:
                best_val_f1 = f1
                best_val_model = copy.deepcopy(ev_inf)

            print("epoch {}. train loss: {}; val loss: {}; val acc: {:.3f}".format(
                epoch, epoch_loss, val_loss, acc))
       
            print(classification_report(val_y, y_hat))
            print("val macro f1: {}".format(f1))
            print("\n\n")

    val_metrics['best_val_f1'] = best_val_f1
    
    #### TEST RESULTS #### 
    test_y, test_y_hat, section_weights, sec_labels, sec_text = sec_attn_make_preds(ev_inf, 
                                                                                    test_Xy, 
                                                                                    batch_size, 
                                                                                    inference_vectorizer, 
                                                                                    verbose_attn_if_first_batch=False, 
                                                                                    gen_big_sections = gen_big_sections, 
                                                                                    recursive_encoding = recursive_encoding)
        
    gen_histogram(sec_text, section_weights, sec_labels, epoch)
            
    test_loss = criterion(test_y_hat, test_y.squeeze())
    y_hat = [int(np.argmax(y_i.cpu())) for y_i in test_y_hat]
    
    if USE_CUDA:
        test_y = test_y.cpu()
        test_loss = test_loss.cpu().item()

    acc = accuracy_score(test_y, y_hat)  
    p, r, f1, _ = precision_recall_fscore_support(test_y, y_hat, labels=None, beta=1, average='macro', pos_label=1, warn_for=('f-score',), sample_weight=None)

    print("Test epoch {}. train loss: {}; val loss: {}; val acc: {:.3f}".format(
        epoch, epoch_loss, test_loss, acc))
   
    print(classification_report(test_y, y_hat))
    print("Test Macro F1: {}".format(f1))
    print("\n\n")

    return best_val_model, inference_vectorizer, train_Xy, val_Xy, val_metrics
