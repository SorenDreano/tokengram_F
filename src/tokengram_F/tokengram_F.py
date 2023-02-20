#!/usr/bin/env python
# coding: utf-8

import os
import sys
import math
import unicodedata
import argparse
import io
from collections import defaultdict
import time
import string
from typing import List, Tuple, Union, DefaultDict
import sentencepiece as spm
from .train_model import train_model
from .train_model import MODEL_DIR


def separate_characters(line: str) -> List[str]:
    return list(line.strip().replace(" ", ""))

def separate_punctuation(line: str, 
                         tokenizer: spm.SentencePieceProcessor) -> List[str]:
    words = line.strip().split()
    tokenized = []
    for w in words:
        if len(w) == 1:
            tokenized.append(w)
        else:
            last_char = w[-1]
            first_char = w[0]
            if last_char in string.punctuation:
                tokenized += [w[:-1], last_char]
            elif first_char in string.punctuation:
                tokenized += [first_char, w[1:]]
            else:
                tokenized.append(w)

    return tokenizer.encode(" ".join(tokenized), out_type=str)


def n_gram_counts(word_list: List[str], 
                  order: int) -> DefaultDict[int, DefaultDict[Tuple[str, ...], float]]:
    counts = defaultdict(lambda: defaultdict(float)) # type: DefaultDict[int, DefaultDict[Tuple[str, ...], float]]
    n_words = len(word_list)
    for i in range(n_words):
        for j in range(1, order + 1):
            if i + j <= n_words:
                n_gram = tuple(word_list[i:i + j])
                counts[j - 1][n_gram] += 1

    return counts


def n_gram_matches(ref_n_grams: DefaultDict[int, DefaultDict[Tuple[str, ...], float]], 
                   hyp_n_grams: DefaultDict[int, DefaultDict[Tuple[str, ...], float]]) -> Tuple[DefaultDict[int, float], DefaultDict[int, float], DefaultDict[int, float]]:
    matching_n_gram_count: DefaultDict[int, float] = defaultdict(float)
    total_ref_tokengram_count: DefaultDict[int, float] = defaultdict(float)
    total_hyp_tokengram_count: DefaultDict[int, float] = defaultdict(float)

    for order in ref_n_grams:
        for n_gram in hyp_n_grams[order]:
            total_hyp_tokengram_count[order] += hyp_n_grams[order][n_gram]
        for n_gram in ref_n_grams[order]:
            total_ref_tokengram_count[order] += ref_n_grams[order][n_gram]
            if n_gram in hyp_n_grams[order]:
                matching_n_gram_count[order] += min(
                    ref_n_grams[order][n_gram], hyp_n_grams[order][n_gram])

    return matching_n_gram_count, total_ref_tokengram_count, total_hyp_tokengram_count


def n_gram_prec_rec_F(matching: DefaultDict[int, float], 
                      ref_len: DefaultDict[int, float], 
                      hyp_len: DefaultDict[int, float], 
                      beta: float) -> Tuple[DefaultDict[int, float], DefaultDict[int, float], DefaultDict[int, float]]:
    n_gram_prec: DefaultDict[int, float] = defaultdict(float)
    n_gram_rec: DefaultDict[int, float] = defaultdict(float)
    n_gram_F: DefaultDict[int, float] = defaultdict(float)

    factor = beta**2

    for order in matching:
        if hyp_len[order] > 0:
            n_gram_prec[order] = matching[order] / hyp_len[order]
        else:
            n_gram_prec[order] = 1e-16
        if ref_len[order] > 0:
            n_gram_rec[order] = matching[order] / ref_len[order]
        else:
            n_gram_rec[order] = 1e-16
        denom = factor * n_gram_prec[order] + n_gram_rec[order]
        if denom > 0:
            n_gram_F[order] = (1 + factor) * \
                n_gram_prec[order] * n_gram_rec[order] / denom
        else:
            n_gram_F[order] = 1e-16

    return n_gram_F, n_gram_rec, n_gram_prec

def get_tokenizer(language: str, 
                  n_tokens: int = 50000) -> spm.SentencePieceProcessor:
    if not os.path.exists(os.path.join(
            MODEL_DIR, f"{language}-{n_tokens}.spm")):
        err = train_model(language, True)
        if err:
            sys.exit(err)
    tokenizer = spm.SentencePieceProcessor(
        model_file=os.path.join(
            MODEL_DIR, f"{language}-{n_tokens}.spm"))
    return tokenizer

def get_counts(line: str, 
               tokenizer: spm.SentencePieceProcessor, 
               nw_order: int, 
               nc_order: int
               )-> Tuple[DefaultDict[int, DefaultDict[Tuple[str, ...], float]], DefaultDict[int, DefaultDict[Tuple[str, ...], float]]]:
    tokengram_counts = n_gram_counts(
        separate_punctuation(
            line, tokenizer), nw_order)
    chr_n_gram_counts = n_gram_counts(
        separate_characters(line), nc_order)
    return tokengram_counts, chr_n_gram_counts
    

def initialisation_dict() -> Tuple[DefaultDict[int, float], DefaultDict[int, float], DefaultDict[int, float], DefaultDict[int, float], DefaultDict[int, float], DefaultDict[int, float]]:
    return defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)

def get_best_for_ref(r_line: str, 
                     tokenizer: spm.SentencePieceProcessor, 
                     hyp_tokengram_counts: DefaultDict[int, DefaultDict[Tuple[str, ...], float]], 
                     hyp_chr_n_gram_counts: DefaultDict[int, DefaultDict[Tuple[str, ...], float]],
                     beta: float, 
                     max_F: float, 
                     nw_order: int, 
                     nc_order: int) -> float:
    ref_tokengram_counts, ref_chr_n_gram_counts = get_counts(r_line, tokenizer, nw_order, nc_order)
    
    n_order = float(nw_order + nc_order)

    # number of overlapping n-grams, total number of ref n-grams,
    # total number of hyp n-grams
    matching_tokengram_counts, total_ref_tokengram_count, total_hyp_tokengram_count = n_gram_matches(
        ref_tokengram_counts, hyp_tokengram_counts)
    matching_chr_n_gram_counts, total_ref_chr_n_gram_count, total_hyp_chr_n_gram_count = n_gram_matches(
        ref_chr_n_gram_counts, hyp_chr_n_gram_counts)

    # n-gram f-scores, recalls and precisions
    tokengram_F, tokengram_rec, tokengram_prec = n_gram_prec_rec_F(
        matching_tokengram_counts, total_ref_tokengram_count, total_hyp_tokengram_count, beta)
    chr_n_gram_F, chr_n_gram_rec, chr_n_gram_prec = n_gram_prec_rec_F(
        matching_chr_n_gram_counts, total_ref_chr_n_gram_count, total_hyp_chr_n_gram_count, beta)

    sent_rec = (sum(tokengram_rec.values()) +
                sum(chr_n_gram_rec.values())) / n_order
    sent_prec = (sum(tokengram_prec.values()) +
                    sum(chr_n_gram_prec.values())) / n_order
    sent_F = (sum(tokengram_F.values()) +
                sum(chr_n_gram_F.values())) / n_order

    if sent_F >= max_F:
        max_F = sent_F
        best_tokengram_matching_count = matching_tokengram_counts
        best_tokengram_ref_count = total_ref_tokengram_count
        best_tokengram_hyp_count = total_hyp_tokengram_count
        best_chr_matching_count = matching_chr_n_gram_counts
        best_chr_ref_count = total_ref_chr_n_gram_count
        best_chr_hyp_count = total_hyp_chr_n_gram_count
        
    return max_F

def collect_document(total_tokengram_matching_count: DefaultDict[int, float], 
                     total_tokengram_ref_count: DefaultDict[int, float], 
                     total_tokengram_hyp_count: DefaultDict[int, float],
                     total_chr_n_gram_matching_count: DefaultDict[int, float], 
                     total_chr_n_gram_ref_count: DefaultDict[int, float], 
                     total_chr_n_gram_hyp_count: DefaultDict[int, float],
                     best_tokengram_matching_count: DefaultDict[int, float], 
                     best_tokengram_ref_count: DefaultDict[int, float], 
                     best_tokengram_hyp_count: DefaultDict[int, float],
                     best_chr_matching_count: DefaultDict[int, float], 
                     best_chr_ref_count: DefaultDict[int, float], 
                     best_chr_hyp_count: DefaultDict[int, float],
                     nw_order: int, 
                     nc_order: int):
    # collect document level n_gram counts
    for order in range(nw_order):
        total_tokengram_matching_count[order] += best_tokengram_matching_count[order]
        total_tokengram_ref_count[order] += best_tokengram_ref_count[order]
        total_tokengram_hyp_count[order] += best_tokengram_hyp_count[order]
    for order in range(nc_order):
        total_chr_n_gram_matching_count[order] += best_chr_matching_count[order]
        total_chr_n_gram_ref_count[order] += best_chr_ref_count[order]
        total_chr_n_gram_hyp_count[order] += best_chr_hyp_count[order]
        
    return

def get_total(total_tokengram_matching_count: DefaultDict[int, float], 
              total_tokengram_ref_count: DefaultDict[int, float], 
              total_tokengram_hyp_count: DefaultDict[int, float],
              total_chr_n_gram_matching_count: DefaultDict[int, float], 
              total_chr_n_gram_ref_count: DefaultDict[int, float], 
              total_chr_n_gram_hyp_count: DefaultDict[int, float],
              average_total_F: float,
              beta: float,
              nw_order: int,
              nc_order: int,
              n_sent: int) -> Tuple[float, float, float, float]:
    n_order = float(nw_order + nc_order)
    # total precision, recall and F (aritmetic mean of all n_grams)
    total_tokengram_F, total_tokengram_rec, total_tokengram_prec = n_gram_prec_rec_F(
        total_tokengram_matching_count, total_tokengram_ref_count, total_tokengram_hyp_count, beta)
    total_chr_n_gram_F, total_chr_n_gram_rec, total_chr_n_gram_prec = n_gram_prec_rec_F(
        total_chr_n_gram_matching_count, total_chr_n_gram_ref_count, total_chr_n_gram_hyp_count, beta)

    total_F = (sum(total_tokengram_F.values()) +
               sum(total_chr_n_gram_F.values())) / n_order
    average_total_F = average_total_F / (n_sent + 1)
    total_rec = (sum(total_tokengram_rec.values()) +
                 sum(total_chr_n_gram_rec.values())) / n_order
    total_prec = (sum(total_tokengram_prec.values()) +
                  sum(total_chr_n_gram_prec.values())) / n_order
    
    return total_F, average_total_F, total_rec, total_prec

def compute_sentence_tokengram_F(
        r_line: str,
        h_line: str,
        nw_order: int,
        nc_order: int,
        beta: float,
        tokenizer: spm.SentencePieceProcessor,
        sentence_level_scores: io.TextIOWrapper,
        total_tokengram_matching_count: DefaultDict[int, float],
        total_tokengram_ref_count: DefaultDict[int, float], 
        total_tokengram_hyp_count: DefaultDict[int, float],
        total_chr_n_gram_matching_count: DefaultDict[int, float],
        total_chr_n_gram_ref_count: DefaultDict[int, float],
        total_chr_n_gram_hyp_count: DefaultDict[int, float],
        average_total_F: float,
        n_sent: int) -> float:

    # initialisation of document level scores

    hyp_tokengram_counts, hyp_chr_n_gram_counts = get_counts(h_line, tokenizer, nw_order, nc_order)

    # going through multiple references
    max_F = 0.0
    refs = r_line.split("*#")
    
    for ref in refs:
        max_F = get_best_for_ref(r_line, tokenizer, hyp_tokengram_counts, hyp_chr_n_gram_counts, beta, max_F, nw_order, nc_order)
    # all the references are done

    # write sentence level scores
    if sentence_level_scores:
        sentence_level_scores.write(
            f"{n_sent}::c{nc_order}+w{nw_order}-F{int(beta)}\t{(100*max_F):.4f}\n")
        
    best_tokengram_matching_count, best_tokengram_ref_count, best_tokengram_hyp_count,\
        best_chr_matching_count, best_chr_ref_count, best_chr_hyp_count = initialisation_dict()

    collect_document(total_tokengram_matching_count, total_tokengram_ref_count, total_tokengram_hyp_count,
                                                          total_chr_n_gram_matching_count, total_chr_n_gram_ref_count, total_chr_n_gram_hyp_count,
                                                          best_tokengram_matching_count, best_tokengram_ref_count, best_tokengram_hyp_count,
                                                          best_chr_matching_count, best_chr_ref_count, best_chr_hyp_count, nw_order, nc_order)

    average_total_F += max_F

    # all sentences are done
    
    return average_total_F

def compute_batch_tokengram_F(
        references: Union[List[str], io.TextIOWrapper],
        hypotheses: Union[List[str], io.TextIOWrapper],
        language: str,
        nw_order: int=2,
        nc_order: int=6,
        beta: float=2.0,
        n_tokens: int=50000,
        tokenizer=None,
        sentence_level_scores=None) -> Tuple[float, float, float, float]:
    if not tokenizer:
        tokenizer = get_tokenizer(language, n_tokens)    

    # initialisation of document level scores
    total_tokengram_matching_count, total_tokengram_ref_count, total_tokengram_hyp_count,\
        total_chr_n_gram_matching_count, total_chr_n_gram_ref_count, total_chr_n_gram_hyp_count = initialisation_dict()
    average_total_F = 0.0
    
    for n_sent, (r_line, h_line) in enumerate(zip(references, hypotheses)):
        average_total_F = compute_sentence_tokengram_F(r_line, h_line, nw_order, nc_order, beta, tokenizer, sentence_level_scores, total_tokengram_matching_count, total_tokengram_ref_count, total_tokengram_hyp_count, total_chr_n_gram_matching_count, total_chr_n_gram_ref_count, total_chr_n_gram_hyp_count, average_total_F, n_sent)
        
        

    # all sentences are done

    # total precision, recall and F (aritmetic mean of all n_grams)
    total_F, average_total_F, total_rec, total_prec = get_total(total_tokengram_matching_count, total_tokengram_ref_count, total_tokengram_hyp_count,
                                                                total_chr_n_gram_matching_count, total_chr_n_gram_ref_count, total_chr_n_gram_hyp_count,
                                                                average_total_F, beta, nw_order, nc_order, n_sent)

    return total_F, average_total_F, total_rec, total_prec
    
def compute_file_tokengram_F(
        reference: str,
        hypothesis: str,
        language: str,
        nw_order: int=2,
        nc_order: int=6,
        beta: float=2.0,
        n_tokens: int=50000,
        tokenizer=None,
        sentence_level_scores=None) -> Tuple[float, float, float, float]:

    with open(reference, "r", encoding="utf-8") as references, \
            open(hypothesis, "r", encoding="utf-8") as hypotheses:
        total_F, average_total_F, total_rec, total_prec = compute_batch_tokengram_F(references, hypotheses, language, nw_order, nc_order, beta, n_tokens, tokenizer, sentence_level_scores)

    return total_F, average_total_F, total_rec, total_prec


def main():
    sys.stdout.write(f"start_time:\t{int(time.time())}\n")
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-R",
        "--reference",
        help="reference translation",
        required=True,
        type=str)
    argParser.add_argument(
        "-H",
        "--hypothesis",
        help="hypothesis translation",
        required=True,
        type=str)
    argParser.add_argument(
        "-L",
        "--language",
        help="language code",
        required=True,
        type=str)
    argParser.add_argument(
        "-nc",
        "--nc_order",
        help="character n-gram order (default=6)",
        type=int,
        default=6)
    argParser.add_argument(
        "-nw",
        "--nw_order",
        help="word n-gram order (default=2)",
        type=int,
        default=2)
    argParser.add_argument(
        "-b",
        "--beta",
        help="beta parameter (default=2)",
        type=float,
        default=2.0)
    argParser.add_argument(
        "-N",
        "--n_tokens",
        help="number of tokens",
        type=int,
        default=50000)
    argParser.add_argument(
        "-s",
        "--sent",
        help="show sentence level scores",
        action="store_true")

    args = argParser.parse_args()

    sentence_level_scores = None
    if args.sent:
        sentence_level_scores = sys.stdout  # Or stderr?

    total_F, average_total_F, total_prec, total_rec = compute_file_tokengram_F(
        args.reference, args.hypothesis, args.language, args.nw_order, args.nc_order, args.beta, args.n_tokens, None, sentence_level_scores)

    sys.stdout.write(
        f"c{args.nc_order}+w{args.nw_order}-F{int(args.beta)}\t{(100*total_F):.4f}\n")
    sys.stdout.write(
        f"c{args.nc_order}+w{args.nw_order}-avgF{int(args.beta)}\t{(100*average_total_F):.4f}\n")

    sys.stdout.write(f"end_time:\t{int(time.time())}\n")


if __name__ == "__main__":
    main()
