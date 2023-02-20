#!/usr/bin/env python
# coding: utf-8

import os
import time
import argparse
import tarfile
import gzip
import fileinput
import sys
import random
from typing import Union
from mistletoe import Document, HTMLRenderer
from bs4 import BeautifulSoup
import requests
import shutil
import sentencepiece as spm

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(FILE_DIR, "models")
DATA_DIR = os.path.join(FILE_DIR, "model_utils")


def get_dataset_link(language: str) -> str:
    with open(os.path.join(DATA_DIR, "MonolingualData.md"), "r", encoding="utf-8") as fin:
        with HTMLRenderer() as renderer:
            doc = Document(fin)
            rendered = renderer.render(doc)
            soup = BeautifulSoup(rendered, features="html.parser")
    for available_language in soup.find_all("ul")[-1].find_all("a"):
        if available_language.text == language:
            return available_language["href"]
    raise ValueError('A very specific bad thing happened')

def download_dataset(language: str, dataset_link: str):
    try:
        with requests.get(dataset_link, stream=True) as r:
            with open(os.path.join(MODEL_DIR, f"{language}.tar"), "wb") as f:
                shutil.copyfileobj(r.raw, f)
        return
    except Exception as e:
        raise e



def extract_dataset(language: str):
    os.mkdir(os.path.join(MODEL_DIR, language))
    with tarfile.open(os.path.join(MODEL_DIR, f"{language}.tar"), "r") as f:
        for member in f.getmembers():
            if "txt.gz" in os.path.split(member.name)[1]:
                f.extract(member, os.path.join(MODEL_DIR, language))
    return


def concatenate_dataset(language: str) -> int:
    if not os.path.exists(os.path.join(MODEL_DIR, language, "data", "release")):
        raise ValueError("Language not yet supported")
    data_release_path = os.path.join(MODEL_DIR, language, "data", "release")
    release_date = os.listdir(data_release_path)[0]
    data_path = os.path.join(data_release_path, release_date, language)
    with open(os.path.join(MODEL_DIR, f"{language}.txt"), "wb+") as concat_f:
        for source in sorted(os.listdir(data_path)):
            with gzip.open(os.path.join(data_path, source), "rb") as f:
                shutil.copyfileobj(f, concat_f)

    for line in fileinput.FileInput(
            os.path.join(
                MODEL_DIR,
                f"{language}.txt"),
            encoding="utf-8",
            inplace=True):
        if not line.isspace():
            sys.stdout.write(line)
    
    with open(os.path.join(MODEL_DIR, f"{language}.txt"), "r", encoding="utf-8") as f:
        num_lines = sum(1 for line in f)
    if num_lines <= 5:
        raise ValueError("Language not yet supported")
    return num_lines

def train_tokenizer(language: str, n_tokens: int):
    try:
        spm.SentencePieceTrainer.train(
            input=os.path.join(MODEL_DIR, f"{language}.txt"),
            model_prefix=os.path.join(MODEL_DIR, language),
            vocab_size=n_tokens,
            model_type="unigram",
            character_coverage=0.9995,
            seed_sentencepiece_size=1000000,
            shuffle_input_sentence=True,
            input_sentence_size=2000000,
        )
    except Exception as e:
        if "<=" in str(e):
            n_tokens = int(str(e).split("<= ")[1].replace(".", ""))
        elif ">=" in str(e):
            n_tokens = int(str(e).split(">= ")[1].replace(".", ""))
        spm.SentencePieceTrainer.train(
            input=os.path.join(MODEL_DIR, f"{language}.txt"),
            model_prefix=os.path.join(MODEL_DIR, language),
            vocab_size=n_tokens,
            model_type="unigram",
            character_coverage=0.9995,
            seed_sentencepiece_size=1000000,
            shuffle_input_sentence=True,
            input_sentence_size=2000000,
        )
    finally:
        return
    
def clean(language: str, n_tokens: int):
    if os.path.exists(os.path.join(MODEL_DIR, f"{language}.model")):
        os.rename(
            os.path.join(
                MODEL_DIR, f"{language}.model"), os.path.join(
                MODEL_DIR, f"{language}-{n_tokens}.spm"))
    if os.path.exists(os.path.join(MODEL_DIR, f"{language}.vocab")):
        os.remove(os.path.join(MODEL_DIR, f"{language}.vocab"))
    if os.path.exists(os.path.join(MODEL_DIR, f"{language}.txt")):
        os.remove(os.path.join(MODEL_DIR, f"{language}.txt"))
    if os.path.exists(os.path.join(MODEL_DIR, f"{language}.tar")):
        os.remove(os.path.join(MODEL_DIR, f"{language}.tar"))
    if os.path.exists(os.path.join(MODEL_DIR, language)):
        shutil.rmtree(os.path.join(MODEL_DIR, language))
    return


def train_model(language: str, n_tokens: int = 50000, verbose: bool=True):
    try:
        dataset_link = get_dataset_link(language)
    except Exception as e:
        raise e

    if verbose:
        sys.stdout.write(f"Downloading dataset\n")
    try:
        download_dataset(language, dataset_link)
    except Exception as e:
        raise e

    if verbose:
        sys.stdout.write(f"Extracting dataset\n")
    extract_dataset(language)

    if verbose:
        sys.stdout.write(f"Concatenating dataset\n")
    try:
        num_lines = concatenate_dataset(language)
    except Exception as e:
        clean(language, n_tokens)
        raise e

    if verbose:
        sys.stdout.write(f"Training model\n")
    train_tokenizer(language, n_tokens)
    clean(language, n_tokens)

    return


def main():
    sys.stdout.write(f"start_time:\t{time.time()}\n")

    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-L",
        "--language",
        help="language code",
        required=True,
        type=str)
    argParser.add_argument(
        "-N",
        "--n_tokens",
        help="number of tokens",
        required=False,
        default=50000,
        type=int)
    argParser.add_argument(
        "-v",
        "--verbose",
        help="show sentence level scores",
        action="store_true")

    args = argParser.parse_args()
    
    try:
        train_model(args.language, args.n_tokens, args.verbose)
    except Exception as e:
        sys.stdout.write(f"{str(e)}\n")

    sys.stdout.write(f"end_time:\t{time.time()}\n")


if __name__ == "__main__":
    main()
