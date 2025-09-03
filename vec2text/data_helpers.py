import logging
import os
import random
from typing import Dict, List, Optional

import datasets
import torch

from vec2text.run_args import DataArguments
from vec2text.utils import dataset_map_multi_worker, get_num_proc


def retain_dataset_columns(
    d: datasets.Dataset, allowed_columns: List[str]
) -> datasets.Dataset:
    column_names_to_remove = [c for c in d.features if c not in allowed_columns]
    return d.remove_columns(column_names_to_remove)


def load_nq_dpr_corpus(streaming: bool = False) -> datasets.Dataset:
    return datasets.load_dataset("jxm/nq_corpus_dpr", streaming=streaming)


def load_msmarco_corpus(streaming: bool = False) -> datasets.Dataset:
    # has columns ["title", "text"]. only one split ("train")
    dataset_dict = datasets.load_dataset(
        "Tevatron/msmarco-passage-corpus", streaming=streaming
    )
    return dataset_dict["train"]


def create_omi_ex(ex: Dict[str, str]) -> Dict[str, str]:
    ex["text"] = ex["user"]
    return ex


def create_ompi_ex(ex: Dict[str, str]) -> Dict[str, str]:
    ex["user"] = ex["user"].strip()
    ex["system"] = ex["system"].strip()
    ex["text"] = ex["system"] + "\n\n" + ex["user"]
    ex["prefix"] = ex["system"] + "\n\n"
    ex["suffix"] = ex["user"]
    return ex


def get_world_size() -> int:
    try:
        return torch.distributed.get_world_size()
    except (RuntimeError, ValueError):
        return 1


def load_one_million_paired_instructions(streaming: bool = False) -> datasets.Dataset:
    # has only "train" split, and "system" (system prompt)
    # and "user" (user input) columns
    dataset_dict = datasets.load_dataset(
        "wentingzhao/one-million-paired-instructions", streaming=streaming
    )
    d = dataset_dict["train"].map(create_ompi_ex)
    return d


def load_one_million_instructions(streaming: bool = False) -> datasets.Dataset:
    # has only "train" split, and "system" (system prompt)
    # and "user" (user input) columns
    dataset_dict = datasets.load_dataset(
        "wentingzhao/one-million-instructions", streaming=streaming
    )
    d = dataset_dict["train"].map(create_ompi_ex)
    return d


def load_anthropic_toxic_prompts(streaming: bool = False) -> datasets.Dataset:
    d = datasets.load_dataset(
        "wentingzhao/anthropic-hh-first-prompt", streaming=streaming
    )["train"]
    d = d.rename_column("user", "text")
    return d


def load_luar_reddit(streaming: bool = False) -> datasets.Dataset:
    d = datasets.load_dataset(
        "friendshipkim/reddit_eval_embeddings_luar", streaming=streaming
    )
    d = d.rename_column("full_text", "text")
    d = d.rename_column("embedding", "frozen_embeddings")
    return d


def dataset_from_args(data_args: DataArguments) -> datasets.DatasetDict:
    """Loads a dataset from data_args created in `run_args`.

    Supports streaming via DataArguments.streaming to avoid local storage.
    When streaming, train/validation are derived using `.skip()`/`.take()` where needed.
    """
    streaming = getattr(data_args, "streaming", False)
    val_take: int = max(1, int(data_args.max_eval_samples or 1000))

    if data_args.dataset_name == "nq":
        ds = load_nq_dpr_corpus(streaming=streaming)
        if streaming:
            # Expect 'train' and 'dev' splits
            train = ds["train"]
            valid = ds["dev"].take(val_take)
            raw_datasets = datasets.IterableDatasetDict({
                "train": train,
                "validation": valid,
            })
        else:
            raw_datasets = ds
            raw_datasets["validation"] = raw_datasets["dev"]
    elif data_args.dataset_name == "msmarco":
        train_ds = load_msmarco_corpus(streaming=streaming)
        if streaming:
            train = train_ds.skip(val_take)
            valid = train_ds.take(val_take)
            raw_datasets = datasets.IterableDatasetDict({
                "train": train,
                "validation": valid,
            })
        else:
            raw_datasets = train_ds.train_test_split(test_size=0.01)
            raw_datasets["validation"] = raw_datasets["test"]
    elif data_args.dataset_name == "one_million_instructions":
        train_ds = load_one_million_instructions(streaming=streaming)
        if streaming:
            train = train_ds.skip(val_take)
            valid = train_ds.take(val_take)
            raw_datasets = datasets.IterableDatasetDict({
                "train": train,
                "validation": valid,
            })
        else:
            raw_datasets = train_ds.train_test_split(test_size=0.01)
            raw_datasets["validation"] = raw_datasets["test"]
    elif data_args.dataset_name == "one_million_paired_instructions":
        train_ds = load_one_million_paired_instructions(streaming=streaming)
        if streaming:
            train = train_ds.skip(val_take)
            valid = train_ds.take(val_take)
            raw_datasets = datasets.IterableDatasetDict({
                "train": train,
                "validation": valid,
            })
        else:
            raw_datasets = train_ds.train_test_split(test_size=0.01)
            raw_datasets["validation"] = raw_datasets["test"]
    elif data_args.dataset_name == "luar_reddit":
        all_luar_datasets = load_luar_reddit(streaming=streaming)
        # For LUAR, 'queries' is a natural validation split
        valid = all_luar_datasets["queries"]
        if streaming:
            valid = valid.take(val_take)
            raw_datasets = datasets.IterableDatasetDict(
                {"train": all_luar_datasets["candidates"], "validation": valid}
            )
        else:
            raw_datasets = datasets.DatasetDict(
                {
                    "train": all_luar_datasets["candidates"],
                    "validation": valid,
                }
            )
    else:
        raise ValueError(f"unsupported dataset {data_args.dataset_name}")
    return raw_datasets


def load_ag_news_test(streaming: bool = False) -> datasets.Dataset:
    return datasets.load_dataset("ag_news", streaming=streaming)["test"]


def load_xsum_val(col: str, streaming: bool = False) -> datasets.Dataset:
    d = datasets.load_dataset("xsum", streaming=streaming)["validation"]
    d = d.rename_column(col, "text")
    return d


def load_wikibio_val(streaming: bool = False) -> datasets.Dataset:
    d = datasets.load_dataset("wiki_bio", trust_remote_code=True, streaming=streaming)[
        "val"
    ]
    d = d.rename_column("target_text", "text")
    return d


def load_arxiv_val(streaming: bool = False) -> datasets.Dataset:
    d = datasets.load_dataset("ccdv/arxiv-summarization", streaming=streaming)[
        "validation"
    ]
    d = d.rename_column("abstract", "text")
    return d


def load_python_code_instructions_18k_alpaca(streaming: bool = False) -> datasets.Dataset:
    d = datasets.load_dataset(
        "iamtarun/python_code_instructions_18k_alpaca", streaming=streaming
    )["train"]
    d = d.rename_column("instruction", "text")
    return d


def load_beir_corpus(name: str) -> List[str]:
    from beir import util as beir_util
    from beir.datasets.data_loader import GenericDataLoader

    #### Download scifact.zip dataset and unzip the dataset
    beir_datasets_cache_dir = "/home/jxm3/research/retrieval/distractor_exp"

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        name
    )
    out_dir = os.path.join(beir_datasets_cache_dir, "datasets")
    data_path = beir_util.download_and_unzip(url, out_dir)

    # Limit each corpus to first 100k documents.
    MAX_N = 100_000

    if name == "cqadupstack":
        full_corpus = []
        for folder in [
            "android",
            "english",
            "gaming",
            "gis",
            "mathematica",
            "physics",
            "programmers",
            "stats",
            "tex",
            "unix",
            "webmasters",
            "wordpress",
        ]:
            corpus, _queries, _qrels = GenericDataLoader(
                data_folder=os.path.join(data_path, folder)
            ).load(split="test")
            full_corpus.extend([k["text"] for k in corpus.values()])
        random.shuffle(full_corpus)
        return full_corpus[:MAX_N]
    else:
        corpus, _queries, _qrels = GenericDataLoader(data_folder=data_path).load(
            split="test"
        )
        corpus = [k["text"] for k in corpus.values()]
        return corpus[:MAX_N]


def load_beir_dataset(name: str) -> datasets.Dataset:
    cache_path = (
        datasets.config.HF_DATASETS_CACHE
    )  # something like /home/jxm3/.cache/huggingface/datasets
    dataset_path = os.path.join(cache_path, "emb_inv_beir", name)
    # print(f"loading BEIR dataset: {name}")
    if os.path.exists(dataset_path):
        logging.info("Loading BEIR dataset %s path %s", dataset_path)
        dataset = datasets.load_from_disk(dataset_path)
    else:
        logging.info(
            "Loading BEIR dataset %s from JSON (slow) at path %s", dataset_path
        )
        corpus = load_beir_corpus(name=name)
        dataset = datasets.Dataset.from_list([{"text": t} for t in corpus])
        os.makedirs(os.path.join(cache_path, "emb_inv_beir"), exist_ok=True)
        dataset.save_to_disk(dataset_path)
        logging.info("Saved BEIR dataset as HF path %s", dataset_path)
    return dataset


def load_beir_datasets() -> datasets.DatasetDict:
    all_beir_datasets = [
        ####### public datasets #######
        "arguana",
        "climate-fever",
        "cqadupstack",
        "dbpedia-entity",
        "fever",
        "fiqa",
        "hotpotqa",
        "msmarco",
        "nfcorpus",
        "nq",
        "quora",
        "scidocs",
        "scifact",
        "trec-covid",
        "webis-touche2020",
        ####### private datasets #######
        "signal1m",
        "trec-news",
        "robust04",
        "bioasq",
    ]
    return datasets.DatasetDict({k: load_beir_dataset(k) for k in all_beir_datasets})


def load_standard_val_datasets(streaming: bool = False, max_eval_samples: Optional[int] = None) -> datasets.DatasetDict:
    """Loads a pre-defined set of standard val datasets."""
    d = {
        "ag_news": load_ag_news_test(streaming=streaming),
        "anthropic_toxic_prompts": load_anthropic_toxic_prompts(streaming=streaming),
        "arxiv": load_arxiv_val(streaming=streaming),
        "python_code_alpaca": load_python_code_instructions_18k_alpaca(
            streaming=streaming
        ),
        # "xsum_doc": load_xsum_val("document", streaming=streaming),
        # "xsum_summ": load_xsum_val("summary", streaming=streaming),
        "wikibio": load_wikibio_val(streaming=streaming),
    }
    d = {k: retain_dataset_columns(v, ["text"]) for k, v in d.items()}
    if streaming and max_eval_samples:
        # Limit samples from each stream
        d = {k: v.take(max_eval_samples) for k, v in d.items()}

    return datasets.DatasetDict(d)
