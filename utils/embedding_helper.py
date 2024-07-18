"""
Resuable helper functions to generate embeddings
"""
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from datasets import Dataset
from tqdm import tqdm
import faiss
import pandas as pd
from types import SimpleNamespace


def load_faiss_data(CFG: SimpleNamespace):
    """Loads all the FAISS index's and maps

    All of these are made from ./code/4-generate-RAG-dbs/*

    Args:
        CFG (SimpleNamespace): CFG loaded from model_helper.load_cfg
    """
    # ADVISORY DATA
    adv_embeddings = faiss.read_index(
        f"{CFG.paths.rag_db}{CFG.data.rag.faiss_advisory}")
    adv_map = pd.read_csv(
        f"{CFG.paths.rag_db}{CFG.data.rag.faiss_advisory_map}")

    # TP GIT-HUNK DATA
    tp_embeddings = faiss.read_index(
        f"{CFG.paths.rag_db}{CFG.data.rag.faiss_tp_git_hunk}")
    tp_map = pd.read_csv(
        f"{CFG.paths.rag_db}{CFG.data.rag.faiss_tp_git_hunk_map}")

    # FP GIT-HUNK DATA
    fp_embeddings = faiss.read_index(
        f"{CFG.paths.rag_db}{CFG.data.rag.faiss_fp_git_hunk}")
    fp_map = pd.read_csv(
        f"{CFG.paths.rag_db}{CFG.data.rag.faiss_fp_git_hunk_map}")

    # COMPLETE GIT-HUNK DATA
    complete_embeddings = faiss.read_index(
        f"{CFG.paths.rag_db}{CFG.data.rag.faiss_complete_git_hunk}")
    complete_map = pd.read_csv(
        f"{CFG.paths.rag_db}{CFG.data.rag.faiss_complete_git_hunk_map}")

    return (adv_embeddings, adv_map, tp_embeddings,
            tp_map, fp_embeddings, fp_map, complete_embeddings, complete_map)


def text_embeddings(model, text: list, device: str = 'cpu') -> list:
    """_summary_

    Args:
        model (_type_): _description_
        text (list): _description_

    Returns:
        list: _description_
    """
    # Encode info. about the text into embeddings
    temp_embeddings = model.encode(text,
                                   batch_size=16,
                                   device=device,
                                   show_progress_bar=False,
                                   convert_to_tensor=True,
                                   normalize_embeddings=True)

    return temp_embeddings


def code_embeddings_batch(model, tokenizer, code: list, device: str = 'cuda:0') -> list:
    """Generate embeddings for code in a batch

    Args:
        model (_type_): Loaded model
        tokenizer (_type_): Loaded tokenizer
        code (str): String of code
        device (str, optional): Device type. Defaults to 'cuda:0'. cpu/cuda:0

    Returns:
        list: _description_
    """
    # tokenize the func1 of the sampled data
    func_tokenized_datasets = tokenizer(code,
                                        padding=True,
                                        truncation=True,
                                        return_attention_mask=True,
                                        return_token_type_ids=False,
                                        return_tensors='pt')

    # convert to a Dataset
    func_tokenized_datasets = Dataset.from_dict(func_tokenized_datasets)

    # convert to a torch format
    func_tokenized_datasets.set_format("torch")

    # convert the Dataset to a DataLoader for later batching of the embeddings
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    func_tokenized_dataloader = DataLoader(
        func_tokenized_datasets, shuffle=False, batch_size=8, collate_fn=data_collator
    )

    # total embeddings holder
    func_embeddings = []

    # run the torch.no_grad so we don't update weights
    with torch.no_grad():
        val_bar = tqdm(func_tokenized_dataloader,
                       total=len(func_tokenized_dataloader))
        # iterate through each batch
        for i, batch in enumerate(val_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            #  pooler_output doesn't exist for T5
            embeddings_batch = outputs
            # detach the embeddings from the GPU to the CPU
            func_embeddings.append(embeddings_batch.cpu().detach())

    # cat the tensors and convert to a numpy array
    func_embeddings = torch.cat(func_embeddings).numpy()

    print(func_embeddings.shape)

    return func_embeddings


def code_embeddings(model, tokenizer, code: str, device: str = 'cpu') -> list:
    """Generate embeddings for code

    Args:
        model (_type_): Loaded model
        tokenizer (_type_): Loaded tokenizer
        code (str): String of code
        device (str, optional): Device type. Defaults to 'cpu'. cpu/cuda:0

    Returns:
        list: _description_
    """
    # tokenize the func1 of the sampled data
    input_data = tokenizer(code,
                           padding=True,
                           truncation=True,
                           return_attention_mask=True,
                           return_token_type_ids=False,
                           return_tensors='pt')

    input_data.to(device)

    # run the torch.no_grad so we don't update weights
    with torch.no_grad():
        # get the embeddings
        outputs = model(**input_data)[0]

        # detach from the GPU
        temp_embeddings = outputs.cpu().detach()

    return temp_embeddings
