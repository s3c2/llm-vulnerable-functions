"""
Helper functions for models

E.g., evaluate our model based on our data
"""
import ast
import os
import sys
import gc
from pathlib import Path
from types import SimpleNamespace
import uuid
import random
import yaml
import numpy as np
import torch
from torch import nn
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from statistics import mean
from utils import prompt_helper
from transformers import set_seed


def load_data(temp_cfg: SimpleNamespace, load_similar: bool = True,
              load_cot_examples: bool = True, do_multi_git_hunks: bool = False):
    """
    Args:
        temp_cfg (SimpleNamespace): _description_
        load_similar (bool): Load similar description. Defaults True
        load_cot_examples (bool): Load COT examples from 4.5-generate-cot-examples. Defaults True.
    """

    # load the data
    temp_data = pd.read_csv(
        f"{temp_cfg.paths.patchparser}{temp_cfg.data.patchparser}")

    # ignore test files
    temp_data = temp_data[~temp_data['file_name'].str.lower().str.contains('test')
                          ].reset_index(drop=True)

    # load the advisory data
    temp_advisory_data = pd.read_csv(
        f"{temp_cfg.paths.advisory}{temp_cfg.data.go_advisory}")

    # set the description of the advisory for the patchparser data
    temp_data_description = temp_data.merge(temp_advisory_data[['id', 'details']].drop_duplicates(),
                                            on=['id'],
                                            how='inner').reset_index(drop=True)

    # clean the advisory details string
    if temp_cfg.data.clean_advisory_details:
        temp_data_description['details'] = temp_data_description.apply(
            lambda x: prompt_helper.clean_advisory_details(
                details=x['details']),
            axis=1
        )

    # load the COT examples
    if load_cot_examples:
        unique_cot_columns = ['id', 'repo_owner', 'repo_name', 'file_name',
                              'sha', 'raw_patch', 'label_combined_unique']

        temp_tp_cot = pd.read_csv(
            f"{temp_cfg.paths.patchparser}{temp_cfg.data.tp_cot_examples}")

        temp_fp_cot = pd.read_csv(
            f"{temp_cfg.paths.patchparser}{temp_cfg.data.fp_cot_examples}")

        temp_fp_cot = temp_fp_cot.rename(
            columns={"cot_example_text": "cot_example_text_fp"})

        temp_data_description = pd.merge(temp_data_description,
                                         temp_tp_cot[unique_cot_columns +
                                                     ['cot_example_text']],
                                         on=unique_cot_columns,
                                         how='left')

        temp_data_description = pd.merge(temp_data_description,
                                         temp_fp_cot[unique_cot_columns +
                                                     ['cot_example_text_fp']],
                                         on=unique_cot_columns,
                                         how='left')

        # update the two columns to a single column
        temp_data_description['cot_example_text'] = temp_data_description.apply(
            lambda x: x['cot_example_text_fp'] if pd.isna(x['cot_example_text'])
            and pd.notna(x['cot_example_text_fp']) else x['cot_example_text'],
            axis=1
        )

        # drop the extra column
        temp_data_description = temp_data_description.drop(
            columns=["cot_example_text_fp"])

    # load multi-git-hunk
    if do_multi_git_hunks:
        temp_data_multi_hunk = temp_data_description[temp_data_description['number_combined_functions_changed'] > 1].reset_index(
            drop=True).copy()

    temp_data_description = temp_data_description[temp_data_description['number_combined_functions_changed'] == 1].reset_index(
        drop=True)

    # seperate the data into TP/FPs
    temp_tp_data = temp_data_description[(temp_data_description['number_combined_functions_changed'] == 1) &
                                         (temp_data_description['label_combined_unique'].str.contains('True'))].reset_index(drop=True)

    temp_fp_data = temp_data_description[(temp_data_description['number_combined_functions_changed'] == 1) &
                                         (temp_data_description['label_combined_unique'].str.contains('False'))].reset_index(drop=True)

    # load the similar description
    # generated from ./code/5-generate-similarity/1-advisory-description.ipynb
    if load_similar:
        temp_description_matches = pd.read_csv(
            f"{temp_cfg.paths.advisory}{temp_cfg.data.go_similar_description}")

        if do_multi_git_hunks:
            # return the extra multi-git-hunk
            return temp_data_description, temp_advisory_data, temp_tp_data, temp_fp_data, temp_description_matches, temp_data_multi_hunk
        else:
            return temp_data_description, temp_advisory_data, temp_tp_data, temp_fp_data, temp_description_matches
    else:
        # do not return the matches if you don't want them
        return temp_data_description, temp_advisory_data, temp_tp_data, temp_fp_data


class RecursiveNamespace(SimpleNamespace):
    """
    Extending SimpleNamespace for Nested Dictionaries
    # https://dev.to/taqkarim/extending-simplenamespace-for-nested-dictionaries-58e8

    Args:
        SimpleNamespace (_type_): Base class is SimpleNamespace

    Returns:
        _type_: A simple class for nested dictionaries
    """
    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)

        return entry

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, RecursiveNamespace(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))


def load_cfg(base_dir: Path,
             filename: str,
             *,
             as_namespace: bool = True,
             update_paths: bool = True) -> SimpleNamespace:
    """
    Load YAML configuration files saved uding the "cfgs" directory
    Args:
        base_dir (Path): Directory to YAML config. file
        filename (str): Name of YAML configuration file to load
        update_paths (str): Combines the base path with the other paths. Defaults True.
    Returns:
        SimpleNamespace: A simple class for calling configuration parameters
    """
    cfg_path = Path(base_dir) / f"{filename}.yaml"
    with open(cfg_path, 'r') as file:
        cfg_dict = yaml.safe_load(file)
    file.close()

    # this will join the paths to the base path from the config
    if update_paths:
        for k in cfg_dict["paths"]:
            # skip the base path
            if k == 'base':
                pass
            else:
                # combine the paths
                cfg_dict["paths"][k] = cfg_dict["paths"]["base"] + \
                    cfg_dict["paths"][k]
                print(cfg_dict["paths"][k])

    if as_namespace:
        cfg = RecursiveNamespace(**cfg_dict)
    else:
        cfg = cfg_dict
    return cfg


def seed_everything(*, seed: int = 42):
    """
    Seed everything

    Args:
        seed (_type_): Seed
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def label_outputs(temp_agg_results: pd.DataFrame) -> pd.DataFrame:
    """
    Label the output for TP/FP/FN/TN in a dataframe
    """
    # TP example:
    # label_combined_unique = True
    # model_output_label = [True, False] -> if true in llama output we count it
    temp_agg_results['tp'] = temp_agg_results.apply(
        lambda x: 1 if (True in x['model_output_label']) & (
            x['label_combined_unique'] == True) else 0,
        axis=1
    )

    # TN example:
    # label_combined_unique = False
    # model_output_label = [False]
    temp_agg_results['tn'] = temp_agg_results.apply(
        lambda x: 1 if (True not in x['model_output_label']) & (
            x['label_combined_unique'] == False) else 0,
        axis=1
    )

    # FN example:
    # label_combined_unique = True
    # model_output_label = [False, False]
    temp_agg_results['fn'] = temp_agg_results.apply(
        lambda x: 1 if (True not in x['model_output_label']) & (
            x['label_combined_unique'] == True) else 0,
        axis=1
    )

    # example:
    # label_combined_unique = False
    # model_output_label = [True]
    temp_agg_results['fp'] = temp_agg_results.apply(
        lambda x: 1 if (True in x['model_output_label']) & (
            x['label_combined_unique'] == False) else 0,
        axis=1
    )

    return temp_agg_results


def calc_metrics(tmp_results: pd.DataFrame, verbose=True) -> dict:
    """
    Calculate and print the classification metrics, including precision and recall.

    This function creates a final label for evaluation based on the 'model_output_label' column
    in the provided DataFrame. It then prints out the classification report comparing the
    'label_combined_unique' column and the calculated 'pred_final_label'.

    Parameters:
    tmp_results (pd.DataFrame): A DataFrame containing the necessary columns 'model_output_label'
                                and 'label_combined_unique' for metric calculation.

    Returns:
    dict: A dictionary containing the calculated metrics. (Note: Currently, this function prints
          the classification report and does not return the metrics. This needs to be modified
          based on specific requirements.)
    """

    # Create a final label that we can evaluate
    tmp_results['pred_final_label'] = tmp_results.apply(
        lambda x: True if True in x['model_output_label'] else False,
        axis=1
    )

    if verbose:
        # Print classification report
        print(classification_report(tmp_results['label_combined_unique'],
                                    tmp_results['pred_final_label']))

    metrics = classification_report(tmp_results['label_combined_unique'],
                                    tmp_results['pred_final_label'],
                                    output_dict=True)

    return metrics


def simple_generate(model, tokenizer, input_text, CFG) -> str:
    """Tokenizes and runs model.generate on a model for a given input text

    Args:
        model (_type_): model
        tokenizer (_type_): tokenizer
        input_text (str): Input text (generally a full prompt)

    Returns:
        str: Decoded response from model.generate
    """
    with torch.no_grad():
        # tokenize the prompt
        batch = tokenizer(input_text,
                          return_tensors="pt",
                          add_special_tokens=False,
                          padding=True).to(
            CFG.gen_cfg.batch_device_map)

        # start timer on generate
        start_time = time.time()

        # generate the response
        # Temperature settings:
        # https://github.com/facebookresearch/llama/blob/main/example_chat_completion.py#L11
        try:
            response = model.generate(
                **batch,
                do_sample=CFG.gen_cfg.do_sample,
                top_k=CFG.gen_cfg.top_k,
                max_new_tokens=CFG.gen_cfg.max_new_tokens,
                top_p=CFG.gen_cfg.top_p,
                temperature=CFG.gen_cfg.temperature,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=CFG.gen_cfg.return_dict_in_generate,
                output_scores=CFG.gen_cfg.output_scores,
                num_return_sequences=CFG.gen_cfg.num_return_sequences,
            )

            # start timer on generate
            end_time = time.time()

            if "deepseek" in CFG.models.model_name or "WizardCoder" in CFG.models.model_name:
                # the response needs to be decoded
                decode_response = tokenizer.decode(
                    response.sequences[0]).split('### Response:')[-1].strip()
            elif "Mixtral" in CFG.models.model_name:
                # the response needs to be decoded
                decode_response = tokenizer.decode(
                    response.sequences[0]).split('[/INST] ')[-1].strip()
            else:
                # the response needs to be decoded
                decode_response = tokenizer.decode(
                    response.sequences[0], skip_special_tokens=True).split('[/INST] ')[-1].strip()

            # delete the batch
            del batch, response
        except Exception as e:
            print(str(e))

            decode_response = "True | Error"
            end_time = 2
            start_time = 1
            # delete the batch
            del batch

    # empty cache
    torch.cuda.empty_cache()
    gc.collect()

    return decode_response, round(end_time - start_time, 2)


def get_token_length(tokenizer, input_text, CFG, verbose: bool = False) -> int:
    """Get the length of the tokens from the tokenizer

    Args:
        tokenizer (_type_): tokenizer
        input_text (str): Input text (generally a full prompt)
        CFG: Typical config

    Returns:
        str: Length generated by the tokenizer
    """
    with torch.no_grad():
        # tokenize the prompt
        # no need to add special tokens, we already have added <s> in the prompt
        # Left padding: https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
        tokenizer.padding_side = "left"
        batch = tokenizer(input_text,
                          return_tensors="pt",
                          add_special_tokens=False,
                          padding=True
                          ).to(CFG.gen_cfg.batch_device_map)

        if verbose:
            print(f"Batch length: {len(batch[0])}")

        return len(batch[0])


def generate_log_probs(model, tokenizer, input_text, CFG, top_k: int = 50, verbose: bool = False) -> str:
    """Tokenizes and runs a model for a given input text
    Returns the log probs for True and False tokens

    Args:
        model (_type_): model
        tokenizer (_type_): tokenizer
        input_text (str): Input text (generally a full prompt)
        CFG: 

    Returns:
        str: Decoded response from model.generate
    """
    with torch.no_grad():
        # tokenize the prompt
        # no need to add special tokens, we already have added <s> in the prompt
        # Left padding: https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
        tokenizer.padding_side = "left"
        batch = tokenizer(input_text,
                          return_tensors="pt",
                          add_special_tokens=False,
                          padding=True
                          ).to(CFG.gen_cfg.batch_device_map)

        if verbose:
            # CodeLlama can do up to (16k tokens) see model.config.max_position_embeddings
            print(f"Batch length: {len(batch[0])}")

        # start timer on generate
        start_time = time.time()

        # generate output
        # understand model outputs: https://huggingface.co/docs/transformers/main_classes/output
        output = model(**batch)

        # start timer on generate
        end_time = time.time()

        # Log probs: https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/17

        # we take the last logit, because this is the first generation of the model
        last_logit = output.logits[0, -1:, :]
        last_logit_probs = nn.Softmax(dim=1)(last_logit).cpu()
        last_logit_argmax = torch.argmax(last_logit_probs)
        last_logit_argmax_decode = tokenizer.decode(last_logit_argmax)
        last_logit_argsort = torch.argsort(last_logit_probs, descending=True)
        last_logit_argsort_decode = tokenizer.decode(
            last_logit_argsort[0][:top_k]).split(' ')
        last_logit_argsort_probs = last_logit_probs[0][last_logit_argsort[0][:top_k]].numpy(
        )

        if "WizardCoder" in CFG.models.model_name or "deepseek" in CFG.models.model_name:
            # finding the True/False tokens in WizardCoder is weird
            last_logit_argsort_decode = tokenizer.decode(
                last_logit_argsort[0][:]).split(' ')

            last_logit_argsort_decode = [tokenizer.decode(
                x) for x in last_logit_argsort[0][:]]
            last_logit_argsort_probs = last_logit_probs[0][last_logit_argsort[0][:]].numpy(
            )
            # create a top token dictionary with probs
            top_toks = dict(zip(last_logit_argsort_decode,
                                last_logit_argsort_probs))

            # find the top True token
            for key in top_toks:
                if "true" in key.lower():
                    print(f"{key} | {top_toks[key]}")
                    true_prob = top_toks[key]
                    break

            # find the top False token
            for key in top_toks:
                if "false" in key.lower():
                    print(f"{key} | {top_toks[key]}")
                    false_prob = top_toks[key]
                    break

        else:
            # create a top 50 token dictionary with probs
            top_toks = dict(zip(last_logit_argsort_decode,
                                last_logit_argsort_probs))

            # get tokens for the 'True' and 'False'
            true_token = tokenizer.encode('True', add_special_tokens=False)[0]
            false_token = tokenizer.encode(
                'False', add_special_tokens=False)[0]

            # get the probs for the True/False tokesn
            true_prob = last_logit_probs[0][true_token]
            false_prob = last_logit_probs[0][false_token]

        # true/false dict
        true_false = {"True": true_prob, "False": false_prob}

        if verbose:
            print(f"True prob: {true_prob}\n"
                  f"False prob: {false_prob}\n"
                  f"Max prob: {last_logit_probs[0][last_logit_argmax]} "
                  f"| Value: {last_logit_argmax_decode}")

    # empty cache
    torch.cuda.empty_cache()

    return last_logit_argmax_decode, round(end_time - start_time, 2), true_false, top_toks


def combine_results(idx, row, response, gen_time, CFG,
                    true_false=None, top_probs=None, faiss_score=None,
                    verbose=True, dyn_scores=None) -> pd.DataFrame:
    """Combine the results to a single dataframe

    Args:
        row (_type_): _description_
        response (_type_): _description_
        gen_time (_type_): _description_
        verbose (True): _description_

    Returns:
        pd.DataFrame: _description_
    """
    if verbose:
        print(f"===================RESULTS========================")
        print(f"{row.id} | GT Label: {row.label_combined_unique} | "
              f"Test Function: {row.changed_combined_functions} | "
              f" Gen Time: {gen_time}\n"
              f"Model Response: {response}\n")

        # print the order of the examples if they were random for the last two
        if dyn_scores is not None:
            print(f"Example Order: {[x['label'] for x in dyn_scores]}")

        # generative decision search for True/False
        model_response_parse = 'True' in response
        model_correct = str(model_response_parse) in row.label_combined_unique
        print(f"~~~~MODEL CORRECT: {model_correct} | GT Label: {row.label_combined_unique}"
              f" | Model Response: {model_response_parse}~~~~~~~\n\n")

    if true_false is not None and top_probs is not None and faiss_score is not None:
        temp_df = pd.DataFrame([[idx, row.id, row.repo_owner,
                                row.repo_name, row.raw_patch,
                                row.sha,
                                row.symbol_name, row.changed_combined_functions,
                                response,
                                ast.literal_eval(
                                    row.label_combined_unique)[0],
                                float(true_false["True"]),
                                float(true_false["False"]),
                                faiss_score,
                                mean(faiss_score),
                                gen_time]],
                               columns=['idx', 'id', 'repo_owner',
                                        'repo_name', 'raw_patch', 'sha',
                                        'symbol_name', 'changed_combined_functions',
                                        'model_original_results', "label_combined_unique",
                                        'true_prob', 'false_prob',
                                        'faiss_scores', 'faiss_scores_mean',
                                        "gen_time"])

        # bayesian posterior score based on prior true/false probs
        temp_df['bayesian_posterior'] = temp_df.apply(
            lambda x: ((x['true_prob'] * CFG.log_prob.p_true_prior) / ((x['true_prob']
                       * CFG.log_prob.p_true_prior) + (x['false_prob'] * CFG.log_prob.p_false_prior))),
            axis=1
        )

        # model bayesian update threshold
        temp_df['model_results_bayesian'] = temp_df.apply(
            lambda x: True if x['bayesian_posterior'] >= CFG.log_prob.bayesian_threshold else False,
            axis=1
        )

        # create a final model value
        temp_df['model_results_threshold'] = temp_df.apply(
            lambda x: True if x['true_prob'] >= CFG.log_prob.true_threshold else False,
            axis=1
        )

        # create a difference score between the true and false probability
        temp_df['prob_diff'] = temp_df.apply(
            lambda x: x['true_prob']-x['false_prob'],
            axis=1
        )

        # create a final model value
        temp_df['model_results_diff_threshold'] = temp_df.apply(
            lambda x: True if x['prob_diff'] >= CFG.log_prob.diff_threshold else False,
            axis=1
        )

        # basic max temp results
        temp_df['model_results'] = temp_df.apply(
            lambda x: True if x['true_prob'] >= x['false_prob'] else False,
            axis=1
        )

        # Create a simple value to quickly see if model is correct
        temp_df['model_correct'] = temp_df.apply(
            lambda x: x['model_results'] == x['label_combined_unique'],
            axis=1
        )

    elif faiss_score is not None:
        # Parse the response if this is a generative model
        model_response_parse = 'True' in response

        temp_df = pd.DataFrame([[idx, row.id, row.repo_owner,
                                row.repo_name, row.raw_patch,
                                row.sha,
                                row.symbol_name, row.changed_combined_functions,
                                model_response_parse,
                                response,
                                ast.literal_eval(
                                    row.label_combined_unique)[0],
                                faiss_score,
                                mean(faiss_score),
                                gen_time]],
                               columns=['idx', 'id', 'repo_owner',
                                        'repo_name', 'raw_patch', 'sha',
                                        'symbol_name', 'changed_combined_functions', 'model_results',
                                        'model_results_full', "label_combined_unique",
                                        'faiss_scores', 'faiss_scores_mean', "gen_time"])

    else:
        # Parse the response if this is a generative model
        model_response_parse = 'True' in response

        temp_df = pd.DataFrame([[idx, row.id, row.repo_owner,
                                row.repo_name, row.raw_patch,
                                row.sha,
                                row.symbol_name, row.changed_combined_functions,
                                model_response_parse,
                                response,
                                ast.literal_eval(
                                    row.label_combined_unique)[0],
                                gen_time]],
                               columns=['idx', 'id', 'repo_owner',
                                        'repo_name', 'raw_patch', 'sha',
                                        'symbol_name', 'changed_combined_functions', 'model_results',
                                        'model_results_full', "label_combined_unique", "gen_time"])

    return temp_df


def quick_calc(tmp_full_results, CFG, config_start_time, wb_logger, config_ver):
    if CFG.log_prob.use_log_probs:  # USE THIS FOR LOG PROB TECHNIQUE
        tmp_full_results['model_output_label'] = tmp_full_results.apply(
            lambda x: x['model_results'],
            axis=1
        )
    else:  # create a final label for the results USE THIS FOR GENERATION TECHNIQUE
        tmp_full_results['model_output_label'] = tmp_full_results.apply(
            lambda x: True if 'True' in str(x['model_results']) else False,
            axis=1
        )

    # create a single function value
    tmp_full_results['single_function'] = tmp_full_results.apply(lambda x: ast.literal_eval(x['changed_combined_functions'])[0],
                                                                 axis=1)

    # aggregate results by changed function
    # condense the unique symbols to a list
    agg_function_results = (
        tmp_full_results.groupby(["id", "sha", "single_function",
                                  "label_combined_unique", "symbol_name"])["model_output_label"]
        .apply(list)
        .reset_index(drop=False)
    )

    # these are the aggregated results of how we will get precision/recall values
    agg_results = label_outputs(
        temp_agg_results=agg_function_results.copy())

    metrics = calc_metrics(tmp_results=agg_results.copy())


def aggregate_save_results(tmp_full_results, CFG, config_start_time, wb_logger, config_ver):
    # End the timer
    end_time = time.time()

    if CFG.log_prob.use_log_probs:  # USE THIS FOR LOG PROB TECHNIQUE
        tmp_full_results['model_output_label'] = tmp_full_results.apply(
            lambda x: x['model_results'],
            axis=1
        )
    else:  # create a final label for the results USE THIS FOR GENERATION TECHNIQUE
        tmp_full_results['model_output_label'] = tmp_full_results.apply(
            lambda x: True if 'True' in str(x['model_results']) else False,
            axis=1
        )

    # create a single function value
    tmp_full_results['single_function'] = tmp_full_results.apply(lambda x: ast.literal_eval(x['changed_combined_functions'])[0],
                                                                 axis=1)

    # aggregate results by changed function
    # condense the unique symbols to a list
    agg_function_results = (
        tmp_full_results.groupby(["id", "sha", "single_function",
                                  "label_combined_unique", "symbol_name"])["model_output_label"]
        .apply(list)
        .reset_index(drop=False)
    )

    # these are the aggregated results of how we will get precision/recall values
    agg_results = label_outputs(
        temp_agg_results=agg_function_results.copy())

    print(f"Configuration: {config_ver}")
    metrics = calc_metrics(tmp_results=agg_results.copy())

    # Calculate the total time taken
    total_time = end_time - config_start_time

    print(f"Total execution time: {round(total_time, 2)} seconds")

    if CFG.results.save_results:

        # aggregated results with tp/fp/tn/fn counts
        agg_results.to_csv(f"{CFG.paths.results}{CFG.results.agg_results.replace('{CONFIG_VER}', config_ver)}",
                           encoding='utf-8', index=False)

        # full_results
        tmp_full_results.to_csv(f"{CFG.paths.results}{CFG.results.full_results.replace('{CONFIG_VER}', config_ver)}",
                                encoding='utf-8', index=False)

    if CFG.wandb.log_results:
        wb_logger.log(metrics)


def parse_config_path(full_results_path):

    tmp = full_results_path.split('_')

    tmp = tmp[3:]

    if "zero" in tmp[0]:
        paradigm = tmp[0]
        system_prompt = tmp[1]
        user_prompt = tmp[2].replace(".csv", "")
        example_prompt = None
        example_type = None
        example_amount = None
        example_order = None
        model_update = tmp[-1]
        if "csv" in model_update:
            model_update = model_update.replace(".csv", "")

    elif "few" in tmp[0]:
        paradigm = tmp[0]
        system_prompt = tmp[1]
        user_prompt = tmp[2]
        example_prompt = tmp[3]
        example_type = tmp[4]
        example_amount = tmp[5].strip('eg')
        example_order = tmp[6]
        model_update = tmp[-1]
        if "csv" in model_update:
            model_update = model_update.replace(".csv", "")

    results = {
        # "path": full_results_path,
        "paradigm": paradigm,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "example_prompt": example_prompt,
        "example_type": example_type,
        "example_amount": example_amount,
        "example_order": example_order,
        "model": model_update,
    }

    return results
