"""
Chain of Thought Prompting
"""
import sys  # noqa
sys.path.append('.')   # noqa

# Custom Modules
from pathlib import Path
import pandas as pd
import torch
import ast
import numpy as np
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          GenerationConfig, pipeline, AutoModel)
from sentence_transformers import SentenceTransformer
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from utils.prompt_helper import (CustomDynamicExamples,
                                 CustomRandomSelector,
                                 create_example_selector,
                                 create_example_selector_faiss)
from utils import model_helper, logger_helper, embedding_helper
from langchain.prompts import load_prompt
from langchain.prompts.pipeline import PipelinePromptTemplate
import warnings
import time
import wandb
import pprint


# To ignore all user warnings
warnings.filterwarnings('ignore', category=UserWarning)

# check if in debug mode:
# is_debugger = logger_helper.debugger_is_active()
is_debugger = True

# if debugger is active we can manually change values
if is_debugger:
    config_file = "sample_config"
    debug_test_amount = 5
    debug_test_range = False
    debug_test_start = 32
    debug_test_end = 36
    # Never log to Weights and Bias and never save results in debug mode
    debug_log_results = False
    debug_save_results = False
    do_missed_results = False

else:  # if not in debug, require system args
    # Check if enough arguments are passed
    if len(sys.argv) < 1:
        print("Usage: python script.py arg1 ...")
        sys.exit(1)

    # Accessing arguments
    config_file = sys.argv[1]

# set base paths
base_dir = Path.cwd()
config_dir = f"{base_dir}/code/llm/cfgs/"

# Load the configuration file
CFG = model_helper.load_cfg(base_dir=Path(config_dir),
                            filename=f"{config_file}")

if is_debugger:
    CFG.data.test_amount = debug_test_amount
    CFG.data.debut_test_range = debug_test_range
    CFG.data.debut_test_start = debug_test_start
    CFG.data.debut_test_end = debug_test_end
    CFG.data.do_missed_results = do_missed_results
    # Never log to Weights and Bias and never save results in debug mode
    CFG.wandb.log_results = debug_log_results
    CFG.results.save_results = debug_save_results

# seed everything
model_helper.seed_everything()

if __name__ == '__main__':
    ###############################################################################
    ###############################################################################
    # Load the data
    data, data_adv, data_tp, data_fp, desc_sim = model_helper.load_data(
        temp_cfg=CFG)

    # make a copy of data
    data_original = data.copy()

    # Load the FAISS index and maps
    (embeddings_adv,
     embeddings_adv_map,
     embeddings_tp_git,
     embeddings_tp_git_map,
     embeddings_fp_git,
     embeddings_fp_git_map,
     embeddings_complete_git,
     embeddings_complete_git_map) = embedding_helper.load_faiss_data(CFG=CFG)

    ###############################################################################
    ###############################################################################
    # Load the models
    # Left padding: https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
    tokenizer = AutoTokenizer.from_pretrained(
        f"{CFG.models.base}{CFG.models.model_name}",
        padding_side="left")

    # Most LLMs don't have a pad token by default
    tokenizer.pad_token = tokenizer.eos_token

    if CFG.gen_cfg.load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(f"{CFG.models.base}{CFG.models.model_name}",
                                                     device_map=CFG.gen_cfg.device_map,
                                                     load_in_8bit=True,
                                                     torch_dtype=torch.float16)
    elif CFG.gen_cfg.load_in_4bit:
        if 'codet5' in CFG.models.model_name:
            model = AutoModelForSeq2SeqLM.from_pretrained(f"{CFG.models.base}{CFG.models.model_name}",
                                                          torch_dtype=torch.float16,
                                                          device_map=CFG.gen_cfg.device_map,
                                                          load_in_4bit=True,
                                                          trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(f"{CFG.models.base}{CFG.models.model_name}",
                                                         device_map=CFG.gen_cfg.device_map,
                                                         load_in_4bit=True,
                                                         torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(f"{CFG.models.base}{CFG.models.model_name}",
                                                     device_map=CFG.gen_cfg.device_map,
                                                     torch_dtype=torch.float16)

    # place the model in eval mode
    model.eval()

    # Load the code embedding models, load to CPU
    code_emb_tokenizer = AutoTokenizer.from_pretrained(
        f"{CFG.models.base}{CFG.models.rag.code_embedding_model}")

    code_emb_model = AutoModel.from_pretrained(
        f"{CFG.models.base}{CFG.models.rag.code_embedding_model}",
        trust_remote_code=True).to(CFG.models.rag.rag_device)

    # Load the text embedding models, load to CPU
    text_emb_model = SentenceTransformer(
        f"{CFG.models.base}{CFG.models.rag.text_embedding_model}",
        device=CFG.models.rag.rag_device)

    # check if using a zero or few-shot template:
    for paradigm in CFG.prompts.paradigm:
        config_start_time = time.time()

        ###################################################################################
        # Load the prompt template
        # set each path type based
        if paradigm == "zero":
            prompt_template_path = f"{CFG.paths.template}{CFG.prompts.template_zero_shot}"
            # set a system prompt type path for later
            system_prompt_type_path = CFG.paths.zero_shot_system_prompts
        elif paradigm == "few":
            prompt_template_path = f"{CFG.paths.template}{CFG.prompts.template_few_shot}"
            # set a system prompt type path for later
            system_prompt_type_path = CFG.paths.few_shot_system_prompts

        print(f"Using prompt template: {prompt_template_path}")
        prompt_template = load_prompt(prompt_template_path)

        ###################################################################################
        # Load the system prompts template
        # set each path type based
        for system_prompt_type in CFG.prompts.system_prompt_type:
            # set the system_prompt_path
            system_prompt_path = f"{system_prompt_type_path}{system_prompt_type}-system-v{CFG.prompts.system_prompt_version}.txt"

            print(f"Using system prompt: {system_prompt_path}")
            system_prompt = Path(system_prompt_path).read_text()

            ###################################################################################
            # Load the user prompts template for the paradigm
            # set the systemp_prompt_path
            user_template_path = f"{CFG.paths.user_template}{system_prompt_type}-user-v{CFG.prompts.user_template_version}.yaml"

            print(f"Using user template: {user_template_path}")
            user_template = load_prompt(user_template_path)

            ###################################################################################
            # Load the example prompts template for the paradigm
            # set each path type based
            if paradigm == "few":
                example_template_path = f"{CFG.paths.example_template}{system_prompt_type}-example-v{CFG.prompts.example_template_version}.yaml"

                print(f"Using example template: {example_template_path}")
                example_template = load_prompt(example_template_path)

            ###################################################################################
            # Start to evaluate the model
            # create a dataframe to hold the results
            full_results = pd.DataFrame()

            # only run a subset in debug mode or test mode
            if is_debugger or CFG.data.test:
                # data = data[:CFG.data.test_amount].reset_index(drop=True)
                if CFG.data.debug_test_range:
                    data = data[CFG.data.debut_test_start:CFG.data.debut_test_end].reset_index(
                        drop=True)
                else:
                    data = data[-CFG.data.test_amount:].reset_index(drop=True)

            # evaluate for zero-shot
            if paradigm == "zero":
                # create a configuration save name
                config_ver = f"{paradigm}_" \
                             f"{system_prompt_type}-system-v{CFG.prompts.system_prompt_version}_" \
                             f"{system_prompt_type}-user-v{CFG.prompts.system_prompt_version}_" \
                             f"{CFG.models.model_name.strip('/')}"

                # iterrate through each row of the data to evaluate
                # Weights and Bias Logger
                if CFG.wandb.log_results:
                    wb_logger = logger_helper.WandB(cfg_basepath=Path(config_dir),
                                                    cfg_filename=f"{config_file}").get_logger()
                else:
                    wb_logger = None
                for idx, row in data.iterrows():
                    print(f"\n\n{idx}/{len(data)} | {config_ver}")
                    # Form the pipeline_prompt
                    # set the input prompts to use in the pipeline
                    input_prompts = [
                        ("user_message", user_template)
                    ]

                    # create a full pipeline
                    pipeline_prompt = PipelinePromptTemplate(
                        final_prompt=prompt_template,
                        pipeline_prompts=input_prompts)

                    # create a final prompt
                    final_prompt = pipeline_prompt.format(
                        system_prompt=system_prompt,
                        details=row.details,
                        functions=f"{list(set(ast.literal_eval(row.changed_combined_functions)))[0]}",
                        code=f"{row.raw_patch}",
                    )

                    # get the response from the model
                    response, gen_time = model_helper.simple_generate(model=model,
                                                                      tokenizer=tokenizer,
                                                                      input_text=final_prompt,
                                                                      CFG=CFG)

                    # create a single clean dataframe of the results
                    temp_results = model_helper.combine_results(idx=idx,
                                                                row=row,
                                                                response=response,
                                                                true_false=None,
                                                                top_probs=None,
                                                                faiss_score=[
                                                                    0.0, 0.0],
                                                                gen_time=gen_time,
                                                                CFG=CFG)
                    # concat the full results
                    full_results = pd.concat([full_results, temp_results])

                    # Quick calc so we can see how we're doing
                    model_helper.quick_calc(tmp_full_results=full_results,
                                            CFG=CFG,
                                            config_start_time=config_start_time,
                                            wb_logger=wb_logger,
                                            config_ver=config_ver
                                            )

                # log config values
                if CFG.wandb.log_results:
                    wb_logger.log({"paradigm": f"{paradigm}-shot",
                                   "system-prompt": f"{system_prompt_type}-system-v{CFG.prompts.system_prompt_version}",
                                   "user-template": f"{system_prompt_type}-user-v{CFG.prompts.system_prompt_version}",
                                   "example-template": None,
                                   "example-type": None,
                                   "example-amount": None,
                                   "example-order": None,
                                   "config-version": config_ver})

                # save all the results
                model_helper.aggregate_save_results(tmp_full_results=full_results,
                                                    CFG=CFG,
                                                    config_start_time=config_start_time,
                                                    wb_logger=wb_logger,
                                                    config_ver=config_ver
                                                    )

                updated_metrics = model_helper.generate_new_metrics(
                    full_results=full_results, wb_logger=wb_logger, CFG=CFG)

                pprint.pprint(updated_metrics)

                if CFG.wandb.log_results:
                    wb_logger.finish()

            if paradigm == "few":
                # form the examples if necessary in few-shot
                for example_type in CFG.prompts.example_type:
                    for example_amount in CFG.prompts.example_amount:
                        for example_order in CFG.prompts.example_order:
                            for example_sort in CFG.prompts.example_sort:
                                # reset the dataFrame
                                full_results = pd.DataFrame()

                                # create a configuration save name
                                config_ver = f"{paradigm}-shot_" \
                                    f"{system_prompt_type}-system-v{CFG.prompts.system_prompt_version}_" \
                                    f"{system_prompt_type}-user-v{CFG.prompts.system_prompt_version}_" \
                                    f"{system_prompt_type}-example-v{CFG.prompts.example_template_version}_" \
                                    f"{example_type}_{example_amount}eg_{example_order}_{example_sort}_" \
                                    f"{CFG.models.model_name.strip('/')}"

                                # evaluate for few-shot
                                # now iterrate through each row of the data to evaluate
                                # Weights and Bias Logger
                                if CFG.wandb.log_results:
                                    wb_logger = logger_helper.WandB(cfg_basepath=Path(config_dir),
                                                                    cfg_filename=f"{config_file}").get_logger()
                                else:
                                    wb_logger = None

                                for idx, row in data.iterrows():
                                    print(
                                        f"\n\n{idx}/{len(data)} | {config_ver}")

                                    # dynamically select examples
                                    example_selector = create_example_selector_faiss(user_input_details=row.details,
                                                                                     user_input_id=row.id,
                                                                                     user_input_raw_patch=row.raw_patch,
                                                                                     example_type=example_type,
                                                                                     example_amount=example_amount,
                                                                                     example_template=example_template,
                                                                                     example_order=example_order,
                                                                                     temp_data=data_original,
                                                                                     temp_data_tp=data_tp,
                                                                                     temp_data_fp=data_fp,
                                                                                     temp_id=row.id,
                                                                                     temp_matches=desc_sim[desc_sim['id']
                                                                                                           == row.id].iloc[0],
                                                                                     CFG=CFG,
                                                                                     embeddings_adv=embeddings_adv,
                                                                                     embeddings_adv_map=embeddings_adv_map,
                                                                                     embeddings_tp_git=embeddings_tp_git,
                                                                                     embeddings_tp_git_map=embeddings_tp_git_map,
                                                                                     embeddings_fp_git=embeddings_fp_git,
                                                                                     embeddings_fp_git_map=embeddings_fp_git_map,
                                                                                     embeddings_complete_git=embeddings_complete_git,
                                                                                     embeddings_complete_git_map=embeddings_complete_git_map,
                                                                                     text_emb_model=text_emb_model,
                                                                                     code_emb_model=code_emb_model,
                                                                                     code_emb_tokenizer=code_emb_tokenizer,
                                                                                     example_sort=example_sort,
                                                                                     example_sort_last_random=CFG.prompts.example_sort_last_random,
                                                                                     example_sort_last_random_amount=CFG.prompts.example_sort_last_random_amount,
                                                                                     random_sort_seed=idx)
                                    # Set the static dynamic examples based on the example selector
                                    dynamic_examples = CustomDynamicExamples(
                                        example_selector=example_selector,
                                        example_prompt=example_template,
                                        suffix="{input}",
                                        input_variables=["input"]
                                    )

                                    # retrieval scores
                                    dynamic_scores = dynamic_examples._get_examples()

                                    # Form the pipeline_prompt
                                    # set the input prompts to use in the pipeline
                                    input_prompts = [
                                        ("user_message", user_template)
                                    ]

                                    # create a full pipeline
                                    pipeline_prompt = PipelinePromptTemplate(
                                        final_prompt=prompt_template,
                                        pipeline_prompts=input_prompts)

                                    # check which final_prompt to use few-shot or zero-shot:
                                    # load the dynamic examples
                                    final_prompt = pipeline_prompt.format(
                                        system_prompt=system_prompt,
                                        examples=dynamic_examples.format(),
                                        details=row.details,
                                        functions=f"{list(set(ast.literal_eval(row.changed_combined_functions)))[0]}",
                                        code=f"{row.raw_patch}",
                                    )

                                    # get the response from the model
                                    response, gen_time = model_helper.simple_generate(model=model,
                                                                                      tokenizer=tokenizer,
                                                                                      input_text=final_prompt,
                                                                                      CFG=CFG)

                                    # create a single clean dataframe of the results
                                    temp_results = model_helper.combine_results(idx=idx,
                                                                                row=row,
                                                                                response=response,
                                                                                true_false=None,
                                                                                top_probs=None,
                                                                                faiss_score=[
                                                                                    x['score'] for x in dynamic_scores],
                                                                                gen_time=gen_time,
                                                                                CFG=CFG)
                                    # concat the full results
                                    full_results = pd.concat(
                                        [full_results, temp_results])

                                # save all the results
                                model_helper.aggregate_save_results(tmp_full_results=full_results,
                                                                    CFG=CFG,
                                                                    config_start_time=config_start_time,
                                                                    wb_logger=wb_logger,
                                                                    config_ver=config_ver
                                                                    )
    print("done")
