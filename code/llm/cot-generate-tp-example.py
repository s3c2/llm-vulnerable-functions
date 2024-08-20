"""
Generate TP examples for COT
I have already generated the data for you in /data/patchparser-data.tar.gz if you do not want to run this.

To execute, make sure the code/llm/cfgs/v1_mixtral_cot_generate_tp.yaml is updated with your paths!!!
Install the requirements.txt file.

Then: python3 ./code/llm/cot-generate-Tp-example.py v1_mixtral_cot_generate_fp
"""
import sys  # noqa
sys.path.append('.')   # noqa

# Custom Modules
from pathlib import Path
import pandas as pd
import torch
import ast
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import model_helper
from langchain.prompts import load_prompt
from langchain.prompts.pipeline import PipelinePromptTemplate
import warnings


# To ignore all user warnings
warnings.filterwarnings('ignore', category=UserWarning)

# check if in debug mode:
# is_debugger = logger_helper.debugger_is_active()
is_debugger = False

# if debugger is active we can manually change values
if is_debugger:
    config_file = "v1_mixtral_cot_generate_tp"
    debug_test_amount = 1
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
        temp_cfg=CFG, load_cot_examples=False)

    ###############################################################################
    ###############################################################################
    # Load the models
    # Left padding: https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
    tokenizer = AutoTokenizer.from_pretrained(
        f"{CFG.models.base}{CFG.models.model_name}", padding_side="left")

    # # Most LLMs don't have a pad token by default
    tokenizer.pad_token = tokenizer.eos_token

    if CFG.gen_cfg.load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(f"{CFG.models.base}{CFG.models.model_name}",
                                                     device_map=CFG.gen_cfg.device_map,
                                                     load_in_8bit=True,
                                                     torch_dtype=torch.float16)
    elif CFG.gen_cfg.load_in_4bit:
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

    # only run a subset in debug mode or test mode
    if is_debugger or CFG.data.test:
        # data = data[:CFG.data.test_amount].reset_index(drop=True)
        data_tp = data_tp[:CFG.data.test_amount].reset_index(drop=True)
        data_fp = data_fp[:CFG.data.test_amount].reset_index(drop=True)

    ###################################################################################
    # Load the prompt template
    # set each path type based
    prompt_template_path = f"{CFG.paths.template}{CFG.prompts.template_zero_shot}"
    # set a system prompt type path for later
    system_prompt_type_path = CFG.paths.zero_shot_system_prompts

    print(f"Using prompt template: {prompt_template_path}")
    prompt_template = load_prompt(prompt_template_path)

    ###################################################################################
    # Load the system prompts template
    # set each path type based
    # set the system_prompt_path
    system_prompt_path = f"{system_prompt_type_path}{CFG.prompts.system_prompt_type[0]}-system-v{CFG.prompts.system_prompt_version}.txt"

    print(f"Using system prompt: {system_prompt_path}")
    system_prompt = Path(system_prompt_path).read_text()

    ###################################################################################
    # Load the user prompts template for the paradigm
    # set the systemp_prompt_path
    user_template_path = f"{CFG.paths.user_template}{CFG.prompts.system_prompt_type[0]}-user-v{CFG.prompts.user_template_version}.yaml"

    print(f"Using user template: {user_template_path}")
    user_template = load_prompt(user_template_path)

    # create a holder for the TP data
    data_tp_examples = pd.DataFrame()

    for idx, row in data_tp.iterrows():
        print(f"\n\n{idx}/{len(data_tp)}")
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

        print("\n\n~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+\n"
              "~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+\n")
        print(final_prompt)
        print(f"\n\n============== RESPONSE ================\n{response}")
        print("~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+\n"
              "~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+\n\n")

        # create a temporary DF
        temp_tp = pd.DataFrame(row).T
        temp_tp['cot_example_text'] = response.replace('</s>', '').strip()

        # append back to the data tp examples
        data_tp_examples = pd.concat([data_tp_examples, temp_tp])

    print(
        f"Saving TP COT examples to: {CFG.paths.patchparser}{CFG.data.tp_cot_examples}")

    data_tp_examples = data_tp_examples.reset_index(drop=True)

    data_tp_examples.to_csv(f"{CFG.paths.patchparser}{CFG.data.tp_cot_examples}",
                            encoding='utf-8', index=False)

    print("done")
