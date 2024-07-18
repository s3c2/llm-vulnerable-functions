"""Prompt template that contains few shot examples."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from langchain.prompts.base import (
    # DEFAULT_FORMATTER_MAPPING,
    StringPromptTemplate,
    check_valid_template,
    get_template_variables,
)
from langchain.prompts.chat import BaseChatPromptTemplate, BaseMessagePromptTemplate
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.prompts.prompt import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain.schema.messages import BaseMessage, get_buffer_string
from utils import embedding_helper

import numpy as np
import ast
import pandas as pd
import random


class _FewShotPromptTemplateMixin(BaseModel):
    """Prompt template that contains few shot examples."""

    examples: Optional[List[dict]] = None
    """Examples to format into the prompt.
    Either this or example_selector should be provided."""

    example_selector: Optional[BaseExampleSelector] = None
    """ExampleSelector to choose the examples to format into the prompt.
    Either this or examples should be provided."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def check_examples_and_selector(cls, values: Dict) -> Dict:
        """Check that one and only one of examples/example_selector are provided."""
        examples = values.get("examples", None)
        example_selector = values.get("example_selector", None)
        if examples and example_selector:
            raise ValueError(
                "Only one of 'examples' and 'example_selector' should be provided"
            )

        if examples is None and example_selector is None:
            raise ValueError(
                "One of 'examples' and 'example_selector' should be provided"
            )

        return values

    def _get_examples(self, **kwargs: Any) -> List[dict]:
        """Get the examples to use for formatting the prompt.

        Args:
            **kwargs: Keyword arguments to be passed to the example selector.

        Returns:
            List of examples.
        """
        if self.examples is not None:
            return self.examples
        elif self.example_selector is not None:
            return self.example_selector.select_examples(kwargs)
        else:
            raise ValueError(
                "One of 'examples' and 'example_selector' should be provided"
            )


class CustomDynamicExamples(_FewShotPromptTemplateMixin, StringPromptTemplate):
    """Prompt template that contains few shot examples."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether or not the class is serializable."""
        return False

    validate_template: bool = False
    """Whether or not to try validating the template."""

    input_variables: List[str]
    """A list of the names of the variables the prompt template expects."""

    example_prompt: PromptTemplate
    """PromptTemplate used to format an individual example."""

    suffix: str
    """A prompt template string to put after the examples."""

    example_separator: str = "\n\n"
    """String separator used to join the prefix, the examples, and suffix."""

    prefix: str = ""
    """A prompt template string to put before the examples."""

    template_format: Union[Literal["f-string"], Literal["jinja2"]] = "f-string"
    """The format of the prompt template. Options are: 'f-string', 'jinja2'."""

    @root_validator()
    def template_is_valid(cls, values: Dict) -> Dict:
        """Check that prefix, suffix, and input variables are consistent."""
        if values["validate_template"]:
            check_valid_template(
                values["prefix"] + values["suffix"],
                values["template_format"],
                values["input_variables"] + list(values["partial_variables"]),
            )
        elif values.get("template_format"):
            values["input_variables"] = [
                var
                for var in get_template_variables(
                    values["prefix"] +
                    values["suffix"], values["template_format"]
                )
                if var not in values["partial_variables"]
            ]
        return values

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def format(self, **kwargs: Any) -> str:
        """Format the prompt with the inputs.

        Args:
            **kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:

        .. code-block:: python

            prompt.format(variable1="foo")
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        # Get the examples to use.
        examples = self._get_examples(**kwargs)
        examples = [
            {k: e[k] for k in self.example_prompt.input_variables} for e in examples
        ]
        # Format the examples.
        example_strings = [
            self.example_prompt.format(**example) for example in examples
        ]
        # Create the overall template.
        pieces = [*example_strings]
        template = self.example_separator.join(
            [piece for piece in pieces if piece])

        # Format the template with the input variables.
        # return DEFAULT_FORMATTER_MAPPING[self.template_format](template, **kwargs)
        return template

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        return "few_shot"

    def save(self, file_path: Union[Path, str]) -> None:
        if self.example_selector:
            raise ValueError(
                "Saving an example selector is not currently supported")
        return super().save(file_path)


class CustomRandomSelector(BaseExampleSelector, BaseModel):
    """Select examples based on length."""

    examples: List[dict]
    """A list of the examples that the prompt template expects."""

    example_amount: Optional[int] = 1
    """A list of the examples that the prompt template expects."""

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to list."""
        self.examples.append(example)
        string_example = self.example_prompt.format(**example)

    def select_examples(self, input_variables: Optional[Dict[str, str]] = None) -> List[dict]:
        """Select random samples.

        See other examples here: https://python.langchain.com/docs/modules/model_io/prompts/example_selectors/length_based
        """
        return np.random.choice(self.examples, size=self.example_amount, replace=False)


class CustomStaticSelector(BaseExampleSelector, BaseModel):
    """Select examples based on static hand-picked items."""

    examples: List[dict]
    """A list of the examples that the prompt template expects."""

    examples_tp: List[dict]
    """A list of the examples that the prompt template expects."""

    examples_fp: List[dict]
    """A list of the examples that the prompt template expects."""

    example_amount: Optional[int] = 2
    """A list of the examples that the prompt template expects."""

    example_static_tp: List[int]
    """A list of the examples that the prompt template expects."""

    example_static_fp: List[int]
    """A list of the examples that the prompt template expects."""

    example_order: str
    """A list of the examples that the prompt template expects."""

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to list."""
        self.examples.append(example)
        string_example = self.example_prompt.format(**example)

    def select_examples(self, input_variables: Optional[Dict[str, str]] = None) -> List[dict]:
        """Select random samples.

        See other examples here: https://python.langchain.com/docs/modules/model_io/prompts/example_selectors/length_based
        """
        tp_examples = self.get_tp_static_examples()
        fp_examples = self.get_fp_static_examples()

        final_examples = []
        if self.example_order == "alt":
            for item_number in range(0, self.example_amount):
                final_examples.append(tp_examples[item_number])
                final_examples.append(fp_examples[item_number])

        return final_examples[:self.example_amount]

    def get_tp_static_examples(self) -> List[dict]:
        """Select random TP samples.
        """
        tp_samples = list(
            self.examples_tp[i] for i in self.example_static_tp)

        return tp_samples

    def get_fp_static_examples(self) -> List[dict]:
        """Select static FP samples.
        """
        fp_samples = list(
            self.examples_fp[i] for i in self.example_static_fp)

        return fp_samples


class CustomSimilarDescriptionSelector(BaseExampleSelector, BaseModel):
    """Select examples based on static hand-picked items."""

    examples: List[dict]
    """A list of the examples that the prompt template expects."""

    examples_tp: List[dict]
    """A list of the examples that the prompt template expects."""

    examples_fp: List[dict]
    """A list of the examples that the prompt template expects."""

    example_amount: Optional[int] = 2
    """A list of the examples that the prompt template expects."""

    matches: dict
    """A dictionary of the matches for a given Go-ID."""

    example_order: str
    """A list of the examples that the prompt template expects."""

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to list."""
        self.examples.append(example)
        string_example = self.example_prompt.format(**example)

    def select_examples(self, input_variables: Optional[Dict[str, str]] = None) -> List[dict]:
        """Select random samples.

        See other examples here: https://python.langchain.com/docs/modules/model_io/prompts/example_selectors/length_based
        """
        tp_examples = self.get_tp_similar_description_examples()
        fp_examples = self.get_fp_similar_description_examples()

        final_examples = []
        if self.example_order == "alt":
            for item_number in range(0, self.example_amount):
                final_examples.append(tp_examples[item_number])
                final_examples.append(fp_examples[item_number])

        return final_examples[:self.example_amount]

    def get_tp_similar_description_examples(self) -> List[dict]:
        """Select TP samples with similar descriptions.
        """
        tp_matches = []

        for match_id in ast.literal_eval(self.matches["match_id"]):
            try:
                tp_matches.append(
                    next(item for item in self.examples_tp if item["id"] == match_id))
            except:
                pass

        return tp_matches

    def get_fp_similar_description_examples(self) -> List[dict]:
        """Select FP samples with similar descriptions.
        """
        fp_matches = []

        for match_id in ast.literal_eval(self.matches["match_id"]):
            try:
                fp_matches.append(
                    next(item for item in self.examples_fp if item["id"] == match_id))
            except:
                pass

        return fp_matches


class CustomSimilarDescriptionFaissSelector(BaseExampleSelector, BaseModel):
    """Select examples with FAISS search in real-time."""
    user_input_details: str
    """The test target advisory details"""

    user_input_id: str
    """The test target advisory GO-ID"""

    user_input_raw_patch: str
    """The test target raw git-hunk patch"""

    examples: List[dict]
    """A list of the examples that the prompt template expects."""

    examples_tp: List[dict]
    """A list of the examples that the prompt template expects."""

    examples_fp: List[dict]
    """A list of the examples that the prompt template expects."""

    embed_adv: Any
    """A FAISS index of the advisory descriptions"""

    embed_adv_map: Any
    """The map for the FAISS index of the advisory"""

    embed_tp_git: Any
    """A FAISS index of the TP descriptions"""

    embed_tp_git_map: Any
    """The map for the FAISS index of the TP descriptions"""

    embed_fp_git: Any
    """A FAISS index of the FP descriptions"""

    embed_fp_git_map: Any
    """The map for the FAISS index of the FP descriptions"""

    text_embedding_model: Any
    """The embedding model used to generate embeddings"""

    example_amount: Optional[int] = 2
    """A list of the examples that the prompt template expects."""

    matches: dict
    """A dictionary of the matches for a given Go-ID."""

    example_order: str
    """A list of the examples that the prompt template expects."""

    example_sort: Optional[str] = "descending"
    """Reverse the order in which examples are returned based on similarity.
    descending = Most similar first (i.e., most similar first example in prompt)
    ascending = Most similar last (i.e., most similar closer to user input in prompt)
    """

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to list."""
        self.examples.append(example)
        string_example = self.example_prompt.format(**example)

    def select_examples(self, input_variables: Optional[Dict[str, str]] = None) -> List[dict]:
        """Select random samples.

        See other examples here: https://python.langchain.com/docs/modules/model_io/prompts/example_selectors/length_based
        """

        # generate embeddings advisory details for the search
        test_adv_embeddings = embedding_helper.text_embeddings(model=self.text_embedding_model,
                                                               text=[self.user_input_details])

        # Search for embeddings most similar to query using FAISS
        score, idx = self.embed_adv.search(
            test_adv_embeddings.reshape(1, -1), self.embed_adv.ntotal)

        # remove the test case from the FAISS search
        update_score, update_idx = self.remove_test_case_from_faiss_results(
            orig_scores=score, orig_idx=idx
        )

        tp_examples = self.get_tp_similar_description_examples(
            faiss_score_idx=update_idx, faiss_score=update_score
        )
        fp_examples = self.get_fp_similar_description_examples(
            faiss_score_idx=update_idx, faiss_score=update_score
        )

        final_examples = []
        # we can't do alt_dependent for description, becuase it could have true/false examples
        if self.example_order == "alt" or self.example_order == "alt_dependent":
            for item_number in range(0, self.example_amount):
                final_examples.append(tp_examples[item_number])
                final_examples.append(fp_examples[item_number])

        if self.example_sort == "ascending":
            # reverse how the examples would appear in prompt
            # ascending = most similar closer to user input
            return final_examples[:self.example_amount][::-1]
        elif self.example_sort == "descending":
            # descending = most similar at top of prompt
            return final_examples[:self.example_amount]

    def remove_test_case_from_faiss_results(self,
                                            orig_scores,
                                            orig_idx):
        """Removes the test case 
        that we're searching from in the returned results
        """

        # get the index location of the test target
        index_map_loc = self.embed_adv_map['id'].tolist().index(
            self.user_input_id)

        # get the index location of the index_map_loc
        index_identical_loc = list(orig_idx[0]).index(index_map_loc)

        # remove the test target from the idx list
        idx_list = orig_idx[0].tolist()
        idx_list.remove(index_map_loc)

        # remove the test target based on the prior index location from scores
        score_list = orig_scores[0].tolist()
        score_list.pop(index_identical_loc)

        return score_list, idx_list

    def get_tp_similar_description_examples(self, faiss_score_idx, faiss_score) -> List[dict]:
        """Select TP samples with similar descriptions.
        """
        tp_matches = []

        # for temp_id in self.embed_adv_map.iloc[faiss_score_idx]["id"]:
        #     try:
        #         tp_matches.append(
        #             next(item for item in self.examples_tp if item["id"] == temp_id))
        #     except:
        #         pass

        # return tp_matches

        tp_matches = []

        # create a temp_examples
        temp_examples = pd.DataFrame(self.examples_tp.copy())

        # create a temporary faiss index
        temp_faiss = self.embed_adv_map.iloc[faiss_score_idx].copy(
        ).reset_index(drop=False).set_index('index')

        # add the scores
        temp_faiss['score'] = faiss_score

        # this is the match after the faiss_score_idx has been removed from test results
        # so tp_matches will be less than self.embed_tp_git_map & temp_examples
        tp_matches = temp_faiss.reset_index().merge(
            temp_examples[['id', 'details', 'code', 'functions',
                           'label']].drop_duplicates(),
            left_on=['id', 'details'],
            right_on=['id', 'details'],
            how='left',
            sort=False).set_index('index').dropna()

        # convert to a list of dictionaries
        tp_matches_list = tp_matches[['id', 'details',
                                      'functions', 'code',
                                      'label', 'score']].to_dict("records")

        return tp_matches_list

    def get_fp_similar_description_examples(self, faiss_score_idx, faiss_score) -> List[dict]:
        """Select FP samples with similar descriptions.
        """
        # fp_matches = []

        # for temp_id in self.embed_adv_map.iloc[faiss_score_idx]["id"]:
        #     try:
        #         fp_matches.append(
        #             next(item for item in self.examples_fp if item["id"] == temp_id))
        #     except:
        #         pass

        # return fp_matches

        fp_matches = []

        # create a temp_examples
        temp_examples = pd.DataFrame(self.examples_fp.copy())

        # create a temporary faiss index
        temp_faiss = self.embed_adv_map.iloc[faiss_score_idx].copy(
        ).reset_index(drop=False).set_index('index')

        # add the scores
        temp_faiss['score'] = faiss_score

        # this is the match after the faiss_score_idx has been removed from test results
        # so tp_matches will be less than self.embed_fp_git_map & temp_examples
        fp_matches = temp_faiss.reset_index().merge(
            temp_examples[['id', 'details', 'code', 'functions',
                           'label']].drop_duplicates(),
            left_on=['id', 'details'],
            right_on=['id', 'details'],
            how='left',
            sort=False).set_index('index').dropna()

        # convert to a list of dictionaries
        fp_matches_list = fp_matches[['id', 'details',
                                      'functions', 'code',
                                      'label', 'score']].to_dict("records")

        return fp_matches_list


class CustomSimilarCodeFaissSelector(BaseExampleSelector, BaseModel):
    """Select examples with FAISS search in real-time."""
    user_input_details: str
    """The test target advisory details"""

    user_input_id: str
    """The test target advisory GO-ID"""

    user_input_raw_patch: str
    """The test target raw git-hunk patch"""

    examples: List[dict]
    """A list of the examples that the prompt template expects."""

    examples_tp: List[dict]
    """A list of the examples that the prompt template expects."""

    examples_fp: List[dict]
    """A list of the examples that the prompt template expects."""

    embed_adv: Any
    """A FAISS index of the advisory descriptions"""

    embed_adv_map: Any
    """The map for the FAISS index of the advisory"""

    embed_tp_git: Any
    """A FAISS index of the TP descriptions"""

    embed_tp_git_map: Any
    """The map for the FAISS index of the TP descriptions"""

    embed_fp_git: Any
    """A FAISS index of the FP descriptions"""

    embed_fp_git_map: Any
    """The map for the FAISS index of the FP descriptions"""

    embed_complete_git: Any
    """A FAISS index of the FP descriptions"""

    embed_complete_git_map: Any
    """The map for the FAISS index of the FP descriptions"""

    code_embedding_model: Any
    """The code embedding model used to generate embeddings"""

    code_embedding_tokenizer: Any
    """The code embedding tokenizer used to generate embeddings"""

    example_amount: Optional[int] = 2
    """A list of the examples that the prompt template expects."""

    matches: dict
    """A dictionary of the matches for a given Go-ID."""

    example_order: str
    """A list of the examples that the prompt template expects."""

    example_sort: Optional[str] = "descending"
    """Reverse the order in which examples are returned based on similarity.
    descending = Most similar first (i.e., most similar first example in prompt)
    ascending = Most similar last (i.e., most similar closer to user input in prompt)
    """

    example_sort_last_random: Optional[bool] = False
    """Randomly sort the last two examples
    True = Randomly sort the last two examples
    False = Do not randomly sort the last two examples
    """

    example_sort_last_random_amount: Optional[int] = 2
    """example_sort_last_random must be set to True for this value to work
    The amount of examples at the end of the prompt you want to randomly sort.
    """

    random_sort_seed: Optional[int] = 42
    """For reproducibility when doing random sort. We can't just have a single key,
    otherwise the sort would be the same each time for the prompts.
    """

    static_sort_seed: Optional[int] = 42
    """For reproducibility when doing random sort. We need a single key,
    for the randomStatic feature.
    """

    CFG: Optional[Any]
    """The CFG"""

    results: Optional[Any]
    """Results that you can parse later without invoking again"""

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to list."""
        self.examples.append(example)
        string_example = self.example_prompt.format(**example)

    def select_examples(self, input_variables: Optional[Dict[str, str]] = None) -> List[dict]:
        """Select random samples.

        See other examples here: https://python.langchain.com/docs/modules/model_io/prompts/example_selectors/length_based
        """

        # generate embeddings advisory details for the search
        test_code_embeddings = embedding_helper.code_embeddings(model=self.code_embedding_model,
                                                                tokenizer=self.code_embedding_tokenizer,
                                                                code=self.user_input_raw_patch)

        # Selects based on similarities for all git-hunks in the DB independently
        # Does not take into account if the returned values are True or False, only the most similar code
        # Regardless of the label
        if self.example_order == "complete":
            # search the complete embeddings
            comp_score, comp_idx = self.embed_complete_git.search(
                test_code_embeddings.reshape(1, -1), self.embed_complete_git.ntotal)

            # remove the test case with same GO-ID from the FAISS search
            comp_update_score, comp_update_idx = self.remove_test_case_from_faiss_results(
                orig_scores=comp_score, orig_idx=comp_idx, label='complete'
            )

            # get the examples for the matches
            comp_examples = self.get_complete_similar_code_examples(
                faiss_score_idx=comp_update_idx, faiss_score=comp_update_score
            )

            # create a final list of dictionaries
            final_examples = []
            for item_number in range(0, self.example_amount):
                final_examples.append(comp_examples[item_number])

            if self.example_sort == "ascending":
                # reverse how the examples would appear in prompt
                # ascending = most similar closer to user input
                self.results = final_examples[::1]
                return final_examples[::-1]
            elif self.example_sort == "descending":
                # descending = most similar at top of prompt
                self.results = final_examples
                return final_examples

        ###########################################################################
        ###########################################################################
        ###########################################################################

        # Selects based on similarities for both true/false independently
        if self.example_order != "complete":
            # Search for embeddings most similar to query using FAISS
            tp_score, tp_idx = self.embed_tp_git.search(
                test_code_embeddings.reshape(1, -1), self.embed_tp_git.ntotal)

            fp_score, fp_idx = self.embed_fp_git.search(
                test_code_embeddings.reshape(1, -1), self.embed_fp_git.ntotal)

            # get a random static sort
            if self.example_order == "randomStatic":
                # # sort the tp_idx and fp_idx for consistent results
                tp_idx = np.sort(tp_idx)
                fp_idx = np.sort(fp_idx)

                # shuffle the tp_idx
                np.random.seed(self.static_sort_seed)
                np.random.shuffle(tp_idx[0])

                # shuffle the fp_idx
                np.random.seed(self.static_sort_seed)
                np.random.shuffle(fp_idx[0])

                # create a new score based on the random sort
                tp_score = [tp_score[0][i] for i in tp_idx[0]]
                tp_score = np.array([tp_score])

                # create a new score based on the random sort
                fp_score = [fp_score[0][i] for i in fp_idx[0]]
                fp_score = np.array([fp_score])

            # remove the test case with same GO-ID from the FAISS search
            tp_update_score, tp_update_idx = self.remove_test_case_from_faiss_results(
                orig_scores=tp_score, orig_idx=tp_idx, label='tp'
            )

            # remove the test case with same GO-ID from the FAISS search
            fp_update_score, fp_update_idx = self.remove_test_case_from_faiss_results(
                orig_scores=fp_score, orig_idx=fp_idx, label='fp'
            )

            # get the examples for the matches
            tp_examples = self.get_tp_similar_code_examples(
                faiss_score_idx=tp_update_idx, faiss_score=tp_update_score
            )
            fp_examples = self.get_fp_similar_code_examples(
                faiss_score_idx=fp_update_idx, faiss_score=fp_update_score
            )

            final_examples = []

            # Do this if using a true/false/true... alternating order LEADS with TRUE
            if self.example_order == "alt" or self.example_order == "randomStatic":
                for item_number in range(0, self.example_amount):
                    final_examples.append(tp_examples[item_number])
                    final_examples.append(fp_examples[item_number])

            # Leads with the example that has the highest FAISS score
            if self.example_order == "alt_dependent":
                if tp_update_score[0] > fp_update_score[0]:
                    # LEADS with the TRUE example
                    for item_number in range(0, self.example_amount):
                        final_examples.append(tp_examples[item_number])
                        final_examples.append(fp_examples[item_number])
                else:
                    # LEADS with the FALSE example
                    for item_number in range(0, self.example_amount):
                        final_examples.append(fp_examples[item_number])
                        final_examples.append(tp_examples[item_number])

            # sorts based on similarity scores
            if self.example_sort == "ascending":
                # reverse how the examples would appear in prompt
                # ascending = most similar closer to user input

                # randomly sort the last few values
                if self.example_sort_last_random:
                    # make a copy
                    final = final_examples[:self.example_amount][::-1].copy()

                    # get a list of the final amount to sort
                    sort_amount = len(final) - \
                        self.example_sort_last_random_amount
                    final_sort_amount = final[sort_amount:].copy()

                    # sort that final_sort_amount list with a seed
                    random.Random(self.random_sort_seed).shuffle(
                        final_sort_amount)

                    # overwrite that final amount with the random shuffle
                    final[sort_amount:] = final_sort_amount.copy()

                    return final
                else:
                    # ascending = most similar closer to user input
                    return final_examples[:self.example_amount][::-1]
            elif self.example_sort == "descending":
                # descending = most similar at top of prompt

                # randomly sort the last few values
                if self.example_sort_last_random:
                    # make a copy
                    final = final_examples[:self.example_amount].copy()

                    # get a list of the final amount to sort
                    sort_amount = len(final) - \
                        self.example_sort_last_random_amount
                    final_sort_amount = final[sort_amount:].copy()

                    # sort that final_sort_amount list with a seed
                    random.Random(self.random_sort_seed).shuffle(
                        final_sort_amount)

                    # overwrite that final amount with the random shuffle
                    final[sort_amount:] = final_sort_amount.copy()

                    return final
                else:
                    # returns a normal descending rank
                    return final_examples[:self.example_amount]
            elif self.example_sort == "random":
                # randomly sort the results instead of an asc/desc approach
                final = final_examples[:self.example_amount].copy()

                # sort that final list with a seed
                random.Random(self.random_sort_seed).shuffle(
                    final)

                # return the randomly sorted order
                return final

    def remove_test_case_from_faiss_results(self,
                                            orig_scores,
                                            orig_idx,
                                            label):
        """Removes the test case 
        that we're searching from in the returned results
        """

        # remove code samples that come from the same GO-ID
        if label == 'tp':
            # get the index location of the test target GO-ID
            index_map_loc = np.where(
                self.embed_tp_git_map['id'].array == self.user_input_id
            )[0].tolist()

        elif label == 'fp':
            # get the index location of the test target GO-ID
            index_map_loc = np.where(
                self.embed_fp_git_map['id'].array == self.user_input_id
            )[0].tolist()

        elif label == 'complete':
            # get the index location of the test target GO-ID
            index_map_loc = np.where(
                self.embed_complete_git_map['id'].array == self.user_input_id
            )[0].tolist()

        # get the index location of the index_map_loc
        # we need to find where those indexes are in the faiss results
        index_identical_loc = [list(orig_idx[0]).index(x)
                               for x in index_map_loc]

        # remove the test target from the idx list
        idx_new = np.delete(orig_idx[0], index_identical_loc).tolist()

        # remove the test target based on the prior index location from scores
        score_new = np.delete(orig_scores[0], index_identical_loc).tolist()

        return score_new, idx_new

    def get_tp_similar_code_examples(self, faiss_score_idx, faiss_score) -> List[dict]:
        """Select TP samples with similar code.
        """
        tp_matches = []

        # create a temp_examples
        temp_examples = pd.DataFrame(self.examples_tp.copy())

        # create a temporary faiss index
        temp_faiss = self.embed_tp_git_map.iloc[faiss_score_idx].copy(
        ).reset_index(drop=False).set_index('index')

        # this is the match after the faiss_score_idx has been removed from test results
        # so tp_matches will be less than self.embed_tp_git_map & temp_examples
        tp_matches = temp_faiss.reset_index().merge(
            temp_examples[['id', 'details', 'code', 'functions',
                           'label', 'cot_example_text']].drop_duplicates(subset=['id', 'details',
                                                                                 'code', 'functions',
                                                                                 'label']),
            left_on=['id', 'raw_patch'],
            right_on=['id', 'code'],
            how='left',
            sort=False).set_index('index')

        # add the scores
        tp_matches['score'] = faiss_score

        # convert to a list of dictionaries
        tp_matches_list = tp_matches[['id', 'details',
                                      'functions', 'code',
                                      'label', 'cot_example_text', 'score']].to_dict("records")

        return tp_matches_list

    def get_fp_similar_code_examples(self, faiss_score_idx, faiss_score) -> List[dict]:
        """Select FP samples with similar code.
        """
        fp_matches = []

        # create a temp_examples
        temp_examples = pd.DataFrame(self.examples_fp.copy())

        # create a temporary faiss index
        temp_faiss = self.embed_fp_git_map.iloc[faiss_score_idx].copy(
        ).reset_index(drop=False).set_index('index')

        # this is the match after the faiss_score_idx has been removed from test results
        # so tp_matches will be less than self.embed_tp_git_map & temp_examples
        fp_matches = temp_faiss.reset_index().merge(
            temp_examples[['id', 'details', 'code', 'functions',
                           'label', 'cot_example_text']].drop_duplicates(subset=['id', 'details',
                                                                                 'code', 'functions',
                                                                                 'label']),
            left_on=['id', 'raw_patch'],
            right_on=['id', 'code'],
            how='left',
            sort=False).set_index('index')

        # add the scores
        fp_matches['score'] = faiss_score

        # convert to a list of dictionaries
        fp_matches_list = fp_matches[['id', 'details',
                                      'functions', 'code',
                                      'label', 'cot_example_text', 'score']].to_dict("records")

        return fp_matches_list

    def get_complete_similar_code_examples(self, faiss_score_idx, faiss_score) -> List[dict]:
        """Select Complete samples with similar code.
        """
        complete_matches = []

        # create a temp_examples
        temp_examples = pd.DataFrame(self.examples.copy())

        # create a temporary faiss index
        temp_faiss = self.embed_complete_git_map.iloc[faiss_score_idx].copy(
        ).reset_index(drop=False).set_index('index')

        # this is the match after the faiss_score_idx has been removed from test results
        # so complete_matches will be less than self.embed_complete_git_map & temp_examples
        complete_matches = temp_faiss.reset_index().merge(
            temp_examples[['id', 'details', 'code', 'functions',
                           'label', 'cot_example_text']].drop_duplicates(),
            left_on=['id', 'raw_patch'],
            right_on=['id', 'code'],
            how='left',
            sort=False).set_index('index')

        # add the scores
        complete_matches['score'] = faiss_score

        # convert to a list of dictionaries
        complete_matches_list = complete_matches[['id', 'details',
                                                  'functions', 'code',
                                                  'label', 'cot_example_text', 'score']].to_dict("records")

        return complete_matches_list


def create_example_selector(example_type, example_amount, example_template, example_order,
                            temp_data, temp_data_tp, temp_data_fp, temp_id, temp_matches, CFG):

    # convert the temp_data DF to a dictionary
    examples = convert_data_to_dict_examples(temp_data=temp_data.copy())
    examples_tp = convert_data_to_dict_examples(temp_data=temp_data_tp.copy())
    examples_fp = convert_data_to_dict_examples(temp_data=temp_data_fp.copy())

    if example_type == "static":
        example_selector = CustomStaticSelector(
            # The examples it has available to choose from.
            examples=examples,
            examples_tp=examples_tp,
            examples_fp=examples_fp,
            # The PromptTemplate being used to format the examples.
            example_prompt=example_template,
            # The maximum length that the formatted examples should be.
            example_amount=example_amount,
            example_static_tp=CFG.prompts.example_static_tp,
            example_static_fp=CFG.prompts.example_static_fp,
            example_order=example_order
        )

    elif example_type == "random":
        example_selector = CustomRandomSelector(
            # The examples it has available to choose from.
            examples=examples,
            # The PromptTemplate being used to format the examples.
            example_prompt=example_template,
            # The maximum length that the formatted examples should be.
            # Length is measured by the get_text_length function below.
            example_amount=example_amount,
        )

    elif example_type == "description":
        example_selector = CustomSimilarDescriptionSelector(
            # The examples it has available to choose from.
            examples=examples,
            examples_tp=examples_tp,
            examples_fp=examples_fp,
            # The PromptTemplate being used to format the examples.
            example_prompt=example_template,
            # The maximum length that the formatted examples should be.
            example_amount=example_amount,
            matches=temp_matches,
            example_order=example_order
        )

    return example_selector


def create_example_selector_faiss(user_input_details, user_input_id, user_input_raw_patch, example_type, example_amount, example_template, example_order,
                                  temp_data, temp_data_tp, temp_data_fp, temp_id, temp_matches, CFG,
                                  embeddings_adv, embeddings_adv_map, embeddings_tp_git, embeddings_tp_git_map,
                                  embeddings_fp_git, embeddings_fp_git_map, embeddings_complete_git, embeddings_complete_git_map,
                                  text_emb_model, code_emb_model,
                                  code_emb_tokenizer, example_sort,
                                  example_sort_last_random, example_sort_last_random_amount, random_sort_seed):

    # convert the temp_data DF to a dictionary
    examples = convert_data_to_dict_examples(temp_data=temp_data.copy())
    examples_tp = convert_data_to_dict_examples(temp_data=temp_data_tp.copy())
    examples_fp = convert_data_to_dict_examples(temp_data=temp_data_fp.copy())

    if example_type == "static":
        example_selector = CustomStaticSelector(
            # The examples it has available to choose from.
            examples=examples,
            examples_tp=examples_tp,
            examples_fp=examples_fp,
            # The PromptTemplate being used to format the examples.
            example_prompt=example_template,
            # The maximum length that the formatted examples should be.
            example_amount=example_amount,
            example_static_tp=CFG.prompts.example_static_tp,
            example_static_fp=CFG.prompts.example_static_fp,
            example_order=example_order,
            example_sort=example_sort,
            example_sort_last_random=example_sort_last_random,
            example_sort_last_random_amount=example_sort_last_random_amount
        )

    elif example_type == "random":
        example_selector = CustomRandomSelector(
            # The examples it has available to choose from.
            examples=examples,
            # The PromptTemplate being used to format the examples.
            example_prompt=example_template,
            # The maximum length that the formatted examples should be.
            # Length is measured by the get_text_length function below.
            example_amount=example_amount,
        )

    elif example_type == "description":
        example_selector = CustomSimilarDescriptionFaissSelector(
            user_input_details=user_input_details,
            user_input_id=user_input_id,
            user_input_raw_patch=user_input_raw_patch,
            # The examples it has available to choose from.
            examples=examples,
            examples_tp=examples_tp,
            examples_fp=examples_fp,
            # The PromptTemplate being used to format the examples.
            example_prompt=example_template,
            # The maximum length that the formatted examples should be.
            example_amount=example_amount,
            matches=temp_matches,
            example_order=example_order,
            example_sort=example_sort,
            example_sort_last_random=example_sort_last_random,
            example_sort_last_random_amount=example_sort_last_random_amount,
            # Add the embeddings
            embed_adv=embeddings_adv,
            embed_adv_map=embeddings_adv_map,
            embed_tp_git=embeddings_tp_git,
            embed_tp_git_map=embeddings_tp_git_map,
            embed_fp_git=embeddings_fp_git,
            embed_fp_git_map=embeddings_fp_git_map,
            text_embedding_model=text_emb_model
        )
    elif example_type == "git-hunk":
        example_selector = CustomSimilarCodeFaissSelector(
            user_input_details=user_input_details,
            user_input_id=user_input_id,
            user_input_raw_patch=user_input_raw_patch,
            # The examples it has available to choose from.
            examples=examples,
            examples_tp=examples_tp,
            examples_fp=examples_fp,
            # The PromptTemplate being used to format the examples.
            example_prompt=example_template,
            # The maximum length that the formatted examples should be.
            example_amount=example_amount,
            matches=temp_matches,
            example_order=example_order,
            example_sort=example_sort,
            example_sort_last_random=example_sort_last_random,
            example_sort_last_random_amount=example_sort_last_random_amount,
            random_sort_seed=random_sort_seed,
            # Add the embeddings
            embed_adv=embeddings_adv,
            embed_adv_map=embeddings_adv_map,
            embed_tp_git=embeddings_tp_git,
            embed_tp_git_map=embeddings_tp_git_map,
            embed_fp_git=embeddings_fp_git,
            embed_fp_git_map=embeddings_fp_git_map,
            embed_complete_git=embeddings_complete_git,
            embed_complete_git_map=embeddings_complete_git_map,
            code_embedding_model=code_emb_model,
            code_embedding_tokenizer=code_emb_tokenizer,
            CFG=CFG
        )

    return example_selector


def convert_data_to_dict_examples(temp_data) -> dict:
    """Converts a dataframe to a list of dictionaries for furture example parsing

    Args:
        temp_data (_type_): Data DF

    Returns:
        dict: details, function, code, label dictionary
    """
    # columns to target from the dataframe
    # columns = ['details', 'changed_combined_functions',
    #            'raw_patch', 'label_combined_unique']
    columns = ['id', 'details', 'changed_combined_functions',
               'raw_patch', 'label_combined_unique', 'cot_example_text']

    # convert strings to a clean format
    temp_data['label_combined_unique'] = temp_data.apply(
        lambda x: ast.literal_eval(x['label_combined_unique'])[0],
        axis=1
    )

    temp_data['changed_combined_functions'] = temp_data.apply(
        lambda x: list(set(ast.literal_eval(
            x['changed_combined_functions'])))[0],
        axis=1
    )

    # create a new df with only the target columns
    temp_df = temp_data[columns].copy()
    # temp_df.columns = ['details', 'functions', 'code', 'label']
    temp_df.columns = ['id', 'details', 'functions',
                       'code', 'label', 'cot_example_text']

    # convert to a list of dictionaries
    temp_dict = temp_df.to_dict('records')

    return temp_dict


def clean_advisory_details(details: str) -> str:
    """Removes extra spaces and new lines from the advisory details

    Args:
        details (str): Advisory details

    Returns:
        str: Clean Advisory details
    """

    # Remove new lines
    no_new_lines = details.replace("\n", " ")

    # Remove extra spaces & leading/ending spaces
    clean_details = ' '.join(no_new_lines.strip().split())

    return clean_details
