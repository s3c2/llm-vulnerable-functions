"""
Helper scripts to parse a VFC

Install patchparser from the repository:
    - Uninstall patchparser.
    - Install patchparser from the local path (wherever you installed it): pip3 install ./patchparser/.
    - Set your GitHub Token environment variable -> export GITHUB_TOKEN="your GH token"
"""

import os
import subprocess
import git
import patchparser
import pathlib
import pandas as pd
from packaging.version import Version, parse
import datetime
from tree_sitter import Language, Parser


def git_diff(clone_path: str, commit_sha: str, df=False) -> dict:
    """Obtains the git diff information using patchparser
    Info: https://github.com/tdunlap607/patchparser

    Args:
        clone_path (str): Location of source code
        commit_sha (_type_): Target commit to parse
        df (bool): If you want a pandas DF back

    Returns:
        (dict): Dictionary of git diff info
    """

    repo_owner = clone_path.split("/")[-3]
    repo_name = clone_path.split("/")[-2]

    diff = patchparser.github_parser_local.commit_local_updated(
        repo_owner=repo_owner,
        repo_name=repo_name,
        sha=commit_sha,
        base_repo_path=clone_path,
    )

    if df:
        diff_df = pd.DataFrame(diff)

        # calculate the line numbers modified in relational to the original file
        diff_df["original_modified_lines"] = diff_df.apply(
            lambda x: git_changed_original_lines(
                raw_patch=x["raw_patch"],
                original_line_start=x["original_line_start"],
            )
            if x["deletions"] > 0
            else None,
            axis=1,
        )

        # calculate the line numbers modified in relation to the fresh commit file
        diff_df["new_modified_lines"] = diff_df.apply(
            lambda x: git_changed_modified_lines(
                raw_patch=x["raw_patch"],
                modified_line_start=x["modified_line_start"],
            )
            if x["additions"] > 0
            else None,
            axis=1,
        )

        return diff_df
    else:
        return diff


def git_changed_original_lines(raw_patch: str, original_line_start: int) -> list:
    """Calculates the line numbers that were changed of the original file during the commit

    Args:
        raw_patch (str): Diff hunk from git_diff
        original_line_start (int): Line number start of diff hunk

    Returns:
        list: List of modified line numbers from the original file
    """
    # split the diff hunk based on newlines
    original = raw_patch.splitlines()

    try:
        # original lines can only be removed, we handled additions in git_changed_modified_lines
        original_lines = [x for x in original if x[0] != "+"][1:]

        # get the line numbers that were removed
        original_line_numbers = [
            idx + original_line_start
            for idx, i in enumerate(original_lines)
            if i[0] == "-"
        ]
    except:
        original_lines = [x for x in original[1:] if x[0] != "+"]

        # get the line numbers that were removed
        original_line_numbers = [
            idx + original_line_start
            for idx, i in enumerate(original_lines)
            if i[0] == "-"
        ]

    return original_line_numbers


def git_changed_modified_lines(raw_patch: str, modified_line_start: int) -> list:
    """Calculates the line numbers that were changed of the modified file during the commit

    Args:
        raw_patch (str): Diff hunk from git_diff
        modified_line_start (int): Line number start of diff hunk

    Returns:
        list: List of modified line numbers
    """
    # split the diff hunk based on newlines
    modified = raw_patch.splitlines()

    # modified lines can only be added, we handled removals in git_changed_original_lines
    try:
        modified_lines = [x for x in modified[1:] if x[0] != "-"]
    except:
        print("wait")

    # get the line numbers that were added
    modified_line_numbers = [
        idx + modified_line_start for idx, i in enumerate(modified_lines) if i[0] == "+"
    ]

    return modified_line_numbers


def match_functions(
    temp_functions: pd.DataFrame, temp_file_name: str, temp_lines: list
):
    """Matches changed line numbers to functions

    Args:
        temp_functions (_type_): _description_
        temp_file_name (_type_): _description_
        temp_lines (_type_): _description_
    """

    # match the file
    temp_file_match = temp_functions[
        temp_functions["file_name"].str.contains(str(temp_file_name))
    ]

    temp_matches = pd.DataFrame()
    temp_functions_list = []

    # make sure temp_lines is of type list
    # TODO: solve the issue where the modified file only includes deletions
    # Example: https://github.com/theupdateframework/go-tuf/commit/ed6788e710fc3093a7ecc2d078bf734c0f200d8d

    # cline/errors.go produces a int(0) for the new_modified_lines
    if isinstance(temp_lines, list):
        for line in temp_lines:
            # match the line number between the StartLine and EndLine of functions
            temp_line_match = temp_file_match[
                (temp_file_match["start_line"] <= line)
                & (temp_file_match["end_line"] >= line)
            ].copy()

            # add the file_name/line number match back to the df
            temp_line_match["file_name"] = temp_file_name
            temp_line_match["line"] = line

            # append to to complete df
            temp_matches = pd.concat([temp_matches, temp_line_match])

            # create a list of the function names
            if len(temp_line_match) > 0:
                temp_functions_list.append(
                    temp_line_match.iloc[0].function_name)

    return temp_matches, temp_functions_list


def extract_functions_file(tree_sitter_path: str, file_path: str, return_df: False):
    """
    Extracts the function name, start line, and end line.
    By default returns a list, unless return_df is set.

    For more info on tree-sitter: https://github.com/tree-sitter/py-tree-sitter

    Args:
        file_path (str): Path to file you want to parse
        return_df (False): Determine if you want to return a DataFrame
    """

    # get the file extension type
    file_extension = pathlib.Path(file_path).suffix

    if file_extension == ".py":
        # set you desired language
        prog_language = Language(
            f"{tree_sitter_path}build/my-languages.so", "python")
    elif file_extension == ".go":
        # set you desired language
        prog_language = Language(
            f"{tree_sitter_path}build/my-languages.so", "go")

    # create a parser
    parser = Parser()
    parser.set_language(prog_language)

    # # you have to read it the file in as type bytes
    file = open(file_path, "rb").read()

    # parse the file
    tree = parser.parse(file)

    if return_df:
        functions = pd.DataFrame()
    else:
        functions = []

    # iterate through the tree
    for children in tree.root_node.children:
        if children.grammar_name == "function_definition":
            for named_children in children.named_children:
                if named_children.type == "identifier":
                    # set the function name, start/end lines
                    tmp_func_name = named_children.text.decode("utf-8")
                    tmp_func_start_line = children.start_point[0]
                    tmp_func_end_line = children.end_point[0]

                    tmp_functions = [tmp_func_name,
                                     tmp_func_start_line,
                                     tmp_func_end_line]

                    if return_df:
                        tmp_df = pd.DataFrame([tmp_functions],
                                              columns=["function_name",
                                                       "start_line",
                                                       "end_line"])

                        functions = pd.concat([functions, tmp_df])
                    else:
                        functions.append(tmp_functions)

    return functions


def extract_functions_go_file(tree_sitter_path: str, file_path: str, repo_owner: str, repo_name: str, file_name: str, sha: str, return_df=False):
    """
    Extracts the function name, start line, and end line.
    By default returns a list, unless return_df is set.

    For more info on tree-sitter: https://github.com/tree-sitter/py-tree-sitter

    Args:
        file_path (str): Path to file you want to parse
        return_df (False): Determine if you want to return a DataFrame
        repo_owner (str): Repo owner
        repo_name (str): Repo Name
        file_name (str): Target file name
        sha (str): target SHA
        return_df (bool, optional): Allows you to return a pandas df. Defaults False
    """

    # get the file extension type
    file_extension = pathlib.Path(file_path).suffix

    if file_extension == ".go":
        # set you desired language
        prog_language = Language(
            f"{tree_sitter_path}build/my-languages.so", "go")

        # create a parser
        parser = Parser()
        parser.set_language(prog_language)

        # # you have to read it the file in as type bytes
        file = open(file_path, "rb").read()

        # parse the file
        tree = parser.parse(file)

        if return_df:
            functions = pd.DataFrame()
        else:
            functions = []

        # iterate through the tree
        for children in tree.root_node.children:
            # reset the declaration and method name before each child
            declaration_name = None
            method_name = None
            if children.grammar_name == "method_declaration":
                # we only need the first two children in a method_declaration for Go
                for named_children in children.named_children[:2]:
                    if named_children.type == "parameter_list":
                        for params in named_children.children:
                            if params.type == "parameter_declaration":
                                for declaration in params.children:
                                    # handle both pointer and type identifiers
                                    if declaration.type == "type_identifier":
                                        declaration_name = declaration.text.decode(
                                            "utf-8")
                                    elif declaration.type == "pointer_type":
                                        declaration_name = declaration.text.decode(
                                            "utf-8").strip("*")
                                        # print(declaration_name)
                    if named_children.type == "field_identifier":
                        # set the function name, start/end lines
                        method_name = named_children.text.decode(
                            "utf-8")
                        tmp_func_start_line = children.start_point[0]
                        tmp_func_end_line = children.end_point[0]

                    # only take the fully qualified function Name
                    if declaration_name is not None and method_name is not None:
                        tmp_functions = [repo_owner,
                                         repo_name,
                                         sha,
                                         file_name,
                                         f"{declaration_name}.{method_name}",
                                         tmp_func_start_line,
                                         tmp_func_end_line]

                        # append to the final functions
                        if return_df:
                            tmp_df = pd.DataFrame([tmp_functions],
                                                  columns=["repo_owner",
                                                           "repo_name",
                                                           "sha",
                                                           "file_name",
                                                           "function_name",
                                                           "start_line",
                                                           "end_line"])

                            functions = pd.concat([functions, tmp_df])
                        else:
                            functions.append(tmp_functions)
            ####################################################################
            # handle FUNCTION declarations
            ####################################################################
            if children.grammar_name == "function_declaration":
                for named_children in children.named_children:
                    if named_children.type == "identifier":
                        # set the function name, start/end lines
                        tmp_func_name = named_children.text.decode("utf-8")
                        tmp_func_start_line = children.start_point[0]
                        tmp_func_end_line = children.end_point[0]

                        tmp_functions = [repo_owner,
                                         repo_name,
                                         sha,
                                         file_name,
                                         tmp_func_name,
                                         tmp_func_start_line,
                                         tmp_func_end_line]

                        if return_df:
                            tmp_df = pd.DataFrame([tmp_functions],
                                                  columns=["repo_owner",
                                                           "repo_name",
                                                           "sha",
                                                           "file_name",
                                                           "function_name",
                                                           "start_line",
                                                           "end_line"])

                            functions = pd.concat([functions, tmp_df])
                        else:
                            functions.append(tmp_functions)

        return functions

    else:
        if return_df:
            functions = pd.DataFrame()
        else:
            functions = []

        # return an empty file if it isn't a Go File
        tmp_functions = [repo_owner,
                         repo_name,
                         sha,
                         file_name,
                         None,
                         None,
                         None]

        if return_df:
            tmp_df = pd.DataFrame([tmp_functions],
                                  columns=["repo_owner",
                                           "repo_name",
                                           "sha",
                                           "file_name",
                                           "function_name",
                                           "start_line",
                                           "end_line"])

            functions = pd.concat([functions, tmp_df])
        else:
            functions.append(tmp_functions)

        return functions
