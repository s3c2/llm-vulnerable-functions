import pandas as pd
import json
import re
import yaml


def parse_ghsa(ghsa_json_filename: str) -> pd.DataFrame:
    """The purpose of this function is to open and parse GHSA files in ghsa_path

    Args:
        ghsa_json_filename (str): Full path to GHSA json

    Returns:
        pd.DataFrame: _description_
    """
    # Open the json
    with open(ghsa_json_filename, 'r') as f:
        ghsa_json = json.load(f)

    temp_df = pd.DataFrame()

    # set fixed event
    try:
        events = [list(i.keys())[0]
                  for i in ghsa_json['affected'][0]['ranges'][0]['events']]
        if 'fixed' in events:
            fixed = True
            fixed_count = len(ghsa_json['affected'])
            if "GHSA-fcg8-mg9g-6hc4" in str(ghsa_json_filename):
                print("wait")
            # get unique package
            fix_package = []
            fix_package_ver = []
            fix_ver = []
            for fix in ghsa_json['affected']:
                fix_package.append(fix['package']['name'])
                for ver in fix['ranges']:
                    fix_ver.append(ver["events"][1]["fixed"])
                    fix_package_ver.append(
                        f"{fix['package']['name']}||{ver['events'][1]['fixed']}")

            fix_package = list(set(fix_package))
            fix_package_ver = list(set(fix_package_ver))
            fix_ver = list(set(fix_ver))

            fix_package_len = len(fix_package)
            fix_package_ver_len = len(fix_package_ver)
            fix_ver_len = len(fix_ver)

        else:
            fixed = False
            fixed_count = 0

            fix_package = []
            fix_package_ver = []
            fix_ver = []

            fix_package_len = 0
            fix_package_ver_len = 0
            fix_ver_len = 0
    except:
        fixed = False
        fixed_count = 0

        fix_package = []
        fix_package_ver = []
        fix_ver = []

        fix_package_len = 0
        fix_package_ver_len = 0
        fix_ver_len = 0

    # set CWEs
    try:
        cwes_len = len(ghsa_json['database_specific']['cwe_ids'])
        cwes = ghsa_json['database_specific']['cwe_ids']
    except:
        cwes_len = 0
        cwe = []

    # extract the reference URLs
    urls = [ref["url"] if "url" in ref else None for ref in ghsa_json["references"]]

    # create a clean DF to return
    if len(urls) > 0:
        temp_df = pd.DataFrame(urls, columns=['url'])
        temp_df['id'] = ghsa_json["id"]
        temp_df['fixed'] = fixed
        temp_df['fixed_count'] = fixed_count
        temp_df['fix_package'] = str(fix_package)
        temp_df['fix_package_len'] = fix_package_len
        temp_df['fix_package_ver'] = str(fix_package_ver)
        temp_df['fix_package_ver_len'] = fix_package_ver_len
        temp_df['fix_ver'] = str(fix_ver)
        temp_df['fix_ver_len'] = fix_ver_len
        temp_df['cwe'] = cwes

        # regex to search for the github commit patch links
        github_commit_regex = r"github.com/(.*)/commit/(.*)"

        temp_df["patch_link"] = temp_df.apply(
            lambda x: True if re.search(
                github_commit_regex, str(x["url"]).strip()) else False,
            axis=1)

        # regex to search for the github pull links
        github_pull_regex = r"github.com/(.*)/pull/(.*)"

        temp_df["pull_link"] = temp_df.apply(
            lambda x: True if re.search(
                github_pull_regex, str(x["url"]).strip()) else False,
            axis=1)

        # regex to search for the github issue links
        github_issues_regex = r"github.com/(.*)/issues/(.*)"

        temp_df["issue_link"] = temp_df.apply(
            lambda x: True if re.search(
                github_issues_regex, str(x["url"]).strip()) else False,
            axis=1)

    return temp_df, ghsa_json


def parse_osv(osv_json_filename: str, osv_schema: dict) -> dict:
    """The purpose of this function is to open and parse OSV formatted files
    OSV Schema: https://ossf.github.io/osv-schema/

    Args:
        osv_json_filename (str): File Location of OSV JSON to parse
        osv_schema (dict): https://github.com/ossf/osv-schema/blob/main/validation/schema.json

    Returns:
        dict: _description_
    """
    # Open the json
    with open(osv_json_filename, "r") as f:
        osv_json = json.load(f)

    # OSV Properities or the keys of the schema
    osv_keys = osv_schema["properties"]

    osv_parsed = dict()
    affected_base = pd.DataFrame()

    # Parse the data for a specific manner that will load in DFs better
    for key in osv_keys:
        if osv_keys[key]["type"] == "string":
            if key in osv_json:
                osv_parsed[key] = osv_json[key]
            else:
                osv_parsed[key] = None
        elif osv_keys[key]["type"] == "array":
            if osv_keys[key]["items"]["type"] == "string":
                if key in osv_json:
                    osv_parsed[key] = [item for item in osv_json[key]]
                else:
                    osv_parsed[key] = []
            elif osv_keys[key]["items"]["type"] == "object":
                if key == "references":
                    if key in osv_json:
                        osv_parsed["reference_type"] = [
                            ref["type"] for ref in osv_json[key]
                        ]
                        osv_parsed["reference_url"] = [
                            ref["url"] if "url" in ref else None
                            for ref in osv_json[key]
                        ]
                        osv_parsed["reference_combined"] = [
                            [ref["type"], ref["url"]] if "url" in ref else None
                            for ref in osv_json[key]
                        ]

                    else:
                        osv_parsed["reference_type"] = []
                        osv_parsed["reference_url"] = []
                        osv_parsed["reference_combined"] = []

                if key == "affected":
                    if key in osv_json:
                        osv_parsed["ecosystem"] = osv_json[key][0]["package"][
                            "ecosystem"
                        ]
                        osv_parsed["package_name"] = osv_json[key][0]["package"]["name"]

                        try:
                            # affected complete
                            affected_base = pd.json_normalize(
                                osv_json, record_path=["affected"]
                            )
                            affected_base = pd.json_normalize(osv_json[key])

                            # affected ranges (versions)
                            affected_ranges = pd.json_normalize(
                                osv_json[key], record_path=["ranges"]
                            )

                            affected_ranges["introduced"] = affected_ranges.apply(
                                lambda x: x["events"][0]["introduced"], axis=1
                            )

                            affected_ranges["fixed"] = affected_ranges.apply(
                                lambda x: x["events"][1]["fixed"]
                                if "fixed" in str(x["events"])
                                else None,
                                axis=1,
                            )

                            affected_ranges["limit"] = affected_ranges.apply(
                                lambda x: x["events"][1]["limit"]
                                if "limit" in str(x["events"])
                                else None,
                                axis=1,
                            )

                            affected_base = pd.merge(
                                affected_base,
                                affected_ranges,
                                right_index=True,
                                left_index=True,
                                how="inner",
                            )

                            affected_base = affected_base.drop(
                                columns=["ranges", "events"]
                            )

                            affected_base["id"] = osv_parsed["id"]
                        except:
                            # Issues will be handled downstream
                            affected_base["id"] = osv_parsed["id"]

                    else:
                        osv_parsed["ecosystem"] = []
                        osv_parsed["package_name"] = []
                        osv_parsed["package_purl"] = []
        elif osv_keys[key]["type"] == "object":
            if key == "database_specific":
                if key in osv_json:
                    osv_parsed["cwe_ids"] = osv_json[key]["cwe_ids"]
                    osv_parsed["severity"] = osv_json[key]["severity"]
                    # TODO: Handle all CWEs instead of just the first one listed
                    try:
                        affected_base["cwe_ids"] = osv_json[key]["cwe_ids"][0]
                    except:
                        affected_base["cwe_ids"] = None
                else:
                    osv_parsed["cwe_ids"] = None
                    osv_parsed["severity"] = None
                    affected_base["cwe_ids"] = None

    return osv_parsed, affected_base


def parse_govulndb_symbols(file_path: str) -> list:
    """_summary_

    Args:
        file_path (str): Full path to the govulndb YAML file

    Returns:
        list: Single list of symbols and derived symbols for a YAML file
    """
    temp_govuln_symbols_list = []
    temp_complete = []

    with open(file_path) as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        temp_id = yaml_data["id"]

        derived_symbols = []
        symbols = []

        if "modules" in yaml_data:
            for module in yaml_data["modules"]:
                module_name = module['module']
                package_name = None
                if "packages" in module:
                    for package in module["packages"]:
                        symbols = []
                        derived_symbols = []
                        package_name = package['package']
                        if "symbols" in package:
                            symbols = package['symbols']
                        if "derived_symbols" in package:
                            derived_symbols = package["derived_symbols"]

                        temp_symbols = [
                            [temp_id, module_name, package_name, x_symbol, 'symbol'] for x_symbol in symbols]
                        temp_derived = [[temp_id, module_name, package_name,
                                         x_derived, 'derived'] for x_derived in derived_symbols]
                        temp_complete = temp_complete + temp_symbols + temp_derived

        if len(temp_complete) > 0:
            temp_govuln_symbols_list = temp_govuln_symbols_list + temp_complete
        else:
            temp_govuln_symbols_list = temp_govuln_symbols_list + \
                [[temp_id, None, None, None, None]]

    return temp_govuln_symbols_list


def parse_govulndb_cwe(file_path: str) -> list:
    """_summary_

    Args:
        file_path (str): _description_

    Returns:
        list: _description_
    """
    # Open the json
    with open(file_path, "r") as f:
        osv_cve = json.load(f)

    id = file_path.split('/')[-1].replace('.json', '')
    cwe = None
    cwe_desc = None

    try:
        cwe = osv_cve['containers']['cna']['problemTypes'][0]['descriptions'][0]['description'].split(':')[
            0].strip()
    except:
        pass

    try:
        cwe_desc = osv_cve['containers']['cna']['problemTypes'][0]['descriptions'][0]['description'].split(
            ':')[-1].strip()
    except:
        pass

    return [id, cwe, cwe_desc], ['id', 'cwe', 'cwe_desc']
