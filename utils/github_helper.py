import requests
import git
import re
import os
import datetime
from dateutil import parser
import pandas as pd
import subprocess
from bs4 import BeautifulSoup
from dateutil import parser
import time
from decouple import config


GITHUB_TOKEN = config('GITHUB_TOKEN')
GITHUB_USERNAME = config('GITHUB_USERNAME')


def test_token():
    print(GITHUB_TOKEN)


def get_soup(url):
    response = requests.get(url=url)  # request response from url
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup


def check_yaml_status(repo):
    headers = {'Authorization': 'token %s' % GITHUB_TOKEN}
    url = "https://api.github.com/search/code?q=yaml.load+in%3afile+language%3apy+repo%3a" + repo
    response = requests.get(url, headers=headers)
    response.close()
    return response.json()


def github_rate_limit():
    headers = {'Authorization': 'token %s' % GITHUB_TOKEN}
    url = "https://api.github.com/rate_limit"
    response = requests.get(url, headers=headers)
    response.close()
    return response.json()


def smart_limit(verbose=False):
    """
    Handles the GitHub rate limit issues
    """
    rate = github_rate_limit()
    rate_limit_remaining = rate['rate']['remaining']
    reset = datetime.datetime.fromtimestamp(rate["rate"]["reset"])
    if verbose:
        print(f"Rate Limit Remaining: {rate_limit_remaining} | "
              f"Reset: {reset} | "
              f"Current time: {datetime.datetime.now()}")

    """Handles rate limit issues"""
    if rate_limit_remaining <= 50:
        """Get seconds until reset occurs"""
        time_until_reset = reset - datetime.datetime.now()
        print(f"Seconds until reset: {time_until_reset.seconds}")
        print(f"Starting sleep countdown now: {datetime.datetime.now()}")
        """Sleep until rate limit reset...add 30 seconds to be safe"""
        for i in reversed(range(0, time_until_reset.seconds, 60)):
            print(f"Sleep state remaining: {i} seconds.")
            time.sleep(60)


def github_clone(repo_owner, repo_name, clone_location=None):
    """
    Clones a given repository from Github to a set path on local machine
    :param repo_owner: Owner of GitHub repo
    :param repo_name: Name of GitHub repo
    """
    # set path
    working_directory = os.getcwd()
    if clone_location is None:
        clone_path = f"{working_directory}/github/repos/{repo_owner}/"
    else:
        clone_path = f"{clone_location}/repos/{repo_owner}/"

    if not os.path.exists(clone_path):
        os.makedirs(clone_path)

    # check if clone already exists
    if os.path.exists(f"{clone_path}{repo_name}"):
        print(f"Path already exists: {clone_path}{repo_name}")
    else:
        # clone repo
        git.Git(clone_path).clone(f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{repo_owner}/"
                                  f"{repo_name.replace('.git','')}.git")


def github_clone_specific_path(clone_path, repo_owner, repo_name, local_name=False):
    """
    Clones a given repository from Github to a set path on local machine
    :param clone_path: Path you want the repo to clone to
    :param repo_owner: Owner of GitHub repo
    :param repo_name: Name of GitHub repo
    :param local_name: Local folder name for cloned repository
    """
    # set path
    if not local_name:
        clone_path = f"{clone_path}{repo_owner}/"
    else:
        clone_path = f"{clone_path}"

    if not os.path.exists(clone_path):
        os.makedirs(clone_path)

    if not local_name:
        # check if clone already exists
        if os.path.exists(f"{clone_path}{repo_name}"):
            print(f"Path already exists: {clone_path}{repo_name}")
        else:
            # clone repo
            git.Git(clone_path).clone(f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{repo_owner}/"
                                      f"{repo_name.replace('.git', '')}.git")
    else:  # specific local folder name
        # check if clone already exists
        if os.path.exists(f"{clone_path}{local_name}"):
            print(f"Path already exists: {clone_path}{local_name}")
        else:
            # clone repo
            git.Git(clone_path).clone(f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{repo_owner}/"
                                      f"{repo_name.replace('.git', '')}.git")


def github_commit_diff(repo_owner, repo_name, commit_sha):
    """
    :param repo_owner: Owner of GitHub repo
    :param repo_name: Name of GitHub repo, saved folder in the repos path
    :param commit_sha: Target commit from a patch link
    :return: commit_info:
    """
    # set path
    working_directory = os.getcwd()
    clone_path = f"{working_directory}/github/repos/"

    repo = git.Repo(f"{clone_path}{repo_owner}/{repo_name}")

    # describe the commit sha to get the target version
    # this should be the broken version = target commit
    try:
        broken_version = repo.git.describe(f"{commit_sha}")
    except:
        try:
            broken_version = repo.git.describe(f"--tags {commit_sha}")
        except:
            broken_version = None

    # the target commit is the commit from the CVE
    target_commit_sha = commit_sha

    # this would be the the files previous of the target commit
    # ^ means go back 1
    # http://schacon.github.io/git/git-rev-parse.html#_specifying_revisions
    parent_commit_sha = target_commit_sha + "^"

    # https://github.com/gitpython-developers/GitPython/issues/30
    # retrieve the git diff hunks : http://www.gnu.org/software/diffutils/manual/html_node/Hunks.html
    # see what was changed between commit
    version_a = repo.commit(parent_commit_sha)
    version_b = repo.commit(target_commit_sha)

    # get commit timestamp
    commit_date = datetime.datetime.fromtimestamp(version_b.committed_date)

    # fill this with file name changes, and diff line hunks
    commit_info = []
    for d in version_a.diff(version_b, create_patch=True):

        # only interested in python files
        try:
            # if ".py" in d.a_path:
            changed_file = d.a_path

            if changed_file is None:  # take the newly added file
                changed_file = d.b_path

            # convert diff bytes to string
            diff_string = d.diff.decode("utf-8")
            # search for code hunk line numbers between @@
            diff_hunk_lines = re.findall('@@(.*?)@@', diff_string)
            # temp list
            temp_hunk = []
            line_range = []
            # print line numbers of diff hunk
            for line in diff_hunk_lines:
                print(line)
                temp_hunk.append(line)
                split_hunk = line.strip().split(
                    "+")[0].replace("-", "").split(",")
                split_hunk[0] = int(split_hunk[0])
                split_hunk[1] = split_hunk[0] + int(split_hunk[1])
                line_range.append(split_hunk)
            commit_info.append([repo_owner, repo_name, commit_sha, changed_file, temp_hunk,
                                line_range, broken_version, commit_date])
        except:
            pass

    # create a clean dataframe
    commit_info = pd.DataFrame(commit_info, columns=["repo_owner", "repo_name", "commit_sha",
                                                     "changed_file", "diff_hunk_full",
                                                     "vuln_lines", "broken_ver", "commit_date"])
    return commit_info


def github_commit_diff_specific_path(clone_path, repo_owner, repo_name, commit_sha):
    """
    :param repo_owner: Owner of GitHub repo
    :param repo_name: Name of GitHub repo, saved folder in the repos path
    :param commit_sha: Target commit from a patch link
    :return: commit_info:
    """
    # set path
    repo = git.Repo(f"{clone_path}{repo_owner}/{repo_name}")

    # describe the commit sha to get the target version
    # this should be the broken version = target commit
    try:
        broken_version = repo.git.describe(f"{commit_sha}")
    except:
        try:
            broken_version = repo.git.describe(f"--tags {commit_sha}")
        except:
            broken_version = None

    # the target commit is the commit from the CVE
    target_commit_sha = commit_sha

    # this would be the the files previous of the target commit
    # ^ means go back 1
    # http://schacon.github.io/git/git-rev-parse.html#_specifying_revisions
    parent_commit_sha = target_commit_sha + "^"

    # https://github.com/gitpython-developers/GitPython/issues/30
    # retrieve the git diff hunks : http://www.gnu.org/software/diffutils/manual/html_node/Hunks.html
    # see what was changed between commit
    version_a = repo.commit(parent_commit_sha)
    version_b = repo.commit(target_commit_sha)

    # get commit timestamp
    commit_date = datetime.datetime.fromtimestamp(version_b.committed_date)

    # fill this with file name changes, and diff line hunks
    commit_info = []
    for d in version_a.diff(version_b, create_patch=True):

        # only interested in python files
        try:
            # if ".py" in d.a_path:
            changed_file = d.a_path

            if changed_file is None:  # take the newly added file
                changed_file = d.b_path

            # convert diff bytes to string
            diff_string = d.diff.decode("utf-8")
            # search for code hunk line numbers between @@
            diff_hunk_lines = re.findall('@@(.*?)@@', diff_string)
            # temp list
            temp_hunk = []
            line_range = []
            # print line numbers of diff hunk
            for line in diff_hunk_lines:
                # print(line)
                temp_hunk.append(line)
                split_hunk = line.strip().split(
                    "+")[0].replace("-", "").split(",")
                split_hunk[0] = int(split_hunk[0])
                split_hunk[1] = split_hunk[0] + int(split_hunk[1])
                line_range.append(split_hunk)
            commit_info.append([repo_owner, repo_name, commit_sha, changed_file, temp_hunk,
                                line_range, broken_version, commit_date])
        except:
            pass

    # create a clean dataframe
    commit_info = pd.DataFrame(commit_info, columns=["repo_owner", "repo_name", "commit_sha",
                                                     "changed_file", "diff_hunk_full",
                                                     "vuln_lines", "broken_ver", "commit_date"])
    return commit_info


class GetPackageProjectLinksPyPI:
    """
    Obtain PyPI links for a package on PyPI
    Home Page
    Bug Tracker
    Documentation
    Source Code
    GitHub Repo
    """

    def __init__(self, package):
        """
        :param package: Package name of PyPI package
        """
        self.package_name = package
        self.home_page = None
        self.bug_tracker = None
        self.documentation = None
        self.source_code = None
        self.github = None
        self.repo_full = None
        self.url = f"https://pypi.org/project/{self.package_name}/"

        soup = get_soup(self.url)
        table = soup.find_all("div", attrs={"class": "sidebar-section"})

        for each in table:
            for row in each.find_all("a"):
                if row.text.strip() == "Homepage":
                    self.home_page = row.get("href")
                    if "github" in self.home_page:
                        self.github = self.home_page
                if row.text.strip() == "Bug Tracker":
                    self.bug_tracker = row.get("href")
                    if "github" in self.bug_tracker:
                        self.github = self.bug_tracker
                if row.text.strip() == "Documentation":
                    self.documentation = row.get("href")
                    if "github" in self.documentation:
                        self.github = self.documentation
                if row.text.strip() == "Source Code":
                    self.source_code = row.get("href")
                    if "github" in self.source_code:
                        self.github = self.source_code

        """ Clean up the GitHub link to only have the Owner/Name in the link"""
        if self.github is not None:
            try:
                clean_github = "/".join(self.github.split(".com/")
                                        [1].split("/")[0:2])
                self.github = f"https://github.com/{clean_github}"
                self.repo_full = clean_github
            except:
                pass


def git_checkout_parent(repo_owner, repo_name, commit_sha, clone_path=None):
    """
    Checkouts the parent commit of a target commit (vulnerable commit from patch link)
    :param repo_owner: Owner of GitHub repo
    :param repo_name: Name of GitHub repo, saved folder in the repos path
    :param commit_sha: Target commit from a patch link
    :param clone_path: Custom directory
    :return: None
    """
    if clone_path == None:
        """Set paths"""
        working_directory = os.getcwd()
        clone_path = f"{working_directory}/github/repos/"

    """Set parent commit"""
    parent_commit_sha = f"{commit_sha}^"

    """Generate commands"""
    if clone_path == None:
        change_directory = f"cd {clone_path}{repo_owner}/{repo_name}"
    else:
        change_directory = f"cd {clone_path}"
    checkout_command = f"git checkout {parent_commit_sha}"

    """
    Changes to the directory and runs a git checkout of parent commit
    Input is trusted, and fails when shell=False
    """
    subprocess.run(f"{change_directory}; {checkout_command}", shell=True)


def git_checkout_target(repo_owner, repo_name, commit_sha, clone_path=None):
    """
    Checkouts the target commit (non-vulnerable commit from patch link)
    :param repo_owner: Owner of GitHub repo
    :param repo_name: Name of GitHub repo, saved folder in the repos path
    :param commit_sha: Target commit from a patch link
    :return: None
    """
    if clone_path == None:
        """Set paths"""
        working_directory = os.getcwd()
        clone_path = f"{working_directory}/github/repos/"

    """Generate commands"""
    if clone_path == None:
        change_directory = f"cd {clone_path}{repo_owner}/{repo_name}"
    else:
        change_directory = f"cd {clone_path}"
    checkout_command = f"git checkout {commit_sha}"

    """
    Changes to the directory and runs a git checkout of parent commit
    Input is trusted, and fails when shell=False
    """
    subprocess.run(f"{change_directory}; {checkout_command}", shell=True)


def github_get_commit_date(clone_path, repo_owner, repo_name, commit_sha):
    """
    :param repo_owner: Owner of GitHub repo
    :param repo_name: Name of GitHub repo, saved folder in the repos path
    :param commit_sha: Target commit from a patch link
    :return: commit_info:
    """
    # set path
    repo = git.Repo(f"{clone_path}{repo_owner}/{repo_name}")

    # the target commit is the commit from the CVE
    target_commit_sha = commit_sha

    # this would be the the files previous of the target commit
    # ^ means go back 1
    # http://schacon.github.io/git/git-rev-parse.html#_specifying_revisions
    parent_commit_sha = target_commit_sha + "^"

    # https://github.com/gitpython-developers/GitPython/issues/30
    # retrieve the git diff hunks : http://www.gnu.org/software/diffutils/manual/html_node/Hunks.html
    # see what was changed between commit
    target_commit = repo.commit(target_commit_sha)

    # get commit timestamp
    commit_date = datetime.datetime.fromtimestamp(target_commit.committed_date)

    return commit_date


def github_get_commit_date_local(repo_path: str, commit_sha: str):
    """Obtain the datetime for a given commit SHA

    Args:
        repo_path (str): Path to locally cloned repo
        commit_sha (str): Target commit sha

    Returns:
        datetime: Date of target commit sha
    """
    # set path
    repo = git.Repo(f"{repo_path}")

    # the target commit is the commit from the CVE
    target_commit_sha = commit_sha

    # this would be the the files previous of the target commit
    # ^ means go back 1
    # http://schacon.github.io/git/git-rev-parse.html#_specifying_revisions
    parent_commit_sha = target_commit_sha + "^"

    # https://github.com/gitpython-developers/GitPython/issues/30
    # retrieve the git diff hunks : http://www.gnu.org/software/diffutils/manual/html_node/Hunks.html
    # see what was changed between commit
    target_commit = repo.commit(target_commit_sha)

    # get commit timestamp
    commit_date = datetime.datetime.fromtimestamp(target_commit.committed_date)

    return commit_date


def get_package_branches(repo_owner, repo_name):
    """
    Gets the project commit history for a given repository
    :param repo_owner: Owner of GitHub repo
    :param repo_name: Name of GitHub repo, saved folder in the repos path
    :return: List of branches
    """

    url = f"https://api.github.com/repos/{repo_name}/{repo_owner}/branches"
    response = requests.get(url)
    response.close()

    return response.json()


def get_package_commit_history(repo_owner, repo_name, per_page=10, target_sha=None, until=None):
    """
    Gets the project commit history for a given repository
    Info: https://docs.github.com/en/rest/reference/commits#get-a-commit
    :param repo_owner: Owner of GitHub repo
    :param repo_name: Name of GitHub repo, saved folder in the repos path
    :param per_page: Limit the amount of commits returned (default 10, max is 100 per page)
    :param target_sha: Target commit to get previous history
    :param until: Only commits before this date will be returned. ISO 8601 date format only.
    :return: A clean DF of the response for columns I find useful
    """

    """Set Auth params"""
    headers = {'Authorization': 'token %s' % GITHUB_TOKEN}

    if target_sha is None and until is None:
        """If no target commit is provided start at latest commit"""
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits?" \
              f"per_page={per_page}"
    elif until is not None:
        """If no target commit is provided start at latest commit"""
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits?" \
              f"per_page={per_page}&" \
              f"until={until}"
    else:
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits?" \
              f"per_page={per_page}&" \
              f"sha={target_sha}"

    response = requests.get(url, headers=headers)
    response.close()

    if response.status_code == 200:
        """Create an non-error based response"""
        """Convert the response to a clean DF"""
        temp_commits = []
        for index, x in enumerate(response.json()):
            temp_dict = dict(repo_owner=repo_owner,
                             repo_name=repo_name,
                             sha=x["sha"],
                             node_id=x["node_id"],
                             commit_author_name=x["commit"]["author"]["name"],
                             commit_author_email=x["commit"]["author"]["email"],
                             commit_author_date=x["commit"]["author"]["date"],
                             commit_committer_date=x["commit"]["committer"]["date"],
                             commit_message=x["commit"]["message"],
                             commit_tree_sha=x["commit"]["tree"]["sha"],
                             commit_tree_url=x["commit"]["tree"]["url"],
                             commit_verification_verified=x["commit"]["verification"]["verified"],
                             commit_verification_reason=x["commit"]["verification"]["reason"],
                             url=x["url"],
                             html_url=x["html_url"],
                             parents=[z["sha"] for z in x["parents"]],
                             response_code=response.status_code)
            temp_commits.append(temp_dict)

        temp_commits = pd.DataFrame(temp_commits)

        """Create a custom commit_date to use in filenames"""
        temp_commits["commit_date"] = temp_commits.apply(lambda row: parser.parse(row['commit_author_date']).strftime("%Y-%m-%d-%H-%M-%S"),
                                                         axis=1)
        """Provide an error status"""
        error = False

        return temp_commits, error
    else:
        """Create an error based response"""
        temp_dict = dict(repo_owner=repo_owner,
                         repo_name=repo_name,
                         sha=None,
                         node_id=None,
                         commit_author_name=None,
                         commit_author_email=None,
                         commit_author_date=None,
                         commit_committer_date=None,
                         commit_message=None,
                         commit_tree_sha=None,
                         commit_tree_url=None,
                         commit_verification_verified=None,
                         commit_verification_reason=None,
                         url=None,
                         html_url=None,
                         parents=[],
                         commit_date=None,
                         response_code=response.status_code)

        temp_commits = pd.DataFrame([temp_dict])

        print(response.json())

        """Provide an error status"""
        error = True

        return temp_commits, error


def get_repo_size(repo_owner, repo_name):
    """
    Gets the project commit history for a given repository
    :param repo_owner: Owner of GitHub repo
    :param repo_name: Name of GitHub repo, saved folder in the repos path
    :return: List of branches
    """

    url = f"https://api.github.com/repos/{repo_name}/{repo_owner}"
    response = requests.get(url)
    response.close()

    return response.json()


def get_api_commit_date(repo_owner, repo_name, commit_sha):
    """
    Gets the project commit date for a given commit_sha
    :param repo_owner: Owner of GitHub repo
    :param repo_name: Name of GitHub repo, saved folder in the repos path
    :param commit_sha: Target commit for the date
    :return: List of branches
    """

    """Set Auth params"""
    headers = {'Authorization': 'token %s' % GITHUB_TOKEN}

    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits?sha={commit_sha}&per_page=1"
    response = requests.get(url, headers=headers)
    response.close()

    date = response.json()[0]["commit"]["author"]["date"]

    return date


def get_child_commit(repo_owner, repo_name, commit_sha):
    """
    Gets the commit after a given target commit
    :param repo_owner: Owner of GitHub repo
    :param repo_name: Name of GitHub repo, saved folder in the repos path
    :param commit_sha: Target commit
    :return: List of branches
    """

    """Set Auth params"""
    headers = {'Authorization': 'token %s' % GITHUB_TOKEN}

    print(repo_owner, repo_name, commit_sha)

    """Create a date time that's 10 days past the patch link ~ hope the child commit is in the 10 day period"""
    commit_date = get_api_commit_date(repo_owner, repo_name, commit_sha)
    commit_date = parser.parse(commit_date)
    commit_date_10_day = commit_date + datetime.timedelta(days=15)
    commit_date_10_day = commit_date_10_day.strftime('%Y-%m-%dT%H:%M:%S.%f%z')

    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits?until={commit_date_10_day}&per_page=100"
    response = requests.get(url, headers=headers)
    response.close()

    """Find the child commit from list"""
    child_commit = None
    try:
        for each in response.json():
            temp_child_commit = each["sha"]
            temp_parent_commit = each["parents"][0]["sha"]

            """Confirm parent commit of child is the target commit_sha"""
            if temp_parent_commit == commit_sha:
                child_commit = temp_child_commit
                print(f"Match: {temp_child_commit} | {temp_parent_commit}")
    except:
        pass

    return child_commit


def get_github_commit_info(repo_owner, repo_name, commit_sha):
    """Hits the commit API for a specific commit_sha
    Info: https://docs.github.com/en/rest/search

    Args:
        repo_owner (str): Target repo owner
        repo_name (str): Target repo name
        commit_sha (str): Target commit SHA from GitHub

    Returns:
        json: Full commit info so end-user can do whatever they desire
    """
    # personal token
    headers = {'Authorization': 'token %s' % GITHUB_TOKEN}

    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits/{commit_sha}"

    response = requests.get(url, headers=headers)
    response.close()

    return response.json()


def get_pr_from_commit(repo_owner, repo_name, commit_sha):
    """Tries to find the associated PR from a commit_sha
    Note: this uses the GitHub search API which has a slower request rate
    Info: https://docs.github.com/en/rest/search

    Args:
        repo_owner (str): Target repo owner
        repo_name (str): Target repo name
        commit_sha (str): Target commit SHA from GitHub

    Returns:
        list: List of dictionaries
    """
    # personal token
    headers = {'Authorization': 'token %s' % GITHUB_TOKEN}

    url = f"https://api.github.com/search/issues?q={commit_sha}"

    response = requests.get(url, headers=headers)
    response.close()

    """Parse the issues search and only return certain fields"""
    commit_pr = []
    for pr in response.json()["items"]:
        """Get the timeline for each issue"""
        timeline = get_github_issues_timeline(
            repo_owner, repo_name, pr["number"])
        for event in timeline:
            """Check the timeline has the target commit associated"""
            if event["event"] == "committed" and event["sha"] == commit_sha:
                pr_temp = dict(
                    url=pr["url"],
                    issue_number=pr["number"],
                    title=pr["title"],
                    created_at=pr["created_at"],
                    state=pr["state"],
                    body=pr["body"],
                    timeline=timeline
                )
                if pr_temp not in commit_pr:
                    commit_pr.append(pr_temp)

    return commit_pr


def get_github_issues_timeline(repo_owner: str, repo_name: str, issue_number) -> dict:
    """Hits the Issues Timeline API for detail on the issue
    Info: https://docs.github.com/en/rest/issues/timeline

    Args:
        repo_owner (str): Target repo owenr
        repo_name (str): Target repo name
        issue_number (int): The number that identifies the issue

    Returns:
        json: Full issues timeline info so end-user can do whatever they desire
    """
    # personal token
    headers = {'Authorization': 'token %s' % GITHUB_TOKEN}

    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}/timeline"

    response = requests.get(url, headers=headers)
    response.close()

    return response.json()


def get_github_tags(repo_owner: str, repo_name: str) -> list:
    """Hits the Tags  API for a list of tags available

    Args:
        repo_owner (str): Target repo owenr
        repo_name (str): Target repo name

    Returns:
        list: Full issues timeline info so end-user can do whatever they desire
    """
    # personal token
    headers = {'Authorization': 'token %s' % GITHUB_TOKEN}

    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/tags?per_page=100"

    response = requests.get(url, headers=headers)
    response.close()

    """Get a clean list of all the tags"""
    tags = []
    for tag in response.json():
        temp_tag = dict(name=tag["name"],
                        commit_sha=tag["commit"]["sha"],
                        url=tag["commit"]["url"])
        """Get the commit date, always compare to committer date as that's when it was merged"""
        tag_date = get_github_commit_info(repo_owner, repo_name, temp_tag["commit_sha"])[
            "commit"]["author"]["date"]
        temp_tag["date"] = tag_date
        tags.append(temp_tag)

    return tags


def get_github_tag_for_commit(repo_owner: str, repo_name: str, commit_sha: str, commit_committer_date: datetime = None) -> dict:
    """Pulls the associated tag for a given commit_sha
    THIS ONLY WORKS ON THE MAIN BRANCH
    Args:
        repo_owner (str): Target repo owenr
        repo_name (str): Target repo name
        commit_sha (str): Target commit sha
        commit_committer_date (datetime): Committer commit date of a target commit_sha

    Returns:
        dict: Series of the commit tag info
    """
    # personal token
    headers = {'Authorization': 'token %s' % GITHUB_TOKEN}

    if commit_committer_date is not None:
        commit_committer_date = parser.parse(commit_committer_date)
    else:
        """Obtain commit_committer_date"""
        commit_committer_date = get_github_commit_info(repo_owner, repo_name, commit_sha)[
            "commit"]["committer"]["date"]
        commit_committer_date = parser.parse(commit_committer_date)

    """Get a clean list of all the tags"""
    tags = get_github_tags(repo_owner, repo_name)

    """Convert to a DF and to datetime"""
    tags = pd.DataFrame(tags)

    """
    Sort by date and take the first row for the 
    associated tag, becuase the commit date should be prior to the next tag
    """
    try:
        tags['date'] = pd.to_datetime(tags['date'])
        commit_tag = tags[tags["date"] >= commit_committer_date].sort_values(
            "date", ascending=True).iloc[0]
    except Exception as e:
        print(f"Missing tags: {str(e)}")
        commit_tag = pd.DataFrame(columns=tags.columns)

    return commit_tag.to_dict()


def get_github_compare_tags(repo_owner, repo_name, previous_tag, current_tag):
    """Gets the list of commits between to commits
    Info: https://docs.github.com/en/rest/commits/commits#compare-two-commits

    Args:
        repo_owner (str): Target repo owner
        repo_name (str): Target repo name
        previous_tag (str): Start tag for comparison
        current_tag (str): End tag for comparison

    Returns:
        json: Full JSON repsone so end-user can do whatever they desire
    """
    # personal token
    headers = {'Authorization': 'token %s' % GITHUB_TOKEN}

    # the three dots (...) represent between tags
    # e.g., https://github.com/tensorflow/tensorflow/compare/v2.11.0-rc1...v2.11.0-rc2
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/compare/{previous_tag}...{current_tag}"

    response = requests.get(url, headers=headers)
    response.close()

    return response.json(), response.status_code


def get_latest_commit(repo_path: str) -> str:
    """Gets latest commit SHA for a locally cloned repository
    Info: git log --format="%H" -n 1

    Args:
        repo_path (str): Path to locally cloned repository

    Returns:
        str: The latest commit SHA
    """
    # set path
    repo = git.Git(f"{repo_path}")

    return repo.log('--format="%H"', '-n 1').strip("'").strip('"')


def git_pull(repo_path: str):
    """Update the local repository
    Info: git pull

    Args:
        repo_path (str): Path to locally cloned repository
    """
    # set path
    repo = git.Git(f"{repo_path}")

    repo.pull()


def get_tags(repo_owner, repo_name, clone_path):
    """Obtains the local git repo tags for a given repository in a certain path

    Args:
        repo_owner (str): Repo owner
        repo_name (str): Name of repo
        clone_path (str): Local clone path of repo

    Returns:
        pd.DataFrame: A sorted pandas df of tags
    """

    # create repo path
    repo_path = f"{clone_path}"

    # execute the git tags command
    git_tags_command = (
        f"(cd {repo_path} && "
        f"git for-each-ref --sort=v:refname --format '%(refname) %(creatordate)' refs/tags)"
    )

    # this is all trusted input....not a vulnerability
    git_tags = subprocess.check_output(
        git_tags_command, shell=True, encoding="UTF-8"
    ).splitlines()

    # load in the tag outputs
    if len(git_tags) > 0:
        temp_df = pd.DataFrame(git_tags, columns=["raw_out"])
        temp_df["repo_owner"] = repo_owner
        temp_df["repo_name"] = repo_name
        temp_df["tag_count"] = len(temp_df)

        # extract the creatordate
        temp_df["creatordate"] = temp_df.apply(
            lambda x: datetime.datetime.strptime(
                " ".join(x["raw_out"].strip("\n").split(" ")[1:-1]),
                "%a %b %d %H:%M:%S %Y",
            ),
            axis=1,
        )
        # extract the tag from the list
        temp_df["tag"] = temp_df.apply(
            lambda x: x["raw_out"].strip("\n").split(
                " ")[0].replace("refs/tags/", ""),
            axis=1,
        )

        # get the correct semver tag order
        # temp_tags = temp_df["tag"].values.tolist()

        # sort the tags
        # sorted_tags = semver_sort(temp_tags)

        # add the sorted tags back to the original df
        # temp_df_sorted = pd.merge(temp_df, sorted_tags, on="tag", how="left")

    else:
        temp_df_sorted = pd.DataFrame(
            [["NO_TAGS", repo_owner, repo_name]],
            columns=["raw_out", "repo_owner", "repo_name"],
        )
        temp_df_sorted["tag_count"] = None
        temp_df_sorted["creatordate"] = None
        temp_df_sorted["tag"] = None
        temp_df_sorted["tag_order"] = None

    return temp_df


def semver_sort(temp_versions):
    """Sorts semver tags based on pythons packaging.version

    Args:
        temp_versions (list): List of tags

    Returns:
        pd.DataFrame: Sorted tags based on semver
    """
    if temp_versions is not None:
        if len(temp_versions) > 0:
            clean_parse = []
            for each in temp_versions:
                try:
                    temp_version = Version(each)
                    temp_version.raw_version = each
                    temp_version.error = False
                    clean_parse.append(temp_version)
                except Exception as err:
                    print(err)
                    # TODO: this needs to be handled better
                    clean_each = ".".join(each.split(".")[:3])
                    temp_version = Version(clean_each)
                    temp_version.raw_version = each
                    temp_version.error = True
                    clean_parse.append(temp_version)

            # sort the clean versions
            clean_parse.sort()

            clean_return = []

            for clean in clean_parse:
                clean_return.append(clean.raw_version)

            # create a df to sort the versions
            clean_return_df = pd.DataFrame(clean_return, columns=["tag"])
            clean_return_df["tag_order"] = clean_return_df.index

            return clean_return_df
    else:
        return []


def git_checkout_target_new(repo_path: str, sha: str, parent=False):
    """Uses GitPython to checkout a given commit

    Args:
        repo_path (str): Full repo path
        sha (str): Target commit to checkout
        parent (bool, optional): Checkout parent. Defaults to False.
    """

    # set path
    repo = git.Git(f"{repo_path}")

    if parent:
        # the ^ represents the parent commit
        repo.checkout(f"{sha}^")
    else:
        repo.checkout(sha)
