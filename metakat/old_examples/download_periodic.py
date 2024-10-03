# File: download_periodic.py
# Author: Jakub Křivánek
# Date: 7. 5. 2024
# Description: This file contains the script for downloading metadata from Kramerius.

import requests
import os
import time
import enlighten
import argparse
from lxml import etree
import numpy as np
from PIL import Image

manager = enlighten.get_manager()


def parse_args():
    parser = argparse.ArgumentParser(description="Download periodical metadata from Kramerius")
    parser.add_argument("--api-url", type=str, required=True, help="URL of the Kramerius API")
    parser.add_argument("--uuid", type=str, help="UUID of the periodical")
    parser.add_argument("--file", type=str, help="File with UUIDs of periodicals")
    parser.add_argument("--doc-model", type=str, help="Filter only lines from --file with this model")
    parser.add_argument("--exclude-collections", nargs="+", help="Exclude periodicals from this collection", default=None)
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--skip", type=int, help="Skip N first lines of --file", default=0)
    parser.add_argument("--pages", action="store_true", help="Download pages metadata", default=False)
    parser.add_argument("--title-pages", action="store_true", help="Download only title pages metadata", default=False)
    parser.add_argument("--unique-periodicals", action="store_true", help="Download metadata only for one title page per periodical", default=False)
    parser.add_argument("--retry-periodicals", action="store_true", help="Retry downloading periodicals metadata", default=False)
    parser.add_argument("--img-output-dir", type=str, help="Output directory for images", default=None)
    parser.add_argument("--max-freq", type=int, help="Minimum frequency of periodicals to download", default=0)
    parser.add_argument("--lang", nargs="+", help="Languages of periodicals to download", default=["cze"])

    return parser.parse_args()


def frequency_to_number(frequency):
    frequency = frequency.lower().replace(";", "")
    mapping = {
        "annual": 365,
        "biennial": 730,
        "bimonthly": 60,
        "biweekly": 14,
        "daily": 1,
        "monthly": 30,
        "quarterly": 90,
        "semimonthly": 15,
        "semiweekly": 3,
        "three times a month": 10,
        "triennial": 1095,
        "unknown": 0,
        "weekly": 7,

        "denně": 1,
        "3x týdně": 2,
        "třikrát týdně": 2,
        "dvakrát týdně": 3,
        "2x týdně": 3,
        "týdeník": 7,
        "týdenník": 7,
        "týdně": 7,
        "1x týdně": 7,
        "jednou týdně": 7,
        "3x měsíčně": 10,
        "1x za 2 týdny": 14,
        "2x měsíčně": 15,
        "dvakrát měsíčně": 15,
        "jednou měsíčně": 30,
        "1x měsíčně": 30,
        "měsíčně": 30,
        "6x ročně": 61,
        "5x ročně": 73,
        "4x ročně": 91,
        "3x ročně": 122,
        "2x ročně": 183,
        "1x ročně": 365,
        "ročně": 365,
        "ročenka": 365,
        "nepravidelně": 0,
        "nepradivelná": 0,
        "neznámá": 0,
        "neurčitá": 0,
        
        "týždně": 7,
        "1x týždenne": 7,
        "2x týždenne": 3,
        "3x týždenne": 2,
        "mesačne": 30,
        "1x mesačne": 30,
        "2x mesačne": 15,
        "3x mesačne": 10,
        "4x mesačne": 7,
        
    }
    if frequency in mapping:
        return mapping[frequency]
    else:
        try_find_f = [k for k in mapping.keys() if k in frequency]
        if len(try_find_f) == 0:
            print(f"Unknown frequency: {frequency}, returning 0")
            return 0
        else:
            return mapping[try_find_f[0]]


def get_response(url, output_dir=None):
    retry_times = 3
    for _ in range(retry_times):
        try:
            response = requests.get(url)
        except requests.exceptions.ConnectTimeout:
            time.sleep(2)
            continue
        if response.status_code == 200 or response.status_code == 404:
            break
        time.sleep(1)
    if response.status_code != 200:
        print(f"Response status code: {response.status_code}, URL: {url}, skipping...")
    if response.status_code in [404, 403] and output_dir is not None:
        structure_file = os.path.join("../..", output_dir, "structure.txt")
        uuid = url.split("uuid:")[1].split("/")[0]
        if not os.path.exists(structure_file):
            with open(structure_file, "w") as f:
                f.write(f"{uuid} {response.status_code} unable to access\n")
        with open(structure_file, "r") as f:
            lines = f.readlines()
        with open(structure_file, "w") as f:
            for line in lines:
                if uuid in line:
                    f.write(f"{uuid} {response.status_code} unable to access\n")
                else:
                    f.write(line)
    return response


def get_lines_uuid(f, document_type, exclude_collection=None):
    model_string = f'"model": "{document_type}"'
    accessibility_string = f'"accessibility": "public"'

    for line in f:
        line = line.replace("'", '"')
        if line.find(model_string) == -1 or line.find(accessibility_string) == -1:
            continue
        if exclude_collection is not None:
            collection_string = line.split('"cdk.collection": ')[1].split("]")[0].split("[")[1].replace('"', "")
            collections = collection_string.split(", ")
            if any(collection in exclude_collection for collection in collections):
                continue
        if line.startswith("uuid"):
            yield line.split(" ")[0].split("uuid:")[1]
        uuid = line.split("root.pid\": \"")[1].split("\"")[0].split("uuid:")[1]
        yield uuid


class Periodical:
    def __init__(self, uuid):
        self.uuid = uuid
        self.language = None
        self.years = {}
        self.frequency = 0

        self.years_count = 0
        self.issues_count = 0
        self.pages_count = 0

    def add_year(self, year_uuid):
        if year_uuid in self.years:
            return
        self.years[year_uuid] = {}
        self.years_count += 1

    def add_issue(self, year_uuid, issue_uuid):
        if issue_uuid in self.years[year_uuid]:
            return
        self.years[year_uuid][issue_uuid] = {}
        self.issues_count += 1

    def add_page(self, year_uuid, issue_uuid, page_uuid):
        self.years[year_uuid][issue_uuid][page_uuid] = []
        self.pages_count += 1

    def get_years(self):
        return [k for k in self.years.keys() if k != "finished"]

    def get_years_count(self):
        return self.years_count

    def get_issues(self, year_uuid):
        return [k for k in self.years[year_uuid].keys() if k != "finished"]

    def get_issues_count(self):
        return self.issues_count

    def get_pages(self, year_uuid, issue_uuid):
        return self.years[year_uuid][issue_uuid]

    def get_pages_count(self):
        return self.pages_count

    def year_finished(self, year_uuid):
        self.years[year_uuid]["finished"] = True

    def issue_finished(self, year_uuid, issue_uuid):
        self.years[year_uuid][issue_uuid]["finished"] = True
        if all([v for k, v in self.years[year_uuid].items() if k != "finished"]):
            self.year_finished(year_uuid)


class KrameriusAPI:
    def __init__(self, base_url, version="v7.0"):
        self.base_url = base_url
        self.version = version

    def download_complete_periodical_metadata(self, uuid, args):
        print("=" * 80)
        print(f"Periodical {uuid}")

        output_dir = os.path.join(args.output_dir, uuid)
        os.makedirs(output_dir, exist_ok=True)

        img_output_dir = args.img_output_dir
        os.makedirs(img_output_dir, exist_ok=True)
        valid_title_pages = os.path.join(args.img_output_dir, "valid_title_pages.txt")
        if not os.path.exists(valid_title_pages):
            with open(valid_title_pages, "w") as f:
                pass
        periodicals_without_valid_title_pages = os.path.join(args.img_output_dir, "periodicals_without_valid_title_pages.txt")
        if not os.path.exists(periodicals_without_valid_title_pages):
            with open(periodicals_without_valid_title_pages, "w") as f:
                pass

        valid_periodicals = []
        with open(valid_title_pages, "r") as f:
            valid_periodicals = [line.split("/")[0] for line in f.read().splitlines()]
        invalid_periodicals = []
        with open(periodicals_without_valid_title_pages, "r") as f:
            invalid_periodicals = f.read().splitlines()
        if uuid in valid_periodicals:
            print(f"Already downloaded")
            return
        if uuid in invalid_periodicals and not args.retry_periodicals:
            print(f"Already tried to download and failed, skipping...")
            return

        unique_periodicals = args.unique_periodicals
        periodical = Periodical(uuid)
        self.download_periodical_metadata(periodical, output_dir)
        print("Metadata for periodical downloaded")

        if args.max_freq != 0 and (periodical.frequency > args.max_freq or periodical.frequency == 0):
            print(f"Frequency of periodical is {periodical.frequency}, not continuing downloading...")
            return

        if periodical.language not in args.lang:
            print(f"Language of periodical is {periodical.language}, not continuing downloading...")
            return

        title_img_downloaded = False
        able_to_continue_downloading = True
        unique_periodicals_download_tries = 0
        max_unique_periodicals_download_tries = 10
        while True:
            self.download_periodical_years_metadata(periodical, output_dir, unique_periodicals=unique_periodicals)
            self.download_periodical_issues_metadata(periodical, output_dir, unique_periodicals=unique_periodicals)
            if args.pages or args.title_pages:
                title_img_downloaded, able_to_continue_downloading = self.download_periodical_pages_metadata(
                    periodical, output_dir, img_output_dir, valid_title_pages, title_pages_only=args.title_pages, unique_periodicals=unique_periodicals)
            if not unique_periodicals or title_img_downloaded:
                break
            if not able_to_continue_downloading:
                print(f"All periodicals issues title pages tried to download but failed, ending search for this periodical...")
                break
            if unique_periodicals_download_tries > max_unique_periodicals_download_tries:
                print(f"Unique periodicals download tries limit ({max_unique_periodicals_download_tries}) reached, ending search for this periodical...")
                break
            unique_periodicals_download_tries += 1

        if unique_periodicals and not title_img_downloaded:
            with open(periodicals_without_valid_title_pages, "a") as f:
                print(f"Valid title page not found")
                f.write(f"{uuid}\n")

        print(f"Metadata for years downloaded ({periodical.get_years_count()} total)")
        print(f"Metadata for issues downloaded ({periodical.get_issues_count()} total)")
        print(f"Metadata for pages downloaded ({periodical.get_pages_count()} total)")

    def download_periodical_metadata(self, periodical, output_dir):
        url = f"{self.base_url}/search/api/client/{self.version}/items/uuid:{periodical.uuid}/metadata/mods"
        response = get_response(url, output_dir)
        if response.status_code != 200:
            return
        with open(f"{output_dir}/uuid:{periodical.uuid}.xml", "wb") as f:
            f.write(response.content)

        frequency = None
        namespaces = {"mods": "http://www.loc.gov/mods/v3"}
        tree = etree.fromstring(response.content)
        for f in tree.iterfind(".//mods:frequency", namespaces):
            if f.get("authority") != "marcfrequency":
                frequency = f
                break
        if frequency is not None and frequency.text is not None:
            periodical.frequency = frequency_to_number(frequency.text)

        language = tree.find(".//mods:languageTerm", namespaces)
        if language is not None:
            periodical.language = language.text

    def download_periodical_years_metadata(self, periodical, output_dir, unique_periodicals=False):
        year_uuids = self.get_children_uuids(periodical.uuid, os.path.join(output_dir, "structure.txt"))
        if len(year_uuids) == 0:
            return
        if unique_periodicals:
            years_downloaded = [y for y in year_uuids if os.path.exists(os.path.join(output_dir, y, f"uuid:{y}.xml"))]
            if any([y for y in years_downloaded if y not in periodical.get_years()]):
                print(f"Getting years already downloaded ({len(years_downloaded)})")
                for year_uuid in years_downloaded:
                    periodical.add_year(year_uuid)
                return
            year_not_yet_added = [y for y in year_uuids if y not in periodical.get_years()]
            if len(year_not_yet_added) == 0:
                return
            for _ in range(5):
                year_uuids = np.random.choice(year_not_yet_added, 1)
                response = requests.get(f"{self.base_url}/search/api/client/{self.version}/items/uuid:{year_uuids[0]}/metadata/dc")
                if response.status_code != 200 or response.content == b"":
                    continue
                dc_tree = etree.fromstring(response.content)
                namespace = {"dc": "http://purl.org/dc/elements/1.1/"}
                year_policy = dc_tree.find(".//dc:rights", namespace)
                if year_policy is not None and "private" not in year_policy.text:
                    break
        for year_uuid in year_uuids:
            periodical.add_year(year_uuid)
            url = f"{self.base_url}/search/api/client/{self.version}/items/uuid:{year_uuid}/metadata/mods"

            year_dir = os.path.join(output_dir, year_uuid)
            os.makedirs(year_dir, exist_ok=True)
            year_path = os.path.join(year_dir, f"uuid:{year_uuid}.xml")
            if os.path.exists(year_path):
                print(f"Year metadata ({year_uuid}) exists, skipping")
                continue
            response = get_response(url, output_dir)
            if response.status_code != 200:
                continue
            with open(year_path, "wb") as f:
                f.write(response.content)

    def download_periodical_issues_metadata(self, periodical, output_dir, unique_periodicals=False):
        year_uuids = periodical.get_years()
        if unique_periodicals:
            unfinished_years = [y for y in year_uuids if "finished" not in periodical.years[y]]
            if len(unfinished_years) == 0:
                return
            year_uuids = np.random.choice(unfinished_years, 1)
        for year_uuid in year_uuids:
            issue_uuids = self.get_children_uuids(year_uuid, os.path.join(output_dir, year_uuid, "structure.txt"))
            if unique_periodicals:
                issues_downloaded = [i for i in issue_uuids if os.path.exists(os.path.join(output_dir, year_uuid, i, f"uuid:{i}.xml"))]
                if any([i for i in issues_downloaded if i not in periodical.get_issues(year_uuid)]):
                    print(f"Getting issues already downloaded ({len(issues_downloaded)})")
                    for issue_uuid in issues_downloaded:
                        periodical.add_issue(year_uuid, issue_uuid)
                    return
                not_yet_added_issue_uuids = [i for i in issue_uuids if i not in periodical.get_issues(year_uuid)]
                if len(not_yet_added_issue_uuids) == 0:
                    periodical.year_finished(year_uuid)
                    continue
                for _ in range(5):
                    issue_uuids = np.random.choice(not_yet_added_issue_uuids, 1)
                    response = requests.get(f"{self.base_url}/search/api/client/{self.version}/items/uuid:{issue_uuids[0]}/metadata/dc")
                    if response.status_code != 200 or response.content == b"":
                        continue
                    dc_tree = etree.fromstring(response.content)
                    namespace = {"dc": "http://purl.org/dc/elements/1.1/"}
                    issue_policy = dc_tree.find(".//dc:rights", namespace)
                    if issue_policy is not None and "private" not in issue_policy.text:
                        break
            for issue_uuid in issue_uuids:
                periodical.add_issue(year_uuid, issue_uuid)
                url = f"{self.base_url}/search/api/client/{self.version}/items/uuid:{issue_uuid}/metadata/mods"

                issue_dir = os.path.join(output_dir, year_uuid, issue_uuid)
                os.makedirs(issue_dir, exist_ok=True)
                issue_path = os.path.join(issue_dir, f"uuid:{issue_uuid}.xml")
                if os.path.exists(issue_path):
                    print(f"Issue metadata ({issue_uuid}) exists, skipping")
                    continue
                response = get_response(url, output_dir)
                if response.status_code != 200:
                    continue
                with open(issue_path, "wb") as f:
                    f.write(response.content)

    def download_periodical_pages_metadata(self, periodical, output_dir, img_output_dir, valid_title_pages, title_pages_only=False, unique_periodicals=False):
        title_img_downloaded_and_valid = False
        able_to_continue_downloading = True
        periodical_years = periodical.get_years()
        if unique_periodicals:
            unfinished_years = [y for y in periodical_years if "finished" not in periodical.years[y]]
            if len(unfinished_years) == 0:
                able_to_continue_downloading = False
                return title_img_downloaded_and_valid, able_to_continue_downloading
            periodical_years = np.random.choice(unfinished_years, 1)
        for year_uuid in periodical_years:
            periodical_issues = periodical.get_issues(year_uuid)
            if unique_periodicals:
                unfinished_periodical_issues = [i for i in periodical_issues if "finished" not in periodical.years[year_uuid][i]]
                if len(unfinished_periodical_issues) == 0:
                    periodical.year_finished(year_uuid)
                    continue
                periodical_issues = np.random.choice(unfinished_periodical_issues, 1)
            for issue_uuid in periodical_issues:
                issue_dir = os.path.join(output_dir, year_uuid, issue_uuid)
                page_uuids = self.get_children_uuids(issue_uuid, os.path.join(issue_dir, "structure.txt"))

                page_dir = os.path.join(output_dir, year_uuid, issue_uuid, "pages")
                os.makedirs(page_dir, exist_ok=True)

                if title_pages_only:
                    title_page_files = [os.path.join(page_dir, f) for f in os.listdir(page_dir) if f.endswith("_title.xml")]
                    if len(title_page_files) > 0:
                        title_page_file = title_page_files[0]
                        if os.path.exists(title_page_file):
                            title_page_uuid = os.path.basename(title_page_file).split("_title.xml")[0].split("uuid:")[1]
                            title_img_downloaded_and_valid = download_page_img(img_output_dir, title_page_uuid, self.base_url)
                            continue

                title_page_uuid = None
                for page_num, page_uuid in enumerate(page_uuids):
                    end_searching = False
                    periodical.add_page(year_uuid, issue_uuid, page_uuid)
                    url = f"{self.base_url}/search/api/client/{self.version}/items/uuid:{page_uuid}/metadata/mods"

                    page_path = os.path.join(page_dir, f"uuid:{page_uuid}.xml")
                    if not os.path.exists(page_path):
                        response = get_response(url, output_dir)
                        if response.status_code != 200:
                            continue

                        max_pages_tries = 4
                        namespaces = {"mods": "http://www.loc.gov/mods/v3"}
                        tree = etree.fromstring(response.content)
                        title_page_element = tree.xpath("//mods:part[@type='titlePage']", namespaces=namespaces)
                        if not title_page_element:
                            title_page_element = tree.xpath("//mods:part[@type='TitlePage']", namespaces=namespaces)
                        if title_page_element:
                            title_page_uuid = page_uuid
                            page_path = os.path.join(page_dir, f"uuid:{page_uuid}_title.xml")
                            if title_pages_only:
                                end_searching = True
                        elif title_pages_only and page_num >= max_pages_tries:
                            end_searching = True
                            print(f"Title page not found within the first {max_pages_tries}")

                        with open(page_path, "wb") as f:
                            f.write(response.content)

                    if title_page_uuid is not None and title_pages_only:
                        title_img_downloaded_and_valid = download_page_img(img_output_dir, title_page_uuid, self.base_url)
                        if title_img_downloaded_and_valid:
                            with open(valid_title_pages, "a") as f:
                                f.write(f"{periodical.uuid}/{year_uuid}/{issue_uuid}/{title_page_uuid}\n")

                            if unique_periodicals:
                                periodical.issue_finished(year_uuid, issue_uuid)
                                return title_img_downloaded_and_valid, able_to_continue_downloading

                    if end_searching:
                        periodical.issue_finished(year_uuid, issue_uuid)
                        break

        return title_img_downloaded_and_valid, able_to_continue_downloading

    def get_children_uuids(self, uuid, structure_file):
        if os.path.exists(structure_file):
            with open(structure_file, "r") as f:
                lines = []
                for line in f:
                    if "unable to access" in line or "not found" in line:
                        continue
                    lines.append(line.strip())

        url = f"{self.base_url}/search/api/client/{self.version}/items/uuid:{uuid}/info/structure"
        response = get_response(url)
        if response.status_code != 200 or response.content == b"":
            print(f"Item {uuid} not found, returning empty list of children...")
            return []
        response = response.json()
        children = response["children"]["own"]
        uuids = [child["pid"].split("uuid:")[1] for child in children]

        with open(structure_file, "w") as f:
            for child in uuids:
                f.write(child + "\n")
        return uuids


def download_page_img(img_output_dir, page_uuid, api_url):
    if os.path.exists(os.path.join(img_output_dir, f"{page_uuid}.jpg")):
        return True
    title_img_downloaded_and_valid = False
    if "ceskadigitalniknihovna" in api_url or "kramerius.kkvysociny" in api_url or "kramerius.kfbz" in api_url:
        download_img_url = f"{api_url}/search/api/client/v7.0/items/uuid:{page_uuid}/image"
    elif "kramerius.mzk" in api_url or "kramerius.cbvk" in api_url or "kramerius.svkpk" in api_url:
        download_img_url = f"{api_url}/search/iiif/uuid:{page_uuid}/full/max/0/default.jpg"
    elif "dikda.snk" in api_url or "k7.mlp" in api_url or "kramerius.knihovna-pardubice" in api_url:
        download_img_url = f"{api_url}/search/api/client/v7.0/items/uuid:{page_uuid}/image/iiif/full/max/0/default.jpg"
    else:
        print(f"Warning: Unknown API URL: {api_url}")
        download_img_url = f"{api_url}/search/iiif/uuid:{page_uuid}/image"
    urls_403_404 = []
    if not os.path.exists(os.path.join(img_output_dir, "403_404.txt")):
        with open(os.path.join(img_output_dir, "403_404.txt"), "w") as f:
            pass
    with open(os.path.join(img_output_dir, "403_404.txt"), "r") as f:
        urls_403_404 = f.read().splitlines()
    if download_img_url in urls_403_404:
        return title_img_downloaded_and_valid
    img_path = os.path.join(img_output_dir, f"{page_uuid}.jpg")
    for _ in range(2):
        try:
            response = requests.get(download_img_url)
        except requests.exceptions.ConnectTimeout:
            time.sleep(2)
            continue
        if response.status_code == 200:
            image = response.content
            with open(img_path, "wb") as f:
                f.write(image)
            try:
                with Image.open(img_path) as img:
                    if img is None:
                        raise IOError
                    img_size = os.path.getsize(img_path)
                    target_quality = 100
                    if img_size > 1000000:
                        target_quality = int(100 / (img_size / 1000000))
                    img.save(img_path, optimize=True, quality=target_quality)
                    title_img_downloaded_and_valid = True
                    break
            except (IOError, SyntaxError):
                print("Error: " + page_uuid + ".jpg downloaded but not valid image. Deleting...")
                print(f"URL: {download_img_url}")
                os.remove(img_path)
                continue
        else:
            if response.status_code in [403, 404]:
                if download_img_url not in urls_403_404:
                    if not os.path.exists(os.path.join(img_output_dir, "403_404.txt")):
                        with open(os.path.join(img_output_dir, "403_404.txt"), "w") as f:
                            f.write(download_img_url + "\n")
                    else:
                        with open(os.path.join(img_output_dir, "403_404.txt"), "a") as f:
                            f.write(download_img_url + "\n")
                break
            continue
    if not title_img_downloaded_and_valid:
        print(f"Download was not successful ({download_img_url})")
    return title_img_downloaded_and_valid


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    api = KrameriusAPI(args.api_url)

    if args.file and args.doc_model:
        with open(args.file, "r") as f:
            count = 0
            for line in get_lines_uuid(f, args.doc_model, args.exclude_collections):
                count += 1
            pbar = manager.counter(total=count, desc="Downloading metadata", unit=f"{args.doc_model}s")
        with open(args.file, "r") as f:
            for i, line_uuid in enumerate(get_lines_uuid(f, args.doc_model, args.exclude_collections)):
                if i < args.skip:
                    pbar.update()
                    continue
                api.download_complete_periodical_metadata(line_uuid, args)
                pbar.update()
    elif args.uuid:
        api.download_complete_periodical_metadata(args.uuid, args)
    else:
        raise ValueError("Either --file with --doc-model or --uuid must be specified")
