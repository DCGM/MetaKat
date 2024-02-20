import requests
import os
import time
import enlighten
import argparse

manager = enlighten.get_manager()


def get_response(url):
    retry_times = 5
    for _ in range(retry_times):
        try:
            response = requests.get(url)
        except requests.exceptions.ConnectTimeout:
            time.sleep(10)
            continue
        if response.status_code == 200 or response.status_code == 404:
            break
        time.sleep(2)
    if response.status_code != 200:
        print(f"Response status code: {response.status_code} after {retry_times} retries, URL: {url}, skipping...")
    return response


def lines_uuid_with_document_type_generator(f, document_type):
    for line in f:
        if f"'model': '{document_type}'" not in line:
            continue
        yield line.strip().split(" ")[0].split("uuid:")[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Download periodical metadata from Kramerius")
    parser.add_argument("--api-url", type=str, required=True, help="URL of the Kramerius API")
    parser.add_argument("--uuid", type=str, help="UUID of the periodical")
    parser.add_argument("--file", type=str, help="File with UUIDs of periodicals")
    parser.add_argument("--doc-model", type=str, help="Filter only lines from --file with this model")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")

    parser.add_argument("--pages", action="store_true", help="Download pages metadata")

    return parser.parse_args()


class Periodical:
    def __init__(self, uuid):
        self.uuid = uuid
        self.years = {}

        self.years_count = 0
        self.issues_count = 0
        self.pages_count = 0

    def add_year(self, year_uuid):
        self.years[year_uuid] = {}
        self.years_count += 1

    def add_issue(self, year_uuid, issue_uuid):
        self.years[year_uuid][issue_uuid] = {}
        self.issues_count += 1

    def add_page(self, year_uuid, issue_uuid, page_uuid):
        self.years[year_uuid][issue_uuid][page_uuid] = []
        self.pages_count += 1

    def get_years(self):
        return self.years.keys()

    def get_years_count(self):
        return self.years_count

    def get_issues(self, year_uuid):
        return self.years[year_uuid].keys()

    def get_issues_count(self):
        return self.issues_count

    def get_pages(self, year_uuid, issue_uuid):
        return self.years[year_uuid][issue_uuid]

    def get_pages_count(self):
        return self.pages_count


class KrameriusAPI:
    def __init__(self, base_url, version="v7.0"):
        self.base_url = base_url
        self.version = version

    def download_complete_periodical_metadata(self, uuid, args):
        print("=" * 80)
        print(f"Periodical {uuid}")
        
        output_dir = os.path.join(args.output_dir, uuid)
        os.makedirs(output_dir, exist_ok=True)

        periodical = Periodical(uuid)
        self.download_periodical_metadata(periodical, output_dir)
        print("Metadata for periodical downloaded")
        self.download_periodical_years_metadata(periodical, output_dir)
        print(f"Metadata for years downloaded ({periodical.get_years_count()} total)")
        self.download_periodical_issues_metadata(periodical, output_dir)
        print(f"Metadata for issues downloaded ({periodical.get_issues_count()} total)")
        if args.pages:
            self.download_periodical_pages_metadata(periodical, output_dir)
            print(f"Metadata for pages downloaded ({periodical.get_pages_count()} total)")

    def download_periodical_metadata(self, periodical, output_dir):
        url = f"{self.base_url}/search/api/client/{self.version}/items/uuid:{periodical.uuid}/metadata/mods"
        response = get_response(url)
        if response.status_code != 200:
            return
        with open(f"{output_dir}/uuid:{periodical.uuid}.xml", "w") as f:
            f.write(response.text)

    def download_periodical_years_metadata(self, periodical, output_dir):
        year_uuids = self.get_children_uuids(periodical.uuid, os.path.join(output_dir, "structure.txt"))
        for year_uuid in year_uuids:
            periodical.add_year(year_uuid)
            url = f"{self.base_url}/search/api/client/{self.version}/items/uuid:{year_uuid}/metadata/mods"

            year_dir = os.path.join(output_dir, year_uuid)
            os.makedirs(year_dir, exist_ok=True)
            year_path = os.path.join(year_dir, f"uuid:{year_uuid}.xml")
            if os.path.exists(year_path):
                print(f"Year metadata for {year_uuid} already exists, skipping downloading...")
                continue
            response = get_response(url)
            if response.status_code != 200:
                continue
            with open(year_path, "w") as f:
                f.write(response.text)

    def download_periodical_issues_metadata(self, periodical, output_dir):
        for year_uuid in periodical.get_years():
            issue_uuids = self.get_children_uuids(year_uuid, os.path.join(output_dir, year_uuid, "structure.txt"))
            for issue_uuid in issue_uuids:
                periodical.add_issue(year_uuid, issue_uuid)
                url = f"{self.base_url}/search/api/client/{self.version}/items/uuid:{issue_uuid}/metadata/mods"

                issue_dir = os.path.join(output_dir, year_uuid, issue_uuid)
                os.makedirs(issue_dir, exist_ok=True)
                issue_path = os.path.join(issue_dir, f"uuid:{issue_uuid}.xml")
                if os.path.exists(issue_path):
                    print(f"Issue metadata for {issue_uuid} already exists, skipping downloading...")
                    continue
                response = get_response(url)
                if response.status_code != 200:
                    continue
                with open(issue_path, "w") as f:
                    f.write(response.text)

    def download_periodical_pages_metadata(self, periodical, output_dir):
        for year_uuid in periodical.get_years():
            for issue_uuid in periodical.get_issues(year_uuid):
                page_uuids = self.get_children_uuids(issue_uuid, os.path.join(output_dir, year_uuid, issue_uuid, "structure.txt"))

                for page_uuid in page_uuids:
                    periodical.add_page(year_uuid, issue_uuid, page_uuid)
                    url = f"{self.base_url}/search/api/client/{self.version}/items/uuid:{page_uuid}/metadata/mods"

                    page_dir = os.path.join(output_dir, year_uuid, issue_uuid, page_uuid)
                    os.makedirs(page_dir, exist_ok=True)
                    page_path = os.path.join(page_dir, f"uuid:{page_uuid}.xml")
                    if os.path.exists(page_path):
                        print(f"Page metadata for {page_uuid} already exists, skipping downloading...")
                    else:
                        response = get_response(url)
                        if response.status_code != 200:
                            continue
                        with open(page_path, "w") as f:
                            f.write(response.text)

    def get_children_uuids(self, uuid, structure_file):
        if os.path.exists(structure_file):
            with open(structure_file, "r") as f:
                return [line.strip() for line in f]

        url = f"{self.base_url}/search/api/client/{self.version}/items/uuid:{uuid}/info/structure"
        response = get_response(url)
        if response.status_code != 200:
            print(f"Item {uuid} not found, returning empty list of children...")
            return []
        response = response.json()
        children = response["children"]["own"]
        uuids = [child["pid"].split("uuid:")[1] for child in children]
        
        with open(structure_file, "w") as f:
            for child in uuids:
                f.write(child + "\n")
        return uuids


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    api = KrameriusAPI(args.api_url)

    if args.file and args.doc_model:
        pbar = manager.counter(total=sum(1 for line in open(args.file) if f"'model': '{args.doc_model}'" in line), desc="Downloading metadata", unit="periodicals")
        with open(args.file, "r") as f:
            for line_uuid in lines_uuid_with_document_type_generator(f, args.doc_model):
                api.download_complete_periodical_metadata(line_uuid, args)
                pbar.update()
    elif args.uuid:
        api.download_complete_periodical_metadata(args.uuid, args)
    else:
        raise ValueError("Either --file with --doc-model or --uuid must be specified")
