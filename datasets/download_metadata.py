import requests
import os
import time
import enlighten
import argparse
from lxml import etree

manager = enlighten.get_manager()

def frequency_to_number(frequency):
    mapping = {
        "Annual": 365,
        "Biennial": 730,
        "Bimonthly": 60,
        "Biweekly": 14,
        "Daily": 1,
        "denně": 1,
        "Monthly": 30,
        "Quarterly": 90,
        "Semimonthly": 15,
        "Semiweekly": 3,
        "Three times a month": 10,
        "Triennial": 1095,
        "Unknown": 0,
        "Weekly": 7
    }
    
    try:
        return mapping[frequency]
    except KeyError:
        print(f"Unknown frequency: {frequency}, returning 0")
        return 0


def get_response(url, output_dir=None):
    retry_times = 5
    for _ in range(retry_times):
        try:
            response = requests.get(url)
        except requests.exceptions.ConnectTimeout:
            time.sleep(5)
            continue
        if response.status_code == 200 or response.status_code == 404:
            break
        time.sleep(1)
    if response.status_code != 200:
        print(f"Response status code: {response.status_code}, URL: {url}, skipping...")
    if response.status_code == 404 and output_dir is not None:
        structure_file = os.path.join("..", output_dir, "structure.txt")
        uuid = url.split("uuid:")[1].split("/")[0]
        with open(structure_file, "r") as f:
            lines = f.readlines()
        with open(structure_file, "w") as f:
            for line in lines:
                if uuid in line:
                    f.write(f"{uuid} 404: Not Found\n")
                else:
                    f.write(line)
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

    parser.add_argument("--pages", action="store_true", help="Download pages metadata", default=False)
    parser.add_argument("--title-pages", action="store_true", help="Download only title pages metadata", default=False)
    parser.add_argument("--max-freq", type=int, help="Minimum frequency of periodicals to download", default=0)

    return parser.parse_args()


class Periodical:
    def __init__(self, uuid):
        self.uuid = uuid
        self.years = {}
        self.frequency = 0

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
        
        if args.max_freq != 0 and (periodical.frequency > args.max_freq or periodical.frequency == 0):
            print(f"Frequency of periodical is {periodical.frequency}, not continuing downloading...")
            return
            
        print("Metadata for periodical downloaded")
        self.download_periodical_years_metadata(periodical, output_dir)
        print(f"Metadata for years downloaded ({periodical.get_years_count()} total)")
        self.download_periodical_issues_metadata(periodical, output_dir)
        print(f"Metadata for issues downloaded ({periodical.get_issues_count()} total)")
        if args.pages or args.title_pages:
            self.download_periodical_pages_metadata(periodical, output_dir, title_pages_only=args.title_pages)
            print(f"Metadata for title pages downloaded ({periodical.get_pages_count()} total)")

    def download_periodical_metadata(self, periodical, output_dir):
        url = f"{self.base_url}/search/api/client/{self.version}/items/uuid:{periodical.uuid}/metadata/mods"
        response = get_response(url, output_dir)
        if response.status_code != 200:
            return
        with open(f"{output_dir}/uuid:{periodical.uuid}.xml", "w") as f:
            f.write(response.text)
        
        frequency = None
        namespaces = {"mods": "http://www.loc.gov/mods/v3"}
        tree = etree.fromstring(response.text.encode())
        for f in tree.iterfind(".//mods:frequency", namespaces):
            if f.get("authority") != "marcfrequency":
                frequency = f
                break
        if frequency is not None:
            periodical.frequency = frequency_to_number(frequency.text)
        

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
            response = get_response(url, output_dir)
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
                response = get_response(url, output_dir)
                if response.status_code != 200:
                    continue
                with open(issue_path, "w") as f:
                    f.write(response.text)

    def download_periodical_pages_metadata(self, periodical, output_dir, title_pages_only=False):
        periodical_years = periodical.get_years()
        for year_num, year_uuid in enumerate(periodical_years):
            if len(periodical_years) > 5:
                if year_num % (len(periodical_years) // 4) != 0:
                    continue
                
            periodical_issues = periodical.get_issues(year_uuid)
            for issuen_num, issue_uuid in enumerate(periodical_issues):
                if len(periodical_issues) > 5:
                    if issuen_num % (len(periodical_issues) // 4) != 0:
                        continue
                
                issue_dir = os.path.join(output_dir, year_uuid, issue_uuid)
                page_uuids = self.get_children_uuids(issue_uuid, os.path.join(issue_dir, "structure.txt"))

                
                for page_num, page_uuid in enumerate(page_uuids):                    
                    title_page_exists = False
                    if title_pages_only:
                        for dirpath, _, filenames in os.walk(issue_dir):
                            for filename in filenames:
                                if "_title.xml" in filename:
                                    print(f"Title page for issue {issue_uuid} already exists, skipping downloading...")
                                    title_page_exists = True
                    
                    if title_page_exists:
                        break                
                        
                    end_searching = False
                    periodical.add_page(year_uuid, issue_uuid, page_uuid)
                    url = f"{self.base_url}/search/api/client/{self.version}/items/uuid:{page_uuid}/metadata/mods"

                    page_dir = os.path.join(output_dir, year_uuid, issue_uuid, page_uuid)
                    os.makedirs(page_dir, exist_ok=True)
                    page_path = os.path.join(page_dir, f"uuid:{page_uuid}.xml")
                    if os.path.exists(page_path):
                        print(f"Page metadata for {page_uuid} already exists, skipping downloading...")
                    else:
                        response = get_response(url, output_dir)
                        if response.status_code != 200:
                            continue
                        
                        namespaces = {"mods": "http://www.loc.gov/mods/v3"}
                        tree = etree.fromstring(response.text.encode())
                        if tree.find(".//mods:part[@type='TitlePage']", namespaces) is not None:
                            page_path = os.path.join(page_dir, f"uuid:{page_uuid}_title.xml")
                            if title_pages_only:
                                end_searching = True
                        elif title_pages_only and page_num > 10:
                            end_searching = True
                            print(f"Title page not found withing first 10 pages, ending search for this issue...")

                        with open(page_path, "w") as f:
                            f.write(response.text)
                        
                        if end_searching:
                            break


    def get_children_uuids(self, uuid, structure_file):
        if os.path.exists(structure_file):
            with open(structure_file, "r") as f:
                lines = []
                for line in f:
                    if "404: Not Found" in line:
                        continue
                    lines.append(line.strip())

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
