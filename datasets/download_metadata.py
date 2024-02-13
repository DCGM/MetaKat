import requests
import os
import time
import enlighten
import argparse
from pathlib import Path

manager = enlighten.get_manager()

def parse_args():
    parser = argparse.ArgumentParser(description="Download periodical metadata from Kramerius")
    parser.add_argument("--api-url", type=str, required=True, help="URL of the Kramerius API")
    parser.add_argument("--uuid", type=str, required=True, help="UUID of the periodical")
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
        output_dir = args.output_dir
        
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
        response = requests.get(url)        
        with open(f"{output_dir}/uuid:{periodical.uuid}.xml", "w") as f:
            f.write(response.text)
    
    def download_periodical_years_metadata(self, periodical, output_dir):
        year_uuids = self.get_children_uuids(periodical.uuid)
        for year_uuid in year_uuids:
            periodical.add_year(year_uuid)
            url = f"{self.base_url}/search/api/client/{self.version}/items/uuid:{year_uuid}/metadata/mods"
            
            year_dir = os.path.join(output_dir, year_uuid)
            try:
                os.makedirs(year_dir, exist_ok=False)
            except FileExistsError:
                print(f"Year directory for {os.path.basename(year_dir)} already exists, skipping downloading...")
                continue
            with open(f"{year_dir}/uuid:{year_uuid}.xml", "w") as f:
                f.write(requests.get(url).text)
                
    def download_periodical_issues_metadata(self, periodical, output_dir):
        for year_uuid in periodical.get_years():
            issue_uuids = self.get_children_uuids(year_uuid)
            for issue_uuid in issue_uuids:
                periodical.add_issue(year_uuid, issue_uuid)
                url = f"{self.base_url}/search/api/client/{self.version}/items/uuid:{issue_uuid}/metadata/mods"
                
                issue_dir = os.path.join(output_dir, year_uuid, issue_uuid)
                try:
                    os.makedirs(issue_dir, exist_ok=False)
                except FileExistsError:
                    print(f"Issue directory for {os.path.basename(issue_dir)} already exists, skipping downloading...")
                    continue
                with open(f"{issue_dir}/uuid:{issue_uuid}.xml", "w") as f:
                    f.write(requests.get(url).text)
    
    def download_periodical_pages_metadata(self, periodical, output_dir):
        for year_uuid in periodical.get_years():
            for issue_uuid in periodical.get_issues(year_uuid):
                page_uuids = self.get_children_uuids(issue_uuid)
                
                pbar = manager.counter(total=len(page_uuids), desc="Downloading pages metadata", unit="pages metadata")
                
                for page_uuid in page_uuids:
                    periodical.add_page(year_uuid, issue_uuid, page_uuid)
                    url = f"{self.base_url}/search/api/client/{self.version}/items/uuid:{page_uuid}/metadata/mods"
                    
                    page_dir = os.path.join(output_dir, year_uuid, issue_uuid, page_uuid)
                    try:
                        os.makedirs(page_dir, exist_ok=False)
                        with open(f"{page_dir}/uuid:{page_uuid}.xml", "w") as f:
                            f.write(requests.get(url).text)
                    except FileExistsError:
                        pass
                    
                    pbar.update()
                    time.sleep(0.1)
                    
                # TODO alter sleep time
                time.sleep(5)
                pbar.close()

    def get_children_uuids(self, uuid):
        url = f"{self.base_url}/search/api/client/{self.version}/items/uuid:{uuid}/info/structure"
        response = requests.get(url).json()
        children = response["children"]["own"]
        uuids = [child["pid"].split("uuid:")[1] for child in children]
        return uuids

if __name__ == "__main__":
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    api = KrameriusAPI(args.api_url)
    api.download_complete_periodical_metadata(args.uuid, args)
    
