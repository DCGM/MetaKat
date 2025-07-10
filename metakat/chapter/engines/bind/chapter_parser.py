



def parse_page_number(page_number: str) -> str | bool:
    try:
        page_number = str(int(page_number.strip()))
        return page_number
    except ValueError:
        return False