import xml.etree.ElementTree as ET


def lxml_element_to_str(e):
    return ET.tostring(e).decode('utf-8')
