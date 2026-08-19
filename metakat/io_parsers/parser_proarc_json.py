import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
from uuid import UUID, uuid4

from metakat.io_parsers.parser_mods import parse_mods
from metakat.schemas.base_objects import (
    HierarchyType,
    MetakatElement,
    MetakatIO,
    MetakatIssue,
    MetakatTitle,
    MetakatVolume,
    ObjectItem,
    ObjectModel,
    PackageType,
    ProarcIO,
)

logger = logging.getLogger(__name__)

# ProArc metadata is catalog ground truth, not a detection - there is no
# confidence score to preserve, so every value is stamped as fully confident.
GROUND_TRUTH_CONFIDENCE = 1.0


def _value(text: Optional[str]) -> Optional[Tuple[str, float, UUID]]:
    if text is None:
        return None
    return text, GROUND_TRUTH_CONFIDENCE, uuid4()


def _values(texts: Optional[List[str]]) -> Optional[List[Tuple[str, float, UUID]]]:
    if not texts:
        return None
    return [_value(text) for text in texts]


def _package_hierarchy(package: ProarcIO) -> HierarchyType:
    if package.type == PackageType.periodical:
        return HierarchyType.PERIODICAL
    has_title = any(obj.model == ObjectModel.title for obj in package.objects)
    return HierarchyType.MULTIPART if has_title else HierarchyType.MONOGRAPH


def _title_element(obj: ObjectItem, hierarchy: HierarchyType) -> MetakatTitle:
    return MetakatTitle(
        id=uuid4(),
        hierarchy=hierarchy,
        title=_value(obj.title),
        subTitle=_value(obj.subTitle),
    )


def _issue_element(obj: ObjectItem, parent_id: Optional[UUID]) -> MetakatIssue:
    return MetakatIssue(
        id=uuid4(),
        parent_id=parent_id,
        partNumber=_value(obj.partNumber),
        dateIssued=_value(obj.dateIssued),
        title=_value(obj.title),
        subTitle=_value(obj.subTitle),
        placeTerm=_value(obj.placeTerm),
        publisher=_values(obj.publisher),
        manufacturePublisher=_values(obj.manufacturePublisher),
        manufacturePlaceTerm=_values(obj.manufacturePlaceTerm),
        redaktor=_values(obj.redaktor),
    )


def _volume_element(obj: ObjectItem, hierarchy: HierarchyType, parent_id: Optional[UUID]) -> MetakatVolume:
    return MetakatVolume(
        id=uuid4(),
        parent_id=parent_id,
        hierarchy=hierarchy,
        partNumber=_value(obj.partNumber),
        partName=_value(obj.partName),
        dateIssued=_value(obj.dateIssued),
        title=_value(obj.title),
        subTitle=_value(obj.subTitle),
        edition=_value(obj.edition),
        placeTerm=_value(obj.placeTerm),
        publisher=_values(obj.publisher),
        manufacturePublisher=_values(obj.manufacturePublisher),
        manufacturePlaceTerm=_values(obj.manufacturePlaceTerm),
        author=_values(obj.author),
        illustrator=_values(obj.illustrator),
        photographer=_values(obj.photographer),
        translator=_values(obj.translator),
        editor=_values(obj.editor),
        seriesName=_values(obj.seriesName),
        seriesNumber=_values(obj.seriesNumber),
    )


def parse_proarc_package(data: dict, batch_id: Optional[UUID] = None) -> MetakatIO:
    """Parse a ProArc packageInfo.json dict into MetaKat's internal representation."""
    package = ProarcIO.model_validate(data)
    for obj in package.objects:
        parsed = parse_mods(obj.metadata)
        for key, value in parsed.items():
            setattr(obj, key, value)

    hierarchy = _package_hierarchy(package)

    title_obj = next((obj for obj in package.objects if obj.model == ObjectModel.title), None)
    elements: List[MetakatElement] = []
    title_id = None
    if title_obj is not None:
        title_element = _title_element(title_obj, hierarchy)
        title_id = title_element.id
        elements.append(title_element)

    for obj in package.objects:
        if obj.model == ObjectModel.title:
            continue
        if obj.model == ObjectModel.unit:
            logger.warning("ProArc object %s has model 'unit'; treating it like 'volume' - "
                            "no sample data was available to confirm this mapping", obj.pid)

        if hierarchy == HierarchyType.PERIODICAL:
            elements.append(_issue_element(obj, title_id))
        else:
            elements.append(_volume_element(obj, hierarchy, title_id))

    return MetakatIO(batch_id=batch_id or uuid4(), elements=elements)


def parse_proarc_json_file(path: Union[str, Path], batch_id: Optional[UUID] = None) -> MetakatIO:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_proarc_package(data, batch_id=batch_id)
