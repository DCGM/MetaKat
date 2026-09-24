"""Rewrite an engine configuration's page-type labels to the NDK page-type words.

PageType values used to be the Kramerius spellings ("TitlePage"); they are now
the words of the NDK description rules ("titlePage"), so that MODS gets them
unchanged. An engine configuration's page_type.core.labels maps PageType
values to the labels the model outputs, so its keys have to follow. The values
- the model's own labels - are left as they are.

    python -m metakat.tools.migrate_page_type_labels engine_config.json [--in-place]

Without --in-place the migrated configuration is printed as JSON, ready to be
pasted into an engine definition on the server.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from metakat.schemas.base_objects import PageType


def _kramerius_spelling(page_type: PageType) -> str:
    if page_type is PageType.FLY_LEAF:
        return "FlyLeaf"
    return page_type.value[0].upper() + page_type.value[1:]


_NEW_BY_OLD = {_kramerius_spelling(page_type): page_type.value for page_type in PageType}


def migrate_labels(labels: dict[str, str]) -> dict[str, str]:
    """Return labels with every key in the current PageType spelling."""
    current = {page_type.value for page_type in PageType}
    migrated: dict[str, str] = {}
    for key, model_label in labels.items():
        if key in current:
            new_key = key
        elif key in _NEW_BY_OLD:
            new_key = _NEW_BY_OLD[key]
        else:
            raise ValueError(f"Unknown page-type label key: {key!r}")
        if new_key in migrated:
            raise ValueError(f"Page type {new_key!r} is labelled twice")
        migrated[new_key] = model_label
    return migrated


def migrate_config(config: dict) -> bool:
    """Migrate page_type.core.labels in place; return whether anything changed."""
    core = (config.get("page_type") or {}).get("core") or {}
    labels = core.get("labels")
    if not labels:
        return False
    migrated = migrate_labels(labels)
    if migrated == labels:
        return False
    core["labels"] = migrated
    return True


def _load(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text)
    return json.loads(text)


def _dump(path: Path, config: dict) -> None:
    if path.suffix in (".yaml", ".yml"):
        import yaml

        text = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    else:
        text = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", type=Path, help="Engine configuration, JSON or YAML")
    parser.add_argument("--in-place", action="store_true", help="Rewrite the file")
    args = parser.parse_args()

    config = _load(args.config)
    changed = migrate_config(config)
    if args.in_place:
        if changed:
            _dump(args.config, config)
        print(f"{args.config}: {'migrated' if changed else 'already current'}", file=sys.stderr)
    else:
        json.dump(config, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
