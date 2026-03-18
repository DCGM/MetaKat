from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel


class Subchapter(BaseModel):
    subchapter_name: str
    subchapter_number: Optional[int] = None
    start_page: Optional[int] = None
    final_page: Optional[int] = None
    author: Optional[str] = None
    subchapters: Optional[List[Subchapter]] = []


class Chapter(BaseModel):
    chapter_name: str
    chapter_number: Optional[int] = None
    start_page: Optional[int] = None
    final_page: Optional[int] = None
    author: Optional[str] = None
    subchapters: Optional[List[Subchapter]] = []


class SectionItem(BaseModel):
    section_name: str
    section_number: Optional[int] = None
    start_page: Optional[int] = None
    final_page: Optional[int] = None
    author: Optional[str] = None
    sections: Optional[List[SectionItem]] = None
    chapters: Optional[List[Chapter]] = None


class Model(BaseModel):
    section: Optional[List[SectionItem]] = None
    chapters: Optional[List[Chapter]] = None
