from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel


class Element(BaseModel):
    type: str  # section, chapter, subchapter
    name: str
    number: Optional[int] = None
    start_page: Optional[int] = None
    final_page: Optional[int] = None
    author: Optional[str] = None
    children: Optional[List[Element]] = None


class Model(BaseModel):
    content: List[Element]
