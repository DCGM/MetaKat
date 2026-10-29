import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any
from typing import List

from metakat.page_number.engines.core.models import PageNumberCoreResult
from metakat.page_number.engines.core.ndk_pagination import NdkPageContext
from metakat.engine_config import require_config_mapping, require_engine_name


logger = logging.getLogger(__name__)


class PageNumberCoreEngine(ABC):
    def __init__(self, config: Mapping[str, Any]):
        self.config = require_config_mapping(config, "Page-number core config")
        self.name = require_engine_name(self.config, "Page-number core config")
        logger.info("Loaded page number core engine: %s", self.name)

    @abstractmethod
    def process(
        self,
        images: List[str],
        alto_files: List[str],
        *,
        page_contexts: Mapping[str, NdkPageContext] | None = None,
        final: bool = False,
    ) -> PageNumberCoreResult:
        """Return one entry per input image, in order; use None when unresolved.

        The default is the preliminary pass, which supports the page-type stage and supplies plain numbers
        only. The final pass runs after the page-type stage, with the documents it found as page_contexts
        units, and also supplies bracketed labels and the labels at the document edges."""
        pass
