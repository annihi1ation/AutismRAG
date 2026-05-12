"""Review queue helpers for ambiguous or failed HyperKG objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .utils import to_plain


@dataclass
class ReviewQueue:
    items: List[Dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        item_type: str,
        object_id: str,
        reason: str,
        suggested_action: str = "",
        article_id: str = "",
        segment_id: str = "",
        summary_text: str = "",
        object_json: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        item = {
            "review_item_id": f"REV:{len(self.items) + 1:06d}",
            "type": item_type,
            "object_id": object_id,
            "article_id": article_id,
            "segment_id": segment_id,
            "summary_text": summary_text,
            "object_json": to_plain(object_json or {}),
            "reason": reason,
            "suggested_action": suggested_action,
        }
        if extra:
            item.update(to_plain(extra))
        self.items.append(item)
        return item

    def extend(self, items: List[Dict[str, Any]]) -> None:
        for item in items:
            if "review_item_id" not in item:
                item = dict(item)
                item["review_item_id"] = f"REV:{len(self.items) + 1:06d}"
            self.items.append(item)

    def to_list(self) -> List[Dict[str, Any]]:
        return list(self.items)
