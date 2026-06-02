"""
Thin Neo4j client. The API is best-effort: if Neo4j is not reachable
(docker not started, wrong creds, etc.) we log and fall back to an
in-memory representation so the rest of the app keeps working.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable

from neo4j import GraphDatabase, Driver

from app.core.config import settings

log = logging.getLogger(__name__)


class Neo4jClient:
    def __init__(self) -> None:
        self._driver: Driver | None = None
        try:
            self._driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                connection_timeout=3,
            )
            self._driver.verify_connectivity()
            log.info("Neo4j connected at %s", settings.NEO4J_URI)
        except Exception as e:
            log.warning("Neo4j unavailable (%s) — running in memory-only mode", e)
            self._driver = None

    @property
    def available(self) -> bool:
        return self._driver is not None

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

    def run_write(self, cypher: str, params: dict[str, Any] | None = None) -> None:
        if not self.available:
            return
        with self._driver.session() as session:
            session.execute_write(lambda tx: tx.run(cypher, params or {}))

    def run_read(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict]:
        if not self.available:
            return []
        with self._driver.session() as session:
            result = session.execute_read(lambda tx: tx.run(cypher, params or {}))
            return [r.data() for r in result]

    def wipe(self) -> None:
        self.run_write("MATCH (n) DETACH DELETE n")

    def upsert_node(self, label: str, key: str, props: dict[str, Any]) -> None:
        props = {**props, key: props[key]}
        cypher = (
            f"MERGE (n:`{label}` {{{key}: ${key}}}) "
            f"SET n += $props"
        )
        self.run_write(cypher, {key: props[key], "props": props})

    def upsert_edge(self, src_label: str, src_id: str, dst_label: str, dst_id: str,
                    rel: str, props: dict[str, Any] | None = None) -> None:
        cypher = (
            f"MATCH (a:`{src_label}` {{id: $src_id}}), (b:`{dst_label}` {{id: $dst_id}}) "
            f"MERGE (a)-[r:{rel}]->(b) "
            f"SET r += $props"
        )
        self.run_write(cypher, {
            "src_id": src_id, "dst_id": dst_id, "props": props or {},
        })

    def count(self, label: str) -> int:
        rows = self.run_read(f"MATCH (n:`{label}`) RETURN count(n) AS c")
        return rows[0]["c"] if rows else 0

    def patient_subgraph(self, patient_id: str) -> dict:
        cypher = """
        MATCH (p:Patient {patient_id: $pid})-[r:HAS_VALUE]->(b:Biomarker)
        OPTIONAL MATCH (b)-[r2:REGULATED_BY|ELEVATED_IN|DEPRESSED_IN]->(tgt)
               WHERE tgt:Organ OR tgt:Disease
        RETURN p, b, r, collect({rel: type(r2), target_id: tgt.id,
                                  target_kind: labels(tgt)[0]}) AS neighbors
        """
        rows = self.run_read(cypher, {"pid": patient_id})
        return {"rows": rows}


neo4j_client = Neo4jClient()
