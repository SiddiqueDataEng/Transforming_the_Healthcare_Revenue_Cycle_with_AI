"""
Schema migration registry for ClaimFeatureRecord Avro schemas.

Stores (from_version, to_version) → migration_fn pairs and applies
chained migrations when an embedded schema version differs from the
current active version.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple


class NoMigrationRuleError(Exception):
    """Raised when no migration rule exists for a given version pair."""

    def __init__(self, from_version: str, to_version: str) -> None:
        self.from_version = from_version
        self.to_version = to_version
        super().__init__(
            f"No migration rule registered for ({from_version!r} → {to_version!r})"
        )


MigrationFn = Callable[[dict], dict]


class SchemaRegistry:
    """
    Registry that maps (from_version, to_version) pairs to migration functions.

    Usage::

        registry = SchemaRegistry()
        registry.register("1.0.0", "1.1.0", migrate_100_to_110)
        updated = registry.apply_migrations(record_dict, "1.0.0", "1.1.0")
    """

    def __init__(self) -> None:
        self._rules: Dict[Tuple[str, str], MigrationFn] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        from_version: str,
        to_version: str,
        fn: MigrationFn,
    ) -> None:
        """Register a migration function for a version pair.

        Parameters
        ----------
        from_version:
            The schema version being migrated *from*.
        to_version:
            The schema version being migrated *to*.
        fn:
            A callable that accepts a ``dict`` (the record fields) and
            returns a new ``dict`` with the migration applied.  The
            function MUST NOT mutate its input in place.
        """
        self._rules[(from_version, to_version)] = fn

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def apply_migrations(
        self,
        record_dict: dict,
        embedded_version: str,
        current_version: str,
    ) -> dict:
        """Apply chained migrations from *embedded_version* to *current_version*.

        The registry resolves a migration path by following registered
        rules one hop at a time.  For example, if rules exist for
        ``("1.0.0", "1.1.0")`` and ``("1.1.0", "2.0.0")``, calling
        ``apply_migrations(d, "1.0.0", "2.0.0")`` will apply both in
        order.

        Parameters
        ----------
        record_dict:
            The raw deserialized record as a plain Python ``dict``.
        embedded_version:
            The ``schema_version`` value stored inside the record.
        current_version:
            The version the caller wants the record to conform to.

        Returns
        -------
        dict
            A new dict with all applicable migrations applied.

        Raises
        ------
        NoMigrationRuleError
            If no registered path exists between *embedded_version* and
            *current_version*.
        """
        if embedded_version == current_version:
            return record_dict

        result = dict(record_dict)
        current = embedded_version

        while current != current_version:
            # Find any rule whose source matches *current*.
            next_hop = self._find_next_hop(current, current_version)
            if next_hop is None:
                raise NoMigrationRuleError(current, current_version)
            fn = self._rules[(current, next_hop)]
            result = fn(result)
            current = next_hop

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _find_next_hop(self, from_version: str, target_version: str) -> str | None:
        """Return the best next hop from *from_version* toward *target_version*.

        Preference order:
        1. A direct rule ``(from_version, target_version)`` if it exists.
        2. Any rule ``(from_version, x)`` for some intermediate *x*.

        Returns ``None`` if no outgoing rule from *from_version* exists.
        """
        # Prefer direct jump to target
        if (from_version, target_version) in self._rules:
            return target_version

        # Otherwise take the first available outgoing edge
        for (src, dst) in self._rules:
            if src == from_version:
                return dst

        return None
