"""
Bundle modular OpenAPI YAML files into a single openapi.json file.

Resolves all external $ref references in the OpenAPI specification and outputs
a single bundled JSON file for easier consumption and distribution.
"""

import json
from pathlib import Path
import yaml
from loguru import logger


def resolve_ref_pointer(data, pointer):
    # type: (dict, str) -> Any
    """
    Resolve a JSON pointer path within a data structure.

    :param data: Dictionary to navigate
    :param pointer: JSON pointer path (e.g., '/properties/field')
    :return: Value at the pointer location
    """
    if not pointer or pointer == "/":
        return data

    # Remove leading slash and split path
    parts = pointer.lstrip("/").split("/")

    current = data
    for part in parts:
        # Unescape JSON pointer special characters
        part = part.replace("~1", "/").replace("~0", "~")

        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            raise ValueError(f"Cannot navigate pointer {pointer} - invalid path")

    return current


def resolve_refs(data, base_path, visited=None):
    # type: (Any, Path, set[str] | None) -> Any
    """
    Recursively resolve all $ref references in the OpenAPI spec.

    Uses stack-based circular reference detection: visited tracks only the current
    recursion path, allowing the same reference to be resolved multiple times in
    different branches while preventing infinite loops.

    :param data: Data structure to process (dict, list, or primitive)
    :param base_path: Base directory for resolving relative file paths
    :param visited: Set of references in current recursion stack (circular detection)
    :return: Data structure with all references resolved
    """
    if visited is None:
        visited = set()

    if isinstance(data, dict):
        # Check for $ref key
        if "$ref" in data:
            ref = data["$ref"]

            # Handle internal references (e.g., '#/components/schemas/Foo')
            if ref.startswith("#"):
                # Internal references can't be resolved without full context
                # Keep them as-is (will be resolved by OpenAPI consumers)
                return data

            # Handle external file references (e.g., './IsccEntry.yaml' or './CommonProperties.yaml#/properties/field')
            if "#" in ref:
                file_path, json_pointer = ref.split("#", 1)
            else:
                file_path = ref
                json_pointer = ""

            # Resolve file path relative to base_path
            full_path = (base_path / file_path).resolve()

            # Check for circular references (stack-based detection)
            path_key = str(full_path) + "#" + json_pointer
            if path_key in visited:
                logger.warning(f"Circular reference detected: {path_key}")
                return data

            # Load referenced file
            if not full_path.exists():
                logger.error(f"Referenced file not found: {full_path}")
                return data

            logger.debug(f"Resolving reference: {ref}")
            with open(full_path, encoding="utf-8") as f:
                referenced_data = yaml.safe_load(f)

            # Navigate to specific JSON pointer if provided
            if json_pointer:
                referenced_data = resolve_ref_pointer(referenced_data, json_pointer)

            # Add to recursion stack, process, then remove (stack-based tracking)
            visited.add(path_key)
            try:
                result = resolve_refs(referenced_data, full_path.parent, visited)
            finally:
                visited.remove(path_key)

            return result

        # Recursively process all dictionary values
        return {key: resolve_refs(value, base_path, visited) for key, value in data.items()}

    elif isinstance(data, list):
        # Recursively process all list items
        return [resolve_refs(item, base_path, visited) for item in data]

    else:
        # Primitive types: return as-is
        return data


def bundle_openapi():
    # type: () -> None
    """Bundle OpenAPI YAML files into single JSON file."""
    # Determine paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    openapi_dir = project_root / "iscc_search" / "openapi"
    source = openapi_dir / "openapi.yaml"
    target = openapi_dir / "openapi.json"

    # Validate source exists
    if not source.exists():
        logger.error(f"Source file not found: {source}")
        return

    logger.info(f"Loading OpenAPI spec from {source}")

    # Load main OpenAPI file
    with open(source, encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    logger.info("Resolving external $ref references...")

    # Resolve all external references
    bundled_spec = resolve_refs(spec, openapi_dir)

    logger.info(f"Writing bundled spec to {target}")

    # Write bundled JSON with LF line endings
    with open(target, "w", encoding="utf-8", newline="\n") as f:
        json.dump(bundled_spec, f, indent=2, ensure_ascii=False)

    logger.success(f"✓ Successfully bundled {source.name} → {target.name}")


if __name__ == "__main__":
    bundle_openapi()
