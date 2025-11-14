"""
Remote index client implementation.

Provides HTTP client for interacting with remote ISCC-Search API servers,
implementing the IsccIndexProtocol interface.
"""

from typing import TYPE_CHECKING

import httpx
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

if TYPE_CHECKING:
    from iscc_search.schema import IsccAddResult  # noqa: F401
    from iscc_search.schema import IsccEntry  # noqa: F401
    from iscc_search.schema import IsccIndex  # noqa: F401
    from iscc_search.schema import IsccQuery  # noqa: F401
    from iscc_search.schema import IsccSearchResult  # noqa: F401


__all__ = ["RemoteIndex"]


console = Console()


class RemoteIndex:
    """
    Remote index client implementing IsccIndexProtocol.

    Connects to a remote ISCC-Search server via HTTP and provides the same
    interface as local index implementations. Supports chunked batch operations
    with progress reporting for large datasets.
    """

    def __init__(self, url, index_name, api_key=None, chunk_size=100):
        # type: (str, str, str|None, int) -> None
        """
        Initialize remote index client.

        :param url: Base URL of remote server (e.g., "https://api.example.com")
        :param index_name: Name of index on remote server
        :param api_key: Optional API key for authentication
        :param chunk_size: Number of assets per batch for add operations (default: 100)
        """
        self.url = url.rstrip("/")
        self.index_name = index_name
        self.api_key = api_key
        self.chunk_size = chunk_size
        self._client = None  # type: httpx.Client|None

    @property
    def client(self):
        # type: () -> httpx.Client
        """
        Get or create HTTP client.

        Lazy initialization of httpx client with authentication headers.

        :return: httpx.Client instance
        """
        if self._client is None:
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            self._client = httpx.Client(
                base_url=self.url,
                headers=headers,
                timeout=60.0,  # 60 seconds timeout
            )
        return self._client

    def _handle_response_errors(self, response):
        # type: (httpx.Response) -> None
        """
        Convert HTTP error responses to appropriate Python exceptions.

        :param response: httpx Response object
        :raises FileExistsError: For 409 Conflict (resource already exists)
        :raises FileNotFoundError: For 404 Not Found
        :raises ValueError: For 400 Bad Request (validation errors)
        :raises RuntimeError: For other HTTP errors
        """
        if response.is_success:
            return

        try:
            error_detail = response.json().get("detail", response.text)
        except Exception:
            error_detail = response.text

        if response.status_code == 404:
            raise FileNotFoundError(error_detail)
        elif response.status_code == 409:
            raise FileExistsError(error_detail)
        elif response.status_code == 400:
            raise ValueError(error_detail)
        else:
            raise RuntimeError(f"HTTP {response.status_code}: {error_detail}")

    def list_indexes(self):
        # type: () -> list[IsccIndex]
        """
        List all available indexes with metadata.

        :return: List of IsccIndex objects with name, assets, and size
        """
        from iscc_search.schema import IsccIndex

        response = self.client.get("/indexes")
        self._handle_response_errors(response)
        data = response.json()
        return [IsccIndex(**idx) for idx in data]

    def create_index(self, index):
        # type: (IsccIndex) -> IsccIndex
        """
        Create a new named index.

        :param index: IsccIndex with name (assets and size fields are ignored)
        :return: Created IsccIndex with initial metadata (assets=0, size=0)
        :raises ValueError: If name is invalid
        :raises FileExistsError: If index with this name already exists
        """
        from iscc_search.schema import IsccIndex

        response = self.client.post("/indexes", json={"name": index.name})
        self._handle_response_errors(response)
        return IsccIndex(**response.json())

    def get_index(self, name):
        # type: (str) -> IsccIndex
        """
        Get index metadata by name.

        :param name: Index name
        :return: IsccIndex with current metadata
        :raises FileNotFoundError: If index doesn't exist
        """
        from iscc_search.schema import IsccIndex

        response = self.client.get(f"/indexes/{name}")
        self._handle_response_errors(response)
        return IsccIndex(**response.json())

    def delete_index(self, name):
        # type: (str) -> None
        """
        Delete an index and all its data.

        :param name: Index name
        :raises FileNotFoundError: If index doesn't exist
        """
        response = self.client.delete(f"/indexes/{name}")
        self._handle_response_errors(response)

    def add_assets(self, index_name, assets):
        # type: (str, list[IsccEntry]) -> list[IsccAddResult]
        """
        Add assets to index with chunked batching and progress bar.

        Splits large asset lists into chunks and sends them in batches.
        Shows progress bar for better user experience with large datasets.

        :param index_name: Target index name
        :param assets: List of IsccEntry objects to add
        :return: List of IsccAddResult with status for each asset
        :raises FileNotFoundError: If index doesn't exist
        :raises ValueError: If assets contain invalid ISCC codes
        """

        if not assets:
            return []

        # Single batch if small enough
        if len(assets) <= self.chunk_size:
            return self._add_assets_batch(index_name, assets)

        # Chunked batches with progress bar
        results = []  # type: list[IsccAddResult]
        chunks = [assets[i : i + self.chunk_size] for i in range(0, len(assets), self.chunk_size)]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Adding assets to {index_name}", total=len(chunks))

            for chunk in chunks:
                chunk_results = self._add_assets_batch(index_name, chunk)
                results.extend(chunk_results)
                progress.update(task, advance=1)

        return results

    def _add_assets_batch(self, index_name, assets):
        # type: (str, list[IsccEntry]) -> list[IsccAddResult]
        """
        Add a single batch of assets.

        :param index_name: Target index name
        :param assets: List of IsccEntry objects to add
        :return: List of IsccAddResult with status for each asset
        :raises FileNotFoundError: If index doesn't exist
        :raises ValueError: If assets contain invalid ISCC codes
        """
        from iscc_search.schema import IsccAddResult

        # Convert to dict for JSON serialization
        assets_data = [asset.model_dump(exclude_unset=True) for asset in assets]

        response = self.client.post(f"/indexes/{index_name}/assets", json=assets_data)
        self._handle_response_errors(response)

        data = response.json()
        return [IsccAddResult(**result) for result in data]

    def get_asset(self, index_name, iscc_id):
        # type: (str, str) -> IsccEntry
        """
        Get a specific asset by ISCC-ID.

        :param index_name: Target index name
        :param iscc_id: ISCC-ID of the asset to retrieve
        :return: IsccEntry with all stored metadata
        :raises FileNotFoundError: If index doesn't exist or asset not found
        :raises ValueError: If ISCC-ID format is invalid
        """
        from iscc_search.schema import IsccEntry

        response = self.client.get(f"/indexes/{index_name}/assets/{iscc_id}")
        self._handle_response_errors(response)
        return IsccEntry(**response.json())

    def search_assets(self, index_name, query, limit=100):
        # type: (str, IsccQuery, int) -> IsccSearchResult
        """
        Search for similar assets in index.

        :param index_name: Target index name
        :param query: IsccQuery to search for (either iscc_code or units required)
        :param limit: Maximum number of results to return (default: 100)
        :return: IsccSearchResult with query and list of matches
        :raises FileNotFoundError: If index doesn't exist
        :raises ValueError: If query asset is invalid
        """
        from iscc_search.schema import IsccSearchResult

        # Convert query to dict for JSON serialization
        query_data = query.model_dump(exclude_unset=True)

        response = self.client.post(
            f"/indexes/{index_name}/search",
            json=query_data,
            params={"limit": limit},
        )
        self._handle_response_errors(response)
        return IsccSearchResult(**response.json())

    def close(self):
        # type: () -> None
        """
        Close HTTP client and cleanup resources.

        Idempotent - safe to call multiple times.
        """
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.debug(f"Closed remote index client for {self.url}")
