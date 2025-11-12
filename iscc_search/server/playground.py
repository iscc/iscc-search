"""ISCC-Search API Playground - Interactive testing interface."""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/playground", response_class=HTMLResponse, include_in_schema=False)
def playground():
    # type: () -> HTMLResponse
    """
    Serve interactive playground for testing ISCC-Search API.

    Provides two main features:
    - File upload: Generate ISCC-CODE via demo.iscc.io and search
    - Text search: Search by plain text content
    """
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ISCC-Search Playground</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://releases.transloadit.com/uppy/v5.1.6/uppy.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/json.min.js"></script>
    <style>
        .uppy-DragDrop-container {
            border: 2px dashed #cbd5e1 !important;
            border-radius: 0.5rem !important;
            background: #f8fafc !important;
        }
        .uppy-DragDrop-container:hover {
            border-color: #3b82f6 !important;
        }
    </style>
</head>
<body class="bg-gray-50">
    <div class="container mx-auto px-4 py-8 max-w-7xl">
        <!-- Header -->
        <div class="mb-8">
            <h1 class="text-4xl font-bold text-gray-900 mb-2">ISCC-Search Playground</h1>
            <p class="text-gray-600">Interactive testing interface for similarity search</p>
        </div>

        <!-- Main Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- File Upload Section -->
            <div class="bg-white rounded-lg shadow-md p-6">
                <h2 class="text-2xl font-semibold text-gray-800 mb-4">File Search</h2>
                <p class="text-sm text-gray-600 mb-4">Upload a file to generate an ISCC-CODE and search for similar content</p>

                <div id="uppyDragDrop" class="mb-4"></div>

                <div id="fileResults" class="hidden">
                    <h3 class="text-lg font-semibold text-gray-800 mb-2">Matching Results</h3>
                    <div id="fileResultsContent" class="bg-gray-50 rounded p-4 overflow-auto max-h-96"></div>
                </div>

                <div id="fileError" class="hidden">
                    <div class="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
                        <p class="font-semibold">Error</p>
                        <p id="fileErrorMessage" class="text-sm"></p>
                    </div>
                </div>

                <div id="fileLoading" class="hidden">
                    <div class="flex items-center justify-center py-8">
                        <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                        <span class="ml-3 text-gray-600" id="fileLoadingMessage">Processing...</span>
                    </div>
                </div>
            </div>

            <!-- Text Matching Section -->
            <div class="bg-white rounded-lg shadow-md p-6">
                <h2 class="text-2xl font-semibold text-gray-800 mb-4">Text Matching</h2>
                <p class="text-sm text-gray-600 mb-4">Enter text to find granular/partial matches in indexed content using simprints</p>

                <textarea
                    id="textInput"
                    class="w-full h-32 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                    placeholder="Enter text to search..."
                ></textarea>

                <button
                    id="textSearchBtn"
                    class="mt-4 w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition"
                >
                    Search
                </button>

                <div id="textResults" class="hidden mt-4">
                    <h3 class="text-lg font-semibold text-gray-800 mb-2">Matching Results</h3>
                    <div id="textResultsContent" class="bg-gray-50 rounded p-4 overflow-auto max-h-96"></div>

                    <details class="mt-4 border border-gray-300 rounded-lg">
                        <summary class="cursor-pointer bg-gray-100 hover:bg-gray-200 px-4 py-2 font-semibold text-gray-700 rounded-t-lg">
                            Raw JSON Response
                        </summary>
                        <div class="p-4 bg-gray-900 rounded-b-lg overflow-auto max-h-96">
                            <pre class="text-sm"><code id="textJsonContent" class="language-json"></code></pre>
                        </div>
                    </details>
                </div>

                <div id="textError" class="hidden mt-4">
                    <div class="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
                        <p class="font-semibold">Error</p>
                        <p id="textErrorMessage" class="text-sm"></p>
                    </div>
                </div>

                <div id="textLoading" class="hidden mt-4">
                    <div class="flex items-center justify-center py-8">
                        <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                        <span class="ml-3 text-gray-600">Searching...</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script type="module">
        import { Uppy, DragDrop, XHRUpload } from 'https://releases.transloadit.com/uppy/v5.1.6/uppy.min.mjs';

        // Utility functions
        function showElement(id) {
            document.getElementById(id).classList.remove('hidden');
        }

        function hideElement(id) {
            document.getElementById(id).classList.add('hidden');
        }

        function showError(prefix, message) {
            hideElement(prefix + 'Results');
            hideElement(prefix + 'Loading');
            document.getElementById(prefix + 'ErrorMessage').textContent = message;
            showElement(prefix + 'Error');
        }

        function showLoading(prefix, message = 'Processing...') {
            hideElement(prefix + 'Results');
            hideElement(prefix + 'Error');
            if (message && prefix === 'file') {
                document.getElementById('fileLoadingMessage').textContent = message;
            }
            showElement(prefix + 'Loading');
        }

        function showResults(prefix, content, rawData = null) {
            hideElement(prefix + 'Loading');
            hideElement(prefix + 'Error');
            document.getElementById(prefix + 'ResultsContent').innerHTML = content;

            // Populate raw JSON for text results
            if (prefix === 'text' && rawData) {
                const jsonElement = document.getElementById('textJsonContent');
                jsonElement.textContent = JSON.stringify(rawData, null, 2);
                hljs.highlightElement(jsonElement);
            }

            showElement(prefix + 'Results');
        }

        function formatResults(data) {
            let html = '';

            // Global matches
            if (data.global_matches && data.global_matches.length > 0) {
                html += '<div class="mb-4"><h4 class="font-semibold text-gray-700 mb-2">Global Matches (' +
                        data.global_matches.length + ')</h4>';
                data.global_matches.forEach((match, idx) => {
                    html += '<div class="bg-white border border-gray-200 rounded p-3 mb-2">';
                    html += '<div class="flex justify-between items-start mb-1">';
                    html += '<span class="font-mono text-sm text-blue-600">' + match.iscc_id + '</span>';
                    html += '<span class="text-xs font-semibold text-green-600">Score: ' +
                            (match.score || 0).toFixed(4) + '</span>';
                    html += '</div>';
                    if (match.metadata && match.metadata.name) {
                        html += '<div class="text-sm text-gray-600">' + match.metadata.name + '</div>';
                    }
                    html += '</div>';
                });
                html += '</div>';
            }

            // Chunk matches
            if (data.chunk_matches && data.chunk_matches.length > 0) {
                html += '<div><h4 class="font-semibold text-gray-700 mb-2">Chunk Matches (' +
                        data.chunk_matches.length + ')</h4>';
                data.chunk_matches.forEach((match, idx) => {
                    html += '<div class="bg-white border border-gray-200 rounded p-3 mb-2">';
                    html += '<div class="flex justify-between items-start mb-1">';
                    html += '<span class="font-mono text-xs text-blue-600">' + match.iscc_id + '</span>';
                    html += '<span class="text-xs font-semibold text-green-600">Score: ' +
                            (match.score || 0).toFixed(4) + '</span>';
                    html += '</div>';

                    // Iterate through simprint types and their chunks
                    if (match.types) {
                        for (const [type, typeData] of Object.entries(match.types)) {
                            html += '<div class="text-xs text-gray-700 mt-1 font-semibold">' + type +
                                    ' (' + typeData.matches + '/' + typeData.queried + ' matches)</div>';
                            if (typeData.chunks && typeData.chunks.length > 0) {
                                typeData.chunks.forEach((chunk, cidx) => {
                                    html += '<div class="text-xs text-gray-500 ml-2">└ Offset: ' +
                                            chunk.offset + ' | Size: ' + chunk.size +
                                            ' | Score: ' + (chunk.score || 0).toFixed(4) + '</div>';
                                });
                            }
                        }
                    }
                    html += '</div>';
                });
                html += '</div>';
            }

            if (!data.global_matches?.length && !data.chunk_matches?.length) {
                html = '<div class="text-center text-gray-500 py-8">No matches found</div>';
            }

            return html;
        }

        // File Upload with Uppy
        const uppy = new Uppy({
            debug: false,
            autoProceed: false,
            restrictions: {
                maxNumberOfFiles: 1
            }
        });

        uppy.use(DragDrop, {
            target: '#uppyDragDrop',
            note: 'Drop a file here or click to browse'
        });

        uppy.on('file-added', async (file) => {
            showLoading('file', 'Generating ISCC-CODE...');

            try {
                // Step 1: Generate ISCC-CODE via demo.iscc.io
                const filename = file.name;
                const filenameB64 = btoa(filename);

                const isccResponse = await fetch('https://demo.iscc.io/api/v1/iscc', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/octet-stream',
                        'X-Upload-Filename': filenameB64
                    },
                    body: file.data
                });

                if (!isccResponse.ok) {
                    throw new Error('Failed to generate ISCC-CODE: ' + isccResponse.statusText);
                }

                const isccData = await isccResponse.json();
                const isccCode = isccData.iscc;

                if (!isccCode) {
                    throw new Error('No ISCC code in response');
                }

                // Step 2: Search using ISCC-CODE
                document.getElementById('fileLoadingMessage').textContent =
                    'Searching with ISCC: ' + isccCode + '...';

                const searchResponse = await fetch('/indexes/default/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        iscc_code: isccCode
                    })
                });

                if (!searchResponse.ok) {
                    throw new Error('Search failed: ' + searchResponse.statusText);
                }

                const searchData = await searchResponse.json();
                showResults('file', formatResults(searchData));

            } catch (error) {
                showError('file', error.message);
            } finally {
                uppy.cancelAll();
            }
        });

        // Text Search
        document.getElementById('textSearchBtn').addEventListener('click', async () => {
            const text = document.getElementById('textInput').value.trim();

            if (!text) {
                showError('text', 'Please enter some text to search');
                return;
            }

            showLoading('text');

            try {
                const response = await fetch('/indexes/default/search/text', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        text: text
                    })
                });

                if (!response.ok) {
                    throw new Error('Search failed: ' + response.statusText);
                }

                const data = await response.json();
                showResults('text', formatResults(data), data);

            } catch (error) {
                showError('text', error.message);
            }
        });

        // Allow Enter key in textarea to trigger search
        document.getElementById('textInput').addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                document.getElementById('textSearchBtn').click();
            }
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html)
