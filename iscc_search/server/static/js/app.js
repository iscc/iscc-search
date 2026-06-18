/* ISCC-Search frontend logic, shared by index.html (normal mode) and aggregator.html (aggregator mode).
   All response data is rendered via createElement/textContent — never innerHTML — to stay XSS-safe. */

"use strict";

const MODE = document.body.dataset.mode;
const SPARK_SAMPLES = 15; // assets snapshots kept → up to 14 per-poll delta bars
const state = { indexName: null, lastFetch: null, assetHistory: [], sessionStartAssets: null, relCells: [] };

/* ---------- DOM helpers ---------- */

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
}

function byId(id) {
    return document.getElementById(id);
}

/* ---------- API helpers ---------- */

function apiKey() {
    try {
        return sessionStorage.getItem("iscc-api-key") || "";
    } catch {
        return ""; // storage blocked (cookies disabled / sandboxed iframe)
    }
}

function storeApiKey(value) {
    try {
        if (value) {
            sessionStorage.setItem("iscc-api-key", value);
        } else {
            sessionStorage.removeItem("iscc-api-key");
        }
    } catch {
        // storage blocked - the page still works, the key just won't persist
    }
}

function apiFetch(url, options = {}) {
    const headers = Object.assign({}, options.headers);
    if (apiKey()) headers["X-API-Key"] = apiKey();
    return fetch(url, Object.assign({}, options, { headers }));
}

async function errorDetail(response, fallback) {
    try {
        const data = await response.json();
        const detail = data.detail;
        if (typeof detail === "string") return detail;
        // FastAPI validation errors carry detail as a list of error objects
        if (Array.isArray(detail)) return detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
        return fallback;
    } catch {
        return fallback;
    }
}

function flagAuthProblem() {
    const card = byId("apiKeyCard");
    if (card) {
        card.classList.add("auth-required");
        if (card.tagName === "DETAILS") card.open = true;
    }
}

/* ---------- Results panel ---------- */

function showLoading(message) {
    const section = byId("resultsSection");
    const content = byId("resultsContent");
    clear(content);
    const box = el("div", "notice");
    box.appendChild(el("span", "spinner"));
    box.appendChild(el("span", "", message));
    content.appendChild(box);
    byId("rawJsonDetails").classList.add("hidden");
    section.classList.remove("hidden");
    section.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function showError(message) {
    const content = byId("resultsContent");
    clear(content);
    content.appendChild(el("div", "error-box", message));
    byId("rawJsonDetails").classList.add("hidden");
    byId("resultsSection").classList.remove("hidden");
}

function showResults(fragment, rawData) {
    const content = byId("resultsContent");
    clear(content);
    content.appendChild(fragment);
    const raw = byId("rawJsonContent");
    raw.textContent = JSON.stringify(rawData, null, 2);
    byId("rawJsonDetails").classList.remove("hidden");
    byId("resultsSection").classList.remove("hidden");
}

function gatewayLink(value) {
    if (typeof value !== "string" || !/^https?:\/\//i.test(value)) return null;
    const link = el("a", "", "Gateway ↗");
    link.setAttribute("href", value);
    link.setAttribute("target", "_blank");
    link.setAttribute("rel", "noopener noreferrer");
    return link;
}

function renderMatch(match, detailed) {
    const box = el("div", "match");
    const head = el("div", "match-head");
    head.appendChild(el("span", "mono", match.iscc_id));
    head.appendChild(el("span", "score", "Score: " + (match.score || 0).toFixed(4)));
    box.appendChild(head);
    if (match.metadata) {
        if (match.metadata.name) box.appendChild(el("div", "meta", match.metadata.name));
        const link = gatewayLink(match.metadata.gateway);
        if (link) {
            const meta = el("div", "meta");
            meta.appendChild(link);
            box.appendChild(meta);
        }
    }
    if (detailed && match.types) {
        for (const [type, info] of Object.entries(match.types)) {
            const line = el("div", "types", type + " (" + info.matches + "/" + info.queried + " matches)");
            box.appendChild(line);
            for (const chunk of info.chunks || []) {
                box.appendChild(
                    el(
                        "div",
                        "types",
                        "└ offset " + chunk.offset + " · size " + chunk.size + " · score " + (chunk.score || 0).toFixed(4)
                    )
                );
            }
        }
    }
    return box;
}

function renderMatches(data) {
    const fragment = document.createDocumentFragment();
    const globals = data.global_matches || [];
    const chunks = data.chunk_matches || [];
    if (globals.length) {
        fragment.appendChild(el("h3", "", "Global matches (" + globals.length + ")"));
        for (const match of globals) fragment.appendChild(renderMatch(match, false));
    }
    if (chunks.length) {
        fragment.appendChild(el("h3", "", "Chunk matches (" + chunks.length + ")"));
        for (const match of chunks) fragment.appendChild(renderMatch(match, true));
    }
    if (!globals.length && !chunks.length) {
        fragment.appendChild(el("div", "notice", "No matches found"));
    }
    return fragment;
}

function renderEntry(entry) {
    const fragment = document.createDocumentFragment();
    fragment.appendChild(el("h3", "", "Indexed asset"));
    const box = el("div", "match");
    box.appendChild(el("div", "mono", entry.iscc_id));
    if (entry.iscc_code) box.appendChild(el("div", "meta mono", entry.iscc_code));
    for (const unit of entry.units || []) box.appendChild(el("div", "types mono", unit));
    if (entry.metadata) {
        for (const [key, value] of Object.entries(entry.metadata)) {
            const link = key === "gateway" ? gatewayLink(value) : null;
            if (link) {
                const meta = el("div", "meta");
                meta.appendChild(link);
                box.appendChild(meta);
            } else {
                box.appendChild(el("div", "meta", key + ": " + String(value)));
            }
        }
    }
    fragment.appendChild(box);
    return fragment;
}

/* ---------- Lookup & search ---------- */

async function postSearch(path, payload) {
    const response = await apiFetch("/indexes/" + state.indexName + path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    if (response.status === 401) flagAuthProblem();
    if (!response.ok) throw new Error("Search failed: " + (await errorDetail(response, response.statusText)));
    const data = await response.json();
    showResults(renderMatches(data), data);
}

async function searchByCode(isccCode) {
    await postSearch("/search", { iscc_code: isccCode });
}

async function runLookup(value) {
    if (!state.indexName) {
        showError("No index available to search.");
        return;
    }
    showLoading("Searching " + value + " …");
    try {
        const response = await apiFetch("/indexes/" + state.indexName + "/assets/" + encodeURIComponent(value));
        if (response.ok) {
            const entry = await response.json();
            showResults(renderEntry(entry), entry);
            return;
        }
        if (response.status === 401) {
            flagAuthProblem();
            throw new Error("Unauthorized: set your API key below.");
        }
        if (response.status >= 500) {
            throw new Error("Lookup failed: " + (await errorDetail(response, response.statusText)));
        }
        if (response.status === 404 && /^ISCC:M[A-Z2-7]+$/i.test(value)) {
            // MainType ISCC-ID (prefix M): similarity search over an ID is futile, surface the 404
            throw new Error("No asset with this ISCC-ID in the index.");
        }
        await searchByCode(value);
    } catch (error) {
        showError(error.message);
    }
}

/* ---------- Size & time formatting ---------- */

function fmtSizeParts(megabytes) {
    if (megabytes >= 1024) return { value: (megabytes / 1024).toFixed(megabytes >= 10240 ? 0 : 1), unit: "GB" };
    if (megabytes < 1) return { value: "< 1", unit: "MB" };
    return { value: megabytes.toLocaleString(), unit: "MB" };
}

function fmtSize(megabytes) {
    const parts = fmtSizeParts(megabytes);
    return parts.value + " " + parts.unit;
}

function sizesText(sizes) {
    if (!sizes) return "";
    return Object.entries(sizes)
        .map(([component, megabytes]) => component + " " + fmtSize(megabytes))
        .join(" · ");
}

function agoText(milliseconds) {
    const seconds = Math.max(0, Math.round((Date.now() - milliseconds) / 1000));
    if (seconds < 5) return "just now";
    if (seconds < 90) return seconds + "s ago";
    if (seconds < 5400) return Math.round(seconds / 60) + "m ago";
    if (seconds < 129600) return Math.round(seconds / 3600) + "h ago";
    return Math.round(seconds / 86400) + "d ago";
}

function relTime(isoString) {
    return agoText(new Date(isoString).getTime());
}

/* ---------- Aggregator: headline stats ---------- */

function renderSparkline(assets) {
    if (state.sessionStartAssets === null) state.sessionStartAssets = assets;
    state.assetHistory.push(assets);
    if (state.assetHistory.length > SPARK_SAMPLES) state.assetHistory.shift();
    const deltas = [];
    for (let i = 1; i < state.assetHistory.length; i++) {
        deltas.push(Math.max(0, state.assetHistory[i] - state.assetHistory[i - 1]));
    }
    const spark = byId("sparkline");
    clear(spark);
    const max = Math.max.apply(null, deltas.concat(1));
    if (deltas.length >= 2 && max > 0) {
        deltas.forEach((delta, i) => {
            const bar = el("span", "sparkline-bar" + (i === deltas.length - 1 ? " is-latest" : ""));
            bar.style.height = Math.round((delta / max) * 100) + "%";
            spark.appendChild(bar);
        });
    }
    const sessionDelta = assets - state.sessionStartAssets;
    const sub = byId("statAssetsDelta");
    clear(sub);
    if (sessionDelta > 0) {
        sub.className = "stat-card-sub is-positive";
        sub.appendChild(el("span", "pulse-dot"));
        sub.appendChild(el("span", "", "+" + sessionDelta.toLocaleString() + " this session"));
    } else {
        sub.className = "stat-card-sub";
    }
}

function renderHeadlineStats(status) {
    const index = status.index;
    if (index) {
        byId("statAssets").textContent = (index.assets || 0).toLocaleString();
        renderSparkline(index.assets || 0);
        const size = fmtSizeParts(index.size || 0);
        const value = byId("statSize");
        clear(value);
        value.appendChild(document.createTextNode(size.value + " "));
        value.appendChild(el("span", "unit", size.unit));
        const detail = sizesText(index.sizes);
        byId("statSizeDetail").textContent = detail ? "across all hubs · " + detail : "across all hubs";
    }
    renderHubStat(status.hubs || []);
}

function renderHubStat(hubs) {
    const total = hubs.length;
    const okCount = hubs.filter((hub) => hub.ok).length;
    const errCount = total - okCount;
    byId("statHubs").textContent = String(okCount);
    const sub = byId("statHubsDetail");
    clear(sub);
    if (errCount > 0) {
        sub.className = "stat-card-sub is-error";
        sub.appendChild(el("span", "dot warn"));
        sub.appendChild(el("span", "", errCount + " reporting error" + (errCount === 1 ? "" : "s")));
    } else if (total > 0) {
        sub.className = "stat-card-sub is-positive";
        sub.appendChild(el("span", "dot ok"));
        sub.appendChild(el("span", "", "all healthy"));
    } else {
        sub.className = "stat-card-sub";
        sub.textContent = "no hubs polled yet";
    }
}

/* ---------- Aggregator: hub table ---------- */

function renderHubRow(hub) {
    const row = el("tr", hub.ok ? "hub-row" : "hub-row row-error");
    row.appendChild(el("td", "hub-name", "#" + String(hub.hub_id).padStart(4, "0")));
    row.appendChild(el("td", "hub-endpoint", hub.url.replace(/^https?:\/\//, "")));
    row.appendChild(el("td", "num", (hub.cursor || 0).toLocaleString()));
    const pollCell = el("td", "num", hub.last_poll ? relTime(hub.last_poll) : "never");
    if (hub.last_poll) state.relCells.push({ node: pollCell, iso: hub.last_poll });
    row.appendChild(pollCell);
    const statusTd = el("td");
    const status = el("span", "status-cell " + (hub.ok ? "ok" : "err"));
    status.appendChild(el("span", "dot " + (hub.ok ? "ok" : "err")));
    status.appendChild(el("span", "", hub.ok ? "healthy" : hub.error || "error"));
    statusTd.appendChild(status);
    row.appendChild(statusTd);
    return row;
}

function renderHubTable(hubs) {
    const body = byId("hubsTableBody");
    clear(body);
    state.relCells = [];
    if (!hubs.length) {
        const row = el("tr");
        const cell = el("td", "notice", "No hubs polled yet");
        cell.colSpan = 5;
        row.appendChild(cell);
        body.appendChild(row);
        return;
    }
    for (const hub of hubs) body.appendChild(renderHubRow(hub));
}

function renderStatusPanel(status) {
    renderHeadlineStats(status);
    renderHubTable(status.hubs || []);
}

/* ---------- Aggregator: live "updated Ns ago" + relative hub times ---------- */

function refreshRelativeTimes() {
    if (state.lastFetch === null) return;
    const liveAge = byId("liveAge");
    liveAge.classList.remove("hidden");
    byId("liveAgeText").textContent = "updated " + agoText(state.lastFetch);
    for (const cell of state.relCells) cell.node.textContent = relTime(cell.iso);
}

function applyNetworkBadge(network) {
    const badge = byId("modeBadge");
    badge.className = "chip net-" + (network === "mainnet" ? "mainnet" : "testnet");
    clear(badge);
    badge.appendChild(el("span", "chip-dot"));
    badge.appendChild(el("span", "", "IDP AGGREGATOR · " + network.toUpperCase()));
    byId("networkName").textContent = network.toUpperCase();
}

/* ---------- Status (shared; drives the aggregator panel) ---------- */

async function fetchStatus() {
    const response = await fetch("/status");
    if (!response.ok) return;
    const status = await response.json();
    byId("versionLabel").textContent = "v" + status.version;
    if (MODE === "aggregator") {
        applyNetworkBadge((status.network || "testnet").toLowerCase());
        state.indexName = status.index_name;
        state.lastFetch = Date.now();
        renderStatusPanel(status);
        refreshRelativeTimes();
    }
}

async function pollStatus() {
    if (!document.hidden) {
        try {
            await fetchStatus();
        } catch {
            // transient network failure (server restart, proxy blip) - retry next tick
        }
    }
    setTimeout(pollStatus, 10000);
}

/* ---------- Operator-detail disclosure ---------- */

function initDisclosure() {
    const toggle = byId("networkToggle");
    if (!toggle) return;
    const panel = byId("operatorDetail");
    const hint = byId("networkToggleHint");
    let open = true;
    try {
        const saved = localStorage.getItem("iscc-operator-detail");
        if (saved !== null) open = saved === "1";
    } catch {
        // localStorage blocked - default to open, just don't persist the choice
    }
    const apply = () => {
        toggle.setAttribute("aria-expanded", String(open));
        panel.classList.toggle("hidden", !open);
        hint.textContent = "operator detail — " + (open ? "hide" : "show");
    };
    apply();
    toggle.addEventListener("click", () => {
        open = !open;
        try {
            localStorage.setItem("iscc-operator-detail", open ? "1" : "0");
        } catch {
            // persistence is best-effort; the toggle still works for this session
        }
        apply();
    });
}

/* ---------- Normal mode: index table ---------- */

function selectIndex(name) {
    state.indexName = name;
    for (const row of byId("indexesTableBody").children) {
        const selected = row.dataset.name === name;
        row.classList.toggle("selected", selected);
        if (row.dataset.name) row.setAttribute("aria-pressed", String(selected));
    }
}

async function loadIndexes() {
    const body = byId("indexesTableBody");
    clear(body);
    try {
        const response = await apiFetch("/indexes");
        if (response.status === 401) {
            flagAuthProblem();
            throw new Error("Unauthorized: set your API key below.");
        }
        if (!response.ok) throw new Error(await errorDetail(response, response.statusText));
        const indexes = await response.json();
        for (const index of indexes) {
            const row = el("tr", "selectable");
            row.dataset.name = index.name;
            row.tabIndex = 0;
            row.setAttribute("role", "button");
            row.appendChild(el("td", "mono", index.name));
            row.appendChild(el("td", "num", (index.assets || 0).toLocaleString()));
            const sizeCell = el("td", "num", fmtSize(index.size || 0));
            sizeCell.title = sizesText(index.sizes);
            row.appendChild(sizeCell);
            row.addEventListener("click", () => selectIndex(index.name));
            row.addEventListener("keydown", (event) => {
                if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    selectIndex(index.name);
                }
            });
            body.appendChild(row);
        }
        if (indexes.length) {
            selectIndex(indexes[0].name);
        } else {
            const row = el("tr");
            const cell = el("td", "notice", "No indexes yet — create one via the API (see /docs)");
            cell.colSpan = 3;
            row.appendChild(cell);
            body.appendChild(row);
        }
    } catch (error) {
        const row = el("tr");
        const cell = el("td", "notice", error.message);
        cell.colSpan = 3;
        row.appendChild(cell);
        body.appendChild(row);
    }
}

/* ---------- API key ---------- */

function initApiKey() {
    const input = byId("apiKeyInput");
    if (!input) return;
    input.value = apiKey();
    input.addEventListener("input", () => {
        storeApiKey(input.value);
        const card = byId("apiKeyCard");
        if (card) card.classList.remove("auth-required");
    });
    // On commit (blur/Enter) retry the index list - a 401 at page load left it empty
    input.addEventListener("change", () => {
        if (MODE !== "aggregator") loadIndexes();
    });
}

/* ---------- Init ---------- */

function init() {
    byId("lookupForm").addEventListener("submit", (event) => {
        event.preventDefault();
        const value = byId("lookupInput").value.trim();
        if (value) runLookup(value);
    });
    initApiKey();
    if (MODE === "aggregator") {
        initDisclosure();
        pollStatus();
        setInterval(() => {
            if (!document.hidden) refreshRelativeTimes();
        }, 1000);
    } else {
        fetchStatus().catch(() => {});
        loadIndexes();
    }
}

init();
