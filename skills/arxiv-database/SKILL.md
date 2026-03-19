---
name: arxiv-database
description: Search and retrieve papers from arXiv for literature tracking and related work. Use when searching for recent papers in gauge-equivariant networks, information geometry, variational inference, or attention mechanism theory. For systematic literature synthesis use literature-review.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# arXiv Database — Literature Search and Retrieval

## Overview

This skill provides structured guidance for searching, retrieving, and organizing papers from arXiv. It covers API usage, category navigation, and strategies for building comprehensive literature searches relevant to the Gauge-Transformer project.

## When to Use This Skill

Use this skill when:
- Searching for recent papers on gauge-equivariant neural networks
- Tracking new submissions in information geometry and deep learning
- Finding related work on variational free energy in ML
- Building bibliography for the GL(K) attention manuscript
- Monitoring arXiv categories for relevant new work
- Retrieving paper metadata (authors, abstracts, citations)

## Relevant arXiv Categories

For the Gauge-Transformer project, prioritize:

| Category | Description | Relevance |
|----------|-------------|-----------|
| `cs.LG` | Machine Learning | Core — attention mechanisms, transformers |
| `cs.CL` | Computation and Language | NLP, language modeling |
| `stat.ML` | Machine Learning (Statistics) | Variational inference, Bayesian methods |
| `hep-th` | High Energy Physics — Theory | Gauge theory, renormalization group |
| `math-ph` | Mathematical Physics | Lie groups, fiber bundles |
| `cs.AI` | Artificial Intelligence | Active inference, free energy principle |
| `math.DG` | Differential Geometry | SPD manifolds, connections |

---

## Core Capabilities

### 1. arXiv API Search

```python
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time

ARXIV_API = 'http://export.arxiv.org/api/query?'

def search_arxiv(query, categories=None, max_results=20, sort_by='submittedDate',
                 sort_order='descending', start=0):
    """Search arXiv API with category filtering.

    Args:
        query: search string (supports AND, OR, ANDNOT, field prefixes)
        categories: list of arXiv categories to filter (e.g., ['cs.LG', 'stat.ML'])
        max_results: number of results to return (max 100 per request)
        sort_by: 'submittedDate', 'lastUpdatedDate', or 'relevance'
    """
    # Build query with category filter
    search_query = f'all:{query}'
    if categories:
        cat_filter = ' OR '.join(f'cat:{c}' for c in categories)
        search_query = f'({search_query}) AND ({cat_filter})'

    params = {
        'search_query': search_query,
        'start': start,
        'max_results': max_results,
        'sortBy': sort_by,
        'sortOrder': sort_order
    }

    url = ARXIV_API + urllib.parse.urlencode(params)
    response = urllib.request.urlopen(url)
    root = ET.fromstring(response.read())

    ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}

    papers = []
    for entry in root.findall('atom:entry', ns):
        paper = {
            'id': entry.find('atom:id', ns).text.split('/')[-1],
            'title': entry.find('atom:title', ns).text.strip().replace('\n', ' '),
            'authors': [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)],
            'abstract': entry.find('atom:summary', ns).text.strip(),
            'published': entry.find('atom:published', ns).text[:10],
            'updated': entry.find('atom:updated', ns).text[:10],
            'categories': [c.get('term') for c in entry.findall('atom:category', ns)],
            'pdf_url': f"https://arxiv.org/pdf/{entry.find('atom:id', ns).text.split('/')[-1]}"
        }
        papers.append(paper)

    return papers
```

### 2. Targeted Search Queries for Gauge-Transformer

```python
# Key search queries for this project
GAUGE_TRANSFORMER_QUERIES = {
    'gauge_equivariant': {
        'query': 'gauge equivariant neural network',
        'categories': ['cs.LG', 'hep-th', 'math-ph'],
    },
    'information_geometry_dl': {
        'query': 'information geometry deep learning',
        'categories': ['cs.LG', 'stat.ML'],
    },
    'variational_free_energy_ml': {
        'query': 'variational free energy minimization machine learning',
        'categories': ['cs.LG', 'cs.AI', 'stat.ML'],
    },
    'active_inference': {
        'query': 'active inference free energy principle',
        'categories': ['cs.AI', 'cs.LG'],
    },
    'attention_theory': {
        'query': 'attention mechanism theory transformer',
        'categories': ['cs.LG', 'cs.CL'],
    },
    'spd_manifold': {
        'query': 'symmetric positive definite manifold neural',
        'categories': ['cs.LG', 'math.DG'],
    },
    'renormalization_group_ml': {
        'query': 'renormalization group machine learning',
        'categories': ['cs.LG', 'hep-th', 'stat.ML'],
    },
    'kl_divergence_attention': {
        'query': 'KL divergence attention mechanism',
        'categories': ['cs.LG', 'cs.CL', 'stat.ML'],
    },
}

def run_all_searches(max_per_query=10):
    """Run all project-relevant searches."""
    all_results = {}
    for name, params in GAUGE_TRANSFORMER_QUERIES.items():
        results = search_arxiv(params['query'], params['categories'], max_results=max_per_query)
        all_results[name] = results
        time.sleep(3)  # Respect arXiv rate limits
    return all_results
```

### 3. Paper Formatting for Related Work

```python
def format_for_bibtex(paper):
    """Generate BibTeX entry from arXiv paper metadata."""
    first_author_last = paper['authors'][0].split()[-1].lower()
    year = paper['published'][:4]
    key = f"{first_author_last}{year}_{paper['id'].replace('.', '_')}"

    authors_bibtex = ' and '.join(paper['authors'])

    return f"""@article{{{key},
    title={{{paper['title']}}},
    author={{{authors_bibtex}}},
    journal={{arXiv preprint arXiv:{paper['id']}}},
    year={{{year}}},
    url={{https://arxiv.org/abs/{paper['id']}}},
}}"""


def format_for_related_work(paper, style='brief'):
    """Format a paper reference for the related work section."""
    first_author = paper['authors'][0].split()[-1]
    year = paper['published'][:4]
    n_authors = len(paper['authors'])

    if n_authors == 1:
        cite = f"{first_author} ({year})"
    elif n_authors == 2:
        second_author = paper['authors'][1].split()[-1]
        cite = f"{first_author} and {second_author} ({year})"
    else:
        cite = f"{first_author} et al. ({year})"

    if style == 'brief':
        return f"{cite}: {paper['title']}"
    else:
        return f"{cite}: {paper['title']}\n  Abstract: {paper['abstract'][:200]}..."
```

### 4. Monitoring New Submissions

```python
def check_new_papers(days_back=7):
    """Check for new papers in the last N days across relevant categories."""
    from datetime import datetime, timedelta

    recent = []
    for name, params in GAUGE_TRANSFORMER_QUERIES.items():
        results = search_arxiv(
            params['query'], params['categories'],
            max_results=5, sort_by='submittedDate'
        )
        cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        new_papers = [p for p in results if p['published'] >= cutoff]
        if new_papers:
            recent.append({'topic': name, 'papers': new_papers})
        time.sleep(3)

    return recent
```

---

## Search Query Syntax

arXiv API supports field-specific searches:

| Prefix | Field | Example |
|--------|-------|---------|
| `ti:` | Title | `ti:gauge equivariant` |
| `au:` | Author | `au:cohen` |
| `abs:` | Abstract | `abs:variational free energy` |
| `all:` | All fields | `all:attention mechanism` |
| `cat:` | Category | `cat:cs.LG` |

Operators: `AND`, `OR`, `ANDNOT`

Example: `ti:gauge AND ti:equivariant AND cat:cs.LG`

---

## Rate Limits and Best Practices

1. **Respect rate limits**: Wait 3 seconds between API requests
2. **Max 100 results per query**: Use pagination (`start` parameter) for more
3. **Cache results locally** to avoid redundant API calls
4. **Use specific field prefixes** (`ti:`, `abs:`) for precise searches
5. **Combine categories** with OR for cross-disciplinary searches
6. **Sort by `submittedDate`** for monitoring new work
7. **Sort by `relevance`** for comprehensive literature searches

---

## Key Authors to Track

For the Gauge-Transformer project, monitor papers by:
- **Taco Cohen** — Gauge equivariant CNNs, geometric deep learning
- **Maurice Weiler** — E(n)-equivariant networks, coordinate-free methods
- **Karl Friston** — Free energy principle, active inference
- **Shun-ichi Amari** — Information geometry foundations
- **Max Welling** — Variational inference, geometric deep learning

---

## Dependencies

```
# No additional dependencies — uses Python standard library (urllib, xml)
```
