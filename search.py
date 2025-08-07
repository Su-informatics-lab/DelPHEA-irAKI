"""
Literature Search Module for DelPHEA-irAKI
==========================================

This module provides functionality to search PubMed and bioRxiv for relevant
literature to support expert clinical reasoning in irAKI classification.

Features:
- PubMed API integration via Entrez
- bioRxiv API integration
- Relevance scoring and ranking
- Citation formatting for clinical reasoning
- Rate limiting and error handling
- Specialty-specific query optimization

Usage:
    from modules.literature_search import LiteratureSearcher

    searcher = LiteratureSearcher()
    results = await searcher.search_iraki_literature(
        query="immune checkpoint inhibitor acute kidney injury",
        max_results=5,
        specialty="nephrology"
    )
"""

import asyncio
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class LiteratureResult:
    """Structure for literature search results"""

    title: str
    authors: List[str]
    journal: str
    publication_date: str
    pmid: Optional[str]
    doi: Optional[str]
    abstract: str
    relevance_score: float
    key_sentences: List[str]
    source: str  # 'pubmed' or 'biorxiv'
    url: str
    citation: str


class LiteratureSearcher:
    """Search and retrieve relevant medical literature for irAKI assessment"""

    def __init__(self, email: str = "delphea@example.com"):
        self.email = email
        self.session = None
        self.last_request_time = 0
        self.rate_limit_delay = 0.34  # ~3 requests per second for PubMed

        # Specialty-specific keywords for query enhancement
        self.specialty_keywords = {
            "medical_oncology": [
                "immune checkpoint inhibitor",
                "immunotherapy",
                "cancer immunotherapy",
                "nivolumab",
                "pembrolizumab",
                "ipilimumab",
                "atezolizumab",
                "durvalumab",
                "immune-related adverse events",
                "irAE",
                "oncology toxicity",
            ],
            "nephrology": [
                "acute kidney injury",
                "acute tubular necrosis",
                "tubulointerstitial nephritis",
                "drug-induced nephrotoxicity",
                "AKI",
                "renal function",
                "creatinine",
                "proteinuria",
                "hematuria",
                "kidney biopsy",
            ],
            "renal_pathology": [
                "kidney biopsy",
                "renal pathology",
                "tubulointerstitial nephritis",
                "acute tubular necrosis",
                "immune infiltrate",
                "pathologic findings",
                "histology",
                "electron microscopy",
                "immunofluorescence",
            ],
            "clinical_pharmacy": [
                "drug interactions",
                "pharmacokinetics",
                "medication safety",
                "adverse drug reactions",
                "drug-induced",
                "pharmacovigilance",
                "concomitant medications",
                "dose adjustment",
            ],
            "clinical_informatics": [
                "clinical decision support",
                "predictive modeling",
                "biomarkers",
                "machine learning",
                "clinical prediction",
                "risk stratification",
                "electronic health records",
                "data mining",
            ],
            "critical_care": [
                "intensive care",
                "critical illness",
                "sepsis",
                "hemodynamic",
                "multi-organ failure",
                "renal replacement therapy",
                "CRRT",
                "acute care",
                "ICU",
            ],
            "rheumatology": [
                "autoimmune",
                "lupus nephritis",
                "vasculitis",
                "immune complex",
                "complement",
                "autoantibodies",
                "immunosuppression",
                "systemic autoimmune disease",
            ],
        }

        # irAKI-specific query templates
        self.iraki_queries = {
            "mechanism": "immune checkpoint inhibitor acute kidney injury mechanism",
            "diagnosis": "immune related acute kidney injury diagnosis biomarker",
            "pathology": "checkpoint inhibitor nephritis biopsy pathology",
            "management": "immune checkpoint inhibitor AKI treatment corticosteroid",
            "rechallenge": "checkpoint inhibitor rechallenge after nephrotoxicity",
            "biomarkers": "biomarkers immune related acute kidney injury",
            "temporal": "temporal relationship immunotherapy acute kidney injury",
        }

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def _rate_limit(self):
        """Enforce rate limiting for API requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)

        self.last_request_time = time.time()

    def _enhance_query_by_specialty(self, base_query: str, specialty: str) -> str:
        """Enhance search query with specialty-specific keywords"""
        if specialty in self.specialty_keywords:
            keywords = self.specialty_keywords[specialty][:3]  # Top 3 keywords
            enhanced_query = f"{base_query} AND ({' OR '.join(keywords)})"
            return enhanced_query
        return base_query

    async def search_pubmed(
        self, query: str, max_results: int = 10, recent_years: int = 5
    ) -> List[LiteratureResult]:
        """Search PubMed for relevant literature"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            # Step 1: Search for PMIDs
            await self._rate_limit()

            # Construct date filter for recent papers
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * recent_years)
            date_filter = f"{start_date.year}/{start_date.month:02d}/{start_date.day:02d}:{end_date.year}/{end_date.month:02d}/{end_date.day:02d}[dp]"

            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": f"({query}) AND {date_filter}",
                "retmax": max_results,
                "email": self.email,
                "tool": "delphea_iraki",
            }

            async with self.session.get(search_url, params=search_params) as response:
                if response.status != 200:
                    logger.error(f"PubMed search failed: {response.status}")
                    return []

                search_content = await response.text()
                search_root = ET.fromstring(search_content)

                pmids = [id_elem.text for id_elem in search_root.findall(".//Id")]

                if not pmids:
                    logger.warning(f"No PubMed results for query: {query}")
                    return []

            # Step 2: Fetch detailed information for PMIDs
            await self._rate_limit()

            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
                "email": self.email,
                "tool": "delphea_iraki",
            }

            async with self.session.get(fetch_url, params=fetch_params) as response:
                if response.status != 200:
                    logger.error(f"PubMed fetch failed: {response.status}")
                    return []

                fetch_content = await response.text()
                fetch_root = ET.fromstring(fetch_content)

                results = []
                for article in fetch_root.findall(".//PubmedArticle"):
                    result = self._parse_pubmed_article(article, query)
                    if result:
                        results.append(result)

                return results

        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return []

    def _parse_pubmed_article(
        self, article_elem, original_query: str
    ) -> Optional[LiteratureResult]:
        """Parse PubMed article XML into LiteratureResult"""
        try:
            # Extract basic information
            article = article_elem.find(".//Article")
            if article is None:
                return None

            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else "No title"

            # Extract authors
            authors = []
            author_list = article.find(".//AuthorList")
            if author_list is not None:
                for author in author_list.findall(".//Author"):
                    last_name = author.find(".//LastName")
                    first_name = author.find(".//ForeName")
                    if last_name is not None:
                        author_name = last_name.text
                        if first_name is not None:
                            author_name = f"{first_name.text} {author_name}"
                        authors.append(author_name)

            # Extract journal
            journal_elem = article.find(".//Journal/Title")
            journal = (
                journal_elem.text if journal_elem is not None else "Unknown journal"
            )

            # Extract publication date
            pub_date = article.find(".//PubDate")
            pub_date_str = "Unknown date"
            if pub_date is not None:
                year = pub_date.find(".//Year")
                month = pub_date.find(".//Month")
                if year is not None:
                    pub_date_str = year.text
                    if month is not None:
                        pub_date_str = f"{month.text} {pub_date_str}"

            # Extract PMID
            pmid_elem = article_elem.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else None

            # Extract DOI
            doi = None
            article_ids = article_elem.findall(".//ArticleId")
            for aid in article_ids:
                if aid.get("IdType") == "doi":
                    doi = aid.text
                    break

            # Extract abstract
            abstract_elem = article.find(".//Abstract/AbstractText")
            abstract = (
                abstract_elem.text
                if abstract_elem is not None
                else "No abstract available"
            )

            # Calculate relevance score and extract key sentences
            relevance_score = self._calculate_relevance_score(
                title, abstract, original_query
            )
            key_sentences = self._extract_key_sentences(abstract, original_query)

            # Create citation
            first_author = authors[0] if authors else "Unknown author"
            citation = f"{first_author} et al. {journal}. {pub_date_str}."
            if pmid:
                citation += f" PMID: {pmid}."

            # Create URL
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

            return LiteratureResult(
                title=title,
                authors=authors,
                journal=journal,
                publication_date=pub_date_str,
                pmid=pmid,
                doi=doi,
                abstract=abstract,
                relevance_score=relevance_score,
                key_sentences=key_sentences,
                source="pubmed",
                url=url,
                citation=citation,
            )

        except Exception as e:
            logger.error(f"Error parsing PubMed article: {e}")
            return None

    async def search_biorxiv(
        self, query: str, max_results: int = 5
    ) -> List[LiteratureResult]:
        """Search bioRxiv for relevant preprints"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            await self._rate_limit()

            # bioRxiv API endpoint
            search_url = "https://api.biorxiv.org/details/biorxiv"

            # Get recent preprints (last 2 years for preprints)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)

            date_from = start_date.strftime("%Y-%m-%d")
            date_to = end_date.strftime("%Y-%m-%d")

            search_params = {
                "server": "biorxiv",
                "format": "json",
                "cursor": "0",
                "count": max_results * 3,  # Get more to filter
            }

            # Construct URL for date range search
            url = f"{search_url}/{date_from}/{date_to}"

            async with self.session.get(url, params=search_params) as response:
                if response.status != 200:
                    logger.error(f"bioRxiv search failed: {response.status}")
                    return []

                data = await response.json()

                if "collection" not in data:
                    logger.warning("No bioRxiv results found")
                    return []

                results = []
                for article in data["collection"]:
                    result = self._parse_biorxiv_article(article, query)
                    if result and result.relevance_score > 0.3:  # Filter by relevance
                        results.append(result)

                # Sort by relevance and return top results
                results.sort(key=lambda x: x.relevance_score, reverse=True)
                return results[:max_results]

        except Exception as e:
            logger.error(f"bioRxiv search error: {e}")
            return []

    def _parse_biorxiv_article(
        self, article_data: Dict, original_query: str
    ) -> Optional[LiteratureResult]:
        """Parse bioRxiv article data into LiteratureResult"""
        try:
            title = article_data.get("title", "No title")
            abstract = article_data.get("abstract", "No abstract available")

            # Check relevance before processing
            relevance_score = self._calculate_relevance_score(
                title, abstract, original_query
            )
            if relevance_score < 0.3:  # Skip irrelevant articles
                return None

            # Extract authors
            authors_str = article_data.get("authors", "")
            authors = [name.strip() for name in authors_str.split(",") if name.strip()]

            # Extract other fields
            doi = article_data.get("doi", "")
            pub_date = article_data.get("date", "Unknown date")

            # Extract key sentences
            key_sentences = self._extract_key_sentences(abstract, original_query)

            # Create citation
            first_author = authors[0] if authors else "Unknown author"
            citation = f"{first_author} et al. bioRxiv. {pub_date}. doi: {doi}"

            # Create URL
            url = f"https://doi.org/{doi}" if doi else ""

            return LiteratureResult(
                title=title,
                authors=authors,
                journal="bioRxiv",
                publication_date=pub_date,
                pmid=None,
                doi=doi,
                abstract=abstract,
                relevance_score=relevance_score,
                key_sentences=key_sentences,
                source="biorxiv",
                url=url,
                citation=citation,
            )

        except Exception as e:
            logger.error(f"Error parsing bioRxiv article: {e}")
            return None

    def _calculate_relevance_score(
        self, title: str, abstract: str, query: str
    ) -> float:
        """Calculate relevance score based on keyword matching"""
        # Define high-value irAKI keywords
        iraki_keywords = {
            "immune checkpoint inhibitor": 3.0,
            "immune related adverse": 2.5,
            "acute kidney injury": 2.5,
            "tubulointerstitial nephritis": 2.0,
            "immunotherapy": 2.0,
            "nephrotoxicity": 2.0,
            "nivolumab": 1.5,
            "pembrolizumab": 1.5,
            "ipilimumab": 1.5,
            "irAE": 2.0,
            "renal toxicity": 1.5,
            "kidney biopsy": 1.5,
            "creatinine": 1.0,
            "proteinuria": 1.0,
        }

        text = f"{title} {abstract}".lower()
        score = 0.0

        for keyword, weight in iraki_keywords.items():
            if keyword.lower() in text:
                score += weight

        # Normalize score
        max_possible_score = sum(iraki_keywords.values())
        normalized_score = min(score / max_possible_score, 1.0)

        return normalized_score

    def _extract_key_sentences(
        self, abstract: str, query: str, max_sentences: int = 3
    ) -> List[str]:
        """Extract key sentences from abstract relevant to the query"""
        if not abstract or abstract == "No abstract available":
            return []

        # Split into sentences
        sentences = re.split(r"[.!?]+", abstract)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Score sentences based on keyword relevance
        scored_sentences = []
        query_words = set(query.lower().split())

        iraki_keywords = [
            "immune",
            "checkpoint",
            "inhibitor",
            "acute",
            "kidney",
            "injury",
            "nephritis",
            "toxicity",
            "renal",
            "immunotherapy",
            "adverse",
            "creatinine",
            "biopsy",
            "treatment",
            "corticosteroid",
        ]

        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = 0

            # Score based on query words
            for word in query_words:
                if word in sentence_lower:
                    score += 1

            # Score based on irAKI keywords
            for keyword in iraki_keywords:
                if keyword in sentence_lower:
                    score += 0.5

            if score > 0:
                scored_sentences.append((sentence, score))

        # Sort by score and return top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        return [sent[0] for sent in scored_sentences[:max_sentences]]

    async def search_iraki_literature(
        self, query: str, specialty: str, max_results: int = 8
    ) -> List[LiteratureResult]:
        """
        Main search function for irAKI-relevant literature

        Args:
            query: Base search query
            specialty: Expert specialty for query enhancement
            max_results: Maximum number of results to return

        Returns:
            List of LiteratureResult objects, sorted by relevance
        """
        # Enhance query based on specialty
        enhanced_query = self._enhance_query_by_specialty(query, specialty)

        logger.info(f"Searching literature for {specialty}: {enhanced_query}")

        # Search both PubMed and bioRxiv
        pubmed_results = await self.search_pubmed(
            enhanced_query, max_results=max_results, recent_years=5
        )

        biorxiv_results = await self.search_biorxiv(
            enhanced_query, max_results=max(2, max_results // 4)  # Fewer preprints
        )

        # Combine and sort results
        all_results = pubmed_results + biorxiv_results
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Return top results
        final_results = all_results[:max_results]

        logger.info(f"Found {len(final_results)} relevant literature results")
        return final_results

    async def search_by_iraki_topic(
        self, topic: str, specialty: str, max_results: int = 5
    ) -> List[LiteratureResult]:
        """Search for literature by specific irAKI topic"""
        if topic in self.iraki_queries:
            base_query = self.iraki_queries[topic]
        else:
            base_query = f"immune checkpoint inhibitor acute kidney injury {topic}"

        return await self.search_iraki_literature(base_query, specialty, max_results)

    def format_literature_citations(self, results: List[LiteratureResult]) -> str:
        """Format literature results for clinical reasoning"""
        if not results:
            return "No relevant literature found."

        formatted = "RELEVANT LITERATURE:\n\n"

        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result.citation}\n"
            formatted += f"   Title: {result.title}\n"

            if result.key_sentences:
                formatted += f"   Key findings: {' '.join(result.key_sentences[:2])}\n"

            formatted += f"   Relevance: {result.relevance_score:.2f} | Source: {result.source}\n"
            if result.url:
                formatted += f"   URL: {result.url}\n"
            formatted += "\n"

        return formatted


# Async factory function for easy import
async def create_literature_searcher(
    email: str = "delphea@example.com",
) -> LiteratureSearcher:
    """Create and initialize a LiteratureSearcher instance"""
    searcher = LiteratureSearcher(email)
    await searcher.__aenter__()
    return searcher


# Example usage and testing
if __name__ == "__main__":

    async def test_literature_search():
        async with LiteratureSearcher() as searcher:
            # Test irAKI search
            results = await searcher.search_iraki_literature(
                query="immune checkpoint inhibitor acute kidney injury",
                specialty="nephrology",
                max_results=5,
            )

            print(f"Found {len(results)} results:")
            for result in results:
                print(f"- {result.title} (Score: {result.relevance_score:.2f})")
                print(f"  {result.citation}")
                print()

    # Run test
    # asyncio.run(test_literature_search())
