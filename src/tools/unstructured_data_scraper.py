import os
import requests
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class UnstructuredDataScraper:
    """
    A class to scrape unstructured data (news articles) from the web using the NewsAPI.
    """

    def __init__(self):
        """
        Initializes the UnstructuredDataScraper and loads the NewsAPI key.
        """
        self.api_key = os.getenv("NEWS_API_KEY")
        if not self.api_key:
            raise ValueError("NEWS_API_KEY not found in environment variables. Please set it in your .env file.")
        self.base_url = "https://newsapi.org/v2/everything"

    def scrape_data(self, query: str, num_articles: int = 10) -> List[Dict[str, str]]:
        """
        Fetches news articles for a given query using the NewsAPI.

        Args:
            query (str): The search query for news articles.
            num_articles (int): The maximum number of articles to return.

        Returns:
            List[Dict[str, str]]: A list of dictionaries, where each dictionary
                                 represents a news article with keys 'title',
                                 'description', 'url', and 'publishedAt'.
        """
        params = {
            'q': query,
            'apiKey': self.api_key,
            'pageSize': num_articles,
            'sortBy': 'relevancy',
            'language': 'en'
        }

        print(f"--- Scraping news for query: '{query}' ---")
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()

            data = response.json()
            articles = data.get('articles', [])

            formatted_articles = [
                {
                    'title': article.get('title'),
                    'description': article.get('description'),
                    'url': article.get('url'),
                    'publishedAt': article.get('publishedAt')
                }
                for article in articles
            ]
            
            print(f"Successfully scraped {len(formatted_articles)} articles.")
            return formatted_articles

        except requests.exceptions.RequestException as e:
            print(f"Error during API request: {e}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return []

if __name__ == '__main__':
    try:
        scraper = UnstructuredDataScraper()
        query = "USD CNY exchange rate"
        articles_data = scraper.scrape_data(query, num_articles=5)

        if articles_data:
            print(f"\n--- Top 5 news articles for '{query}' ---")
            for i, article in enumerate(articles_data, 1):
                print(f"\nArticle {i}:")
                print(f"  Title: {article['title']}")
                print(f"  Description: {article['description']}")
                print(f"  URL: {article['url']}")
                print(f"  Published At: {article['publishedAt']}")
        else:
            print("Could not retrieve any articles.")

    except ValueError as e:
        print(f"Error: {e}")