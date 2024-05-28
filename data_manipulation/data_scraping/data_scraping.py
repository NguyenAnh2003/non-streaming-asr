import bs4
from bs4 import BeautifulSoup
import requests, os
# web scraper

# vars
_URL = "https://realpython.com/beautiful-soup-web-scraper-python/"
_page = requests.get(_URL)

# functions
def audio_scraping():
    soup = BeautifulSoup(_page.content, "html.parser")
    return soup

if __name__ == "__main__":
    soup = audio_scraping() # 
    print(soup) # 