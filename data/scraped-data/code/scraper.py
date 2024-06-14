from selenium import webdriver
from bs4 import BeautifulSoup


def get_content(url):
    driver = webdriver.Chrome()
    driver.get(url)
    url_page_source = driver.page_source
    driver.quit()
    content = BeautifulSoup(url_page_source, 'html.parser')
    print(content.prettify())
    return content