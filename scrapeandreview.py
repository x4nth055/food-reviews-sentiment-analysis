import selenium.webdriver as webdriver
from bs4 import BeautifulSoup
from test import spitter

driver = webdriver.Chrome()
driver.get("https://www.trustpilot.com/review/olacabs.com")
html = driver.page_source
soup = BeautifulSoup(html, "html.parser")

customerlist = []
for customer in soup.find_all('div', {"class": "consumer-information__name"}):
    customer = customer.text.replace("\n", "").replace("  ", "")
    customerlist.append(customer)

headerlist = []
for header in soup.find_all('a', {"class": "link link--large link--dark"}):
    header = header.text.replace("\n", "").replace("  ", "")
    headerlist.append(header)

bodylist = []
for body in soup.find_all('p', {"class": "review-content__text"}):
    body = body.text.replace("\n", "").replace("  ", "")
    bodylist.append(body)

articlelist = []
for article in soup.find_all('article'):
    article = article.text.replace("\n", "").replace("  ", "")
    articlelist.append(article)
    

print(len(headerlist))
print(len(bodylist))
print(len(customerlist))
print(articlelist)

for body in articlelist:
    print(spitter(body))