import requests
import json
from bs4 import BeautifulSoup, NavigableString
from collections import defaultdict

url = "cgi-bin/perseus/citequery3.pl?dbname=GreekMay20&getid=0&query=Pl.%20Ep.%20309a"
r = requests.get("http://artflsrv02.uchicago.edu/" + url)
outer_soup = BeautifulSoup(r.text, "html.parser")

words = defaultdict(list)
section = None
while outer_soup.find("td", align="right").contents[0]["href"] != url:

    tag = outer_soup.find("div", id="perseuscontent").contents[0]
    
    while (True in ("id" in parent.attrs and parent["id"] == "perseuscontent" for parent in tag.parents)):
        if tag.name == "w":
            print("LEMMA")
            word_id = tag["id"]
            word = tag.string
            inner_url = "http://artflsrv02.uchicago.edu/cgi-bin/perseus/morph.pl?lang=greek&id=" +  word_id
            inner_r = requests.get(inner_url)
            inner_soup = BeautifulSoup(inner_r.text, "html.parser")

            lemma = inner_soup.find("th", class_="lemma").string
            parse = inner_soup.find("tr", class_=["parse probable parserow", "parse disambiguated parserow"]).contents[1].string

            word_entry = {"word": word, "lemma": lemma, "parse": parse}
            words[section].append(word_entry)

        elif isinstance(tag, NavigableString):
            print("CHAR")
            if str(tag) != " ":
                words[section].append({"word": str(tag)})

        elif ("class" in tag.attrs and "mstonecustom" in tag["class"]):
            print("MILESTONE")
            print(words[section])
            section = tag["id"]

        elif len(tag.contents) > 0:
            print("STEPPING DOWN")
            if "class" in tag.attrs and "bibl" not in tag["class"]:
                tag = tag.contents[0]
                continue

        while tag.next_sibling == None:
            print("STEPPING UP")
            tag = tag.parent
        
        print("STEPPING SIDEWAYS")
        tag = tag.next_sibling
    
    url = outer_soup.find("td", align="right").contents[0]["href"]
    print(url)
    r = requests.get("http://artflsrv02.uchicago.edu/" + url)
    outer_soup = BeautifulSoup(r.text, "html.parser")

with open('lemmatized_epistles.json', 'w') as fp:
    json.dump(words, fp)






    


