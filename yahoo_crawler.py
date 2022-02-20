import json
import requests
from bs4 import BeautifulSoup

def LoadFromAJAX(url):
  person_string = get_web_page(url)
  person_dict = json.loads(person_string)
  # Get AJAX data information
  soup = BeautifulSoup(person_dict['data'], 'html5lib')
  span = soup.find_all('span', class_=None)
  c = []
  for s in span:
    c.append(s.text)
    #print(s.text)
  return c

def getpagecomment(id,num):
  url = 'https://movies.yahoo.com.tw/movieinfo_review.html/id=' + str(id) + '?sort=update_ts&order=desc&page=' + str(num)
  #print('url', url)
  article_page = get_web_page(url)
  soup = BeautifulSoup(article_page, 'html5lib')
  comments = []

  # 1. all comments on html ( no need to click "read more")
  usercom_list = soup.find('ul','usercom_list')
  spans = usercom_list.find_all('span', class_=None)
  for span in spans:
    # print(span.text)
    comments.append(span.text)

  # 3. get all AJAX data
  ajax = soup.find_all('div','check_reply')
  for a in ajax:
    # url = "https://movies.yahoo.com.tw/ajax/review_reply/130000000001513127"
    # print(LoadFromAJAX(url))
    # print(a['data-href'])
    comments = comments + LoadFromAJAX(a['data-href'])

  return comments

def get_web_page(url):
  resp = requests.get(
      url=url,
      cookies={'over18': '1'}
  )
  if resp.status_code != 200:
      print('Invalid url:', resp.url)
      return None
  else:
      return resp.text


def CrawlingMovie(id,data):

  # Get Main Page HTML : requests, soup
  url = 'https://movies.yahoo.com.tw/movieinfo_review.html/id=' + str(id)
  print('# url - %s' % url)
  article_page = get_web_page(url)
  soup = BeautifulSoup(article_page, 'html5lib')

  # Get Name ( Ch, En ) and Date
  name = soup.find('h1', 'inform_title')
  if name == None:
    print('No ID', id)
    return
  name_ch = name.text.split("\n")[0].strip()
  name_en = name.text.split("\n")[1].strip()
  date = soup.find('div','release_movie_time')
  d = date.text.split("：")[1].strip()
  print(id, name_ch, name_en, d)

  # Get Comments
  comments = []
  page_numbox = soup.find('div','page_numbox')
  max = 1

  # 1. See how many comments pages, find the max number
  if page_numbox != None:
    pages = page_numbox.find_all('a', class_=None)
    num_list = []
    for page in pages :
      p = int(float(page['href'].split("=")[-1]))
      if p > max:
        max = p
  # print('max=',max)

  # 2. Download the comments on HTMLe & AJAX
  for num in range(1, max + 1):
    print("comments downloading: %2d/%2d" % (num,max))
    comments = comments + getpagecomment(id,num)

  # 3. Append movie data
  data['movie'].append({
      'id': id,
      'url': url,
      'name': { 'Chinese': name_ch, 'English': name_en },
      'date': d,
      'cmment': comments
  })

  return data

rangeX = 6841
rangeY = 20000

data = {}
data['movie'] = []

for id in range(rangeX,rangeY):
  print('---------%05d---------'%(id))
  CrawlingMovie(id,data)
  # Save data every 10 movies
  if id % 10 == 0:
    with open('data/%05d-data.txt'%(id), 'w') as outfile:
      json.dump(data, outfile, ensure_ascii=False)
      print('#########%05d#########'%(id))
    data = {}
    data['movie'] = []
    
