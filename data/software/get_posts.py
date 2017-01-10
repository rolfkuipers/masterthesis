"""
A simple example script to get all posts on a user's timeline.
Originally created by Mitchell Stewart.
<https://gist.github.com/mylsb/10294040>
"""
import facebook
import requests
from collections import Counter

#id of CNN page
# page_id = 5550296508 # CNN
page_id = 200273583406054 #eae
#page_id = 4
#page_id = 'FoxNews' # Fox news
page_id = 'universityofgroningen'
page_id = '111828655539438'
#graph = facebook.GraphAPI(access_token='EAACEdEose0cBAMzZB5by5SFxkrLz9lqhidvaBLQHe5nPGFR6qIEC7TqLuEhXucKrK5K3JWQz53LsuWZA1qsCDQ7eTNU5ZC7wSLhSgTdJspquLIYJvAFe3jqS8SocHukENQrbLk2cEZBtS8GzQZCDjVJP9c6eONrQjyMsYEWW4KZBOkUMjaMc5d', version='2.6')
graph = facebook.GraphAPI(access_token='EAACEdEose0cBAONH2SvqK0Uo6GsXrNJ8wZBTytImCE9aktFwb1ZAIQYZApH5aLroJQ7eIJbQz8BeU56UCV5oiW4SZAjn8SAEkJLvU8n4dlDEnpdKBK0zMkOKXBvkuCsGvRY71QYuYoZCAw8EnJJpBZA30pvZBqk6nJaKKiAZBmZAQNwZDZD', version='2.6')

def get_all(items):
    all_items = []
    
    # Wrap this block in a while loop so we can keep paginating requests until
    # finished.
    i = 0
    print("Getting data", end="")
    while True:
        print("*", end="")
        try:
            # Perform some action on each post in the collection we receive from
            # Facebook.
            all_items.extend(items['data'])
            # Attempt to make a request to the next page of data, if it exists.
            items = requests.get(items['paging']['next']).json()
        except KeyError:
            # When there are no more pages (['paging']['next']), break from the
            # loop and end the script.
            print("....Finished getting data")
            break
        if 'created_time' in all_items[-1]:
            year = all_items[-1]['created_time'][0:4]
            if year != '2016':
                print("....Finished getting data")
                break
        if len(all_items) > 100:
            break
      
    return all_items

all_posts = get_all(graph.get_object(id="{}/feed".format(page_id)))

#dict_keys(['updated_time', 'name', 'created_time', 'id', 'picture', 'caption', 'type', 'from', 'actions', 'link', 'is_expired', 'status_type', 'message', 'privacy', 'icon', 'is_hidden', 'description'])
for post in all_posts:
    if 'message' in post:
        print(post['message'])
        reactions = get_all(graph.get_object(id="{}/reactions".format(post['id'])))
        reaction_types = Counter([reaction['type'] for reaction in reactions])
        print(reaction_types)
        print()