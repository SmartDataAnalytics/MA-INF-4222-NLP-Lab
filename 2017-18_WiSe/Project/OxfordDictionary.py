import requests
# import json

app_id = '5246da4d'
app_key = 'ccd7077872decfd0215dc2f6ae7c0aa9'
language = 'en'
base_url = 'https://od-api.oxforddictionaries.com:443/api/v1/entries/'
# word_id = 'birthday'


def getSynonyms(word_id):

    url = base_url + language + '/' + word_id.lower() + '/synonyms'

    r = requests.get(url, headers = {'app_id': app_id, 'app_key': app_key})

    if r.status_code != 200:
        return False
    # print("text \n" + r.text)
    # print("json \n" + json.dumps(r.json()))

    results = r.json()
    synonyms = []
    for synonym in results["results"][0]["lexicalEntries"][0]["entries"][0]["senses"][0]["synonyms"]:
        synonyms.append((synonym["text"], synonym["id"]))

    # print synonyms
    return synonyms


# getSynonyms(word_id)