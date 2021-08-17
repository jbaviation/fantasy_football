import gspread
from yahoo_oauth import OAuth2

def load_sheet():
    json_file = 'fantasy-football-323118-a9b31ca12801.json'
    gc = gspread.service_account(filename=json_file)
    sh = gc.open('2021 Wellington Fantasy Draft').worksheet('2021')

    sh.update('A1', 'Hello World')
    # print(test.get(range="A3:K20"))
    
def yoauth():
    # Reference https://yahoo-fantasy-api.readthedocs.io/en/latest/authentication.html
    code = '7v9e6ef'  # generated on 8/16/21
    creds = {
    'consumer_key': 'dj0yJmk9WGxncWVWdVc3N0czJmQ9WVdrOWFXOHlORUkwZFhnbWNHbzlNQT09JnM9Y29uc3VtZXJzZWNyZXQmc3Y9MCZ4PTA2',
    'consumer_secret': '85ad9fa6ddb720460872fcc0c0a78d70496c4675'
            }
    
    with open(args['<json>'], 'w') as f:
        f.write(json.dumps(creds))
    oauth = OAuth2(None, None, from_file='oauth2.json')

    
if __name__ == '__main__':
    pass
#     yoauth()