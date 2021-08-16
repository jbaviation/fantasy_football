import gspread

json_file = 'fantasy-football-323118-a9b31ca12801.json'
gc = gspread.service_account(filename=json_file)
sh = gc.open('2021 Wellington Fantasy Draft').worksheet('2021')

sh.update('A1', 'Hello World')
# print(test.get(range="A3:K20"))