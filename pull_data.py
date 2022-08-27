import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import re
import difflib

fteams = ['RK', '1.0', 'ES', 'AY', 'KA', 'BL', 'NG', 'RW', 'BM', 'JB', 'BD', '2.0']  # 2022 order

replacements = {'Pat Mahomes': 'Patrick Mahomes', 
                'Allen Robinson II': 'Allen Robinson',
                'D.K. Metcalf': 'DK Metcalf',
                'J.K. Dobbins': 'JK Dobbins',
                'D.J. Moore': 'DJ Moore',
                'A.J. Brown': 'AJ Brown',
                'A.J. Dillon': 'AJ Dillon',
                'Ben Roethlisberger PIT': 'Ben Roethlisberger',
                'D.J. Chark': 'DJ Chark',
                'DJ Chark Jr.': 'DJ Chark',
                'Darrell Henderson Jr.': 'Darrell Henderson',
                'J.D. McKissic': 'JD McKissic',
                'Justin Fields CHI': 'Justin Fields',
                'Kirk Cousins MIN': 'Kirk Cousins',
                'Marvin Jones Jr.': 'Marvin Jones',
                'Melvin Gordon III': 'Melvin Gordon',
                'Michael Pittman Jr.': 'Michael Pittman',
                'Odell Beckham Jr': 'Odell Beckham',
                'Odell Beckham Jr.': 'Odell Beckham',
                'Robert Tonyan Jr.': 'Robert Tonyan',
                'Ronald Jones II': 'Ronald Jones',
                'Teddy Birdgewater DEN': 'Teddy Birdgewater',
                'Trey Lance SF': 'Trey Lance',
                'Will Fuller V': 'Will Fuller',
                'Zack Wilson NYJ': 'Zach Wilson',
                'Brian Robinson Jr': 'Brian Robinson',
                'Darrell Henderson Jr': 'Darrell Henderson',
                'Irv Smith Jr': 'Irv Smith',
                'Isaih Pacheco': 'Isiah Pacheco',
                'Kenneth Walker III': 'Kenneth Walker',
                'Mark Ingram II': 'Mark Ingram',
                'Michael Pittman Jr': 'Michael Pittman',
                'Mitchell Trubisky': 'Mitch Trubisky',
                'Travis Etienne Jr': 'Travis Etienne'
                }

ffc_teams =    {'Carolina': 'CAR',
                'Kansas City': 'KC',
                'Minnesota': 'MIN',
                'New Orleans': 'NO',
                'Buffalo': 'BUF',
                'Tennessee': 'TEN',
                'Baltimore': 'BAL',
                'Dallas': 'DAL',
                'Arizona': 'ARI',
                'Green Bay': 'GB',
                'NY Giants': 'NYG',
                'Seattle': 'SEA',
                'LA Chargers': 'LAC',
                'Indianapolis': 'IND',
                'Cleveland': 'CLE',
                'Atlanta': 'ATL',
                'Washington': 'WAS',
                'Commanders': 'WAS',
                'Tampa Bay': 'TB',
                'Pittsburgh': 'PIT',
                'Cincinnati': 'CIN',
                'Las Vegas': 'LV',
                'Chicago': 'CHI',
                'San Francisco': 'SF',
                'LA Rams': 'LAR',
                'Detroit': 'DET',
                'Philadelphia': 'PHI',
                'Miami': 'MIA',
                'Jacksonville': 'JAX',
                'Denver': 'DEN',
                'NY Jets': 'NYJ',
                'New England': 'NE',
                'Houston': 'HOU'}

espn_teams =   {'Carolina Panthers': 'CAR',
                'Kansas City Chiefs': 'KC',
                'Minnesota Vikings': 'MIN',
                'New Orleans Saints': 'NO',
                'Buffalo Bills': 'BUF',
                'Tennessee Titans': 'TEN',
                'Baltimore Ravens': 'BAL',
                'Dallas Cowboys': 'DAL',
                'Arizona Cardinals': 'ARI',
                'Green Bay Packers': 'GB',
                'New York Giants': 'NYG',
                'Seattle Seahawks': 'SEA',
                'Los Angeles Chargers': 'LAC',
                'Indianapolis Colts': 'IND',
                'Cleveland Browns': 'CLE',
                'Atlanta Falcons': 'ATL',
                'Washington Football Team': 'WAS',
                'Washington Commanders': 'WAS',
                'Tampa Bay Buccaneers': 'TB',
                'Pittsburgh Steelers': 'PIT',
                'Cincinnati Bengals': 'CIN',
                'Las Vegas Raiders': 'LV',
                'Chicago Bears': 'CHI',
                'San Francisco 49ers': 'SF',
                'Los Angeles Rams': 'LAR',
                'Detroit Lions': 'DET',
                'Philadelphia Eagles': 'PHI',
                'Miami Dolphins': 'MIA',
                'Jacksonville Jaguars': 'JAX',
                'Denver Broncos': 'DEN',
                'New York Jets': 'NYJ',
                'New England Patriots': 'NE',
                'Houston Texans': 'HOU'}
sn_teams = {'Panthers': 'CAR',
            'Chiefs': 'KC',
            'Vikings': 'MIN',
            'Saints': 'NO',
            'Bills': 'BUF',
            'Titans': 'TEN',
            'Ravens': 'BAL',
            'Cowboys': 'DAL',
            'Cardinals': 'ARI',
            'Packers': 'GB',
            'Giants': 'NYG',
            'Seahawks': 'SEA',
            'Chargers': 'LAC',
            'Colts': 'IND',
            'Browns': 'CLE',
            'Falcons': 'ATL',
            'Washington': 'WAS',
            'Commanders': 'WAS',
            'Buccaneers': 'TB',
            'Steelers': 'PIT',
            'Bengals': 'CIN',
            'Raiders': 'LV',
            'Bears': 'CHI',
            '49ers': 'SF',
            'Rams': 'LAR',
            'Lions': 'DET',
            'Eagles': 'PHI',
            'Dolphins': 'MIA',
            'Jaguars': 'JAX',
            'Broncos': 'DEN',
            'Jets': 'NYJ',
            'Patriots': 'NE',
            'Texans': 'HOU'}

def ds_df():
    """Draft Sharks"""
    url = 'https://www.draftsharks.com/adp/superflex'
    ds = pd.read_html(url)[0]

    # Break up string
    ds['ds_rank'] = ds['Positional Rank'].str.split('.', n=1).str[0]
    ds['name_pos'] = ds['Positional Rank'].str.split('.', n=1).str[-1]

    reg_pat = r'^(.*)(QB|RB|WR|TE|DEF|K)$'
    ds[['name', 'pos']] = ds['name_pos'].str.extract(reg_pat)

    # Clean up strings
    for col in ['ds_rank', 'name', 'pos']:
        ds[col] = ds[col].str.strip()
    ds['ds_rank'] = ds['ds_rank'].astype(int)

    # Rename columns
    ds = ds.rename(lambda x: x.lower().replace(' ','_'), axis=1)

    # Pick necessary columns
    cols = ['ds_rank', 'name', 'pos', 'bye_week']
    ds = ds[cols]

    # Make position rank column
    pos_rank = ds.groupby('pos')['ds_rank'].rank().astype(int)
    ds['ds_pos'] = ds['pos'] + pos_rank.astype(str)

    # Drop position column
    ds = ds.drop('pos', axis=1)
    
    # Replace common characters
    ds['name'] = ds['name'].str.replace(r'\.', '', regex=True).str.strip()
    ds = ds.replace(replacements).replace(sn_teams).replace(espn_teams, regex=True)
    return ds

# ESPN
def espn_df():
    url = 'https://www.espn.com/fantasy/football/story/_/id/34058190/eric-karabell-fantasy-football-superflex-rankings-2022'
    req = requests.get(url)
    soup = BeautifulSoup(req.content, features='lxml')
    para = ["name", "team_posrank"]
    ls = soup.find_all('p')[3].text
    ls = re.split(r'\n', ls)  # remove newline characters
    df = pd.DataFrame([re.split(r'(?:\A\w+\.)|(?:,)', txt) for txt in ls]).drop(0, axis=1)
    df.columns = para
    df['name'] = df['name'].str.strip()
    df['team_posrank'] = df['team_posrank'].str.strip()

    reg_pat = r'^(\w*)\s*\((.*)\)$'
    df[['team', 'espn_pos']] = df['team_posrank'].str.extract(reg_pat)

    df = df.reset_index().rename({'index': 'espn_rank'}, axis=1)
    df['espn_rank'] = df['espn_rank'].astype(int) + 1
    df = df.drop('team_posrank', axis=1)

    # Replace common characters
    df['team'] = df['team'].str.upper()
    df['name'] = (df['name'].str.replace(r'\.', '', regex=True)
                            .str.strip()
                            .str.replace(r'\s*DST$', '', regex=True))
    df = df.replace(replacements).replace(sn_teams).replace(espn_teams, regex=True)
    return df

def rt_df():
    url = 'https://www.rototrade.com/rankings/2022/superflex/halfppr'
    req = requests.get(url)
    soup = BeautifulSoup(req.content, features='lxml')

    data = []
    for row in soup.find_all(attrs={'class': 'playerrow'}):
        row_data = []
        row_data.append(row.find('div', {'class': 'overallrank'}).text)
        row_data.append(row.find('div', {'class': 'name'}).text)
        row_data.append(row.find('div', {'class': 'pos'}).text)
        row_data.append(row.find('div', {'class': 'posandteam'}).text)
        data.append(row_data)

    rt = pd.DataFrame(data, columns=['rt_rank', 'name', 'rt_pos', 'team'])

    # Replace common characters
    rt['name'] = (rt['name'].str.replace(r'\.', '', regex=True)
                            .str.strip())
    rt['team'] = rt['team'].str.strip()
    rt = rt.replace(replacements).replace(sn_teams).replace(espn_teams, regex=True)
    return rt

def ffc_df():
    url = 'https://fantasyfootballcalculator.com/rankings/2qb'
    req = requests.get(url)
    soup = BeautifulSoup(req.content, features='lxml')
    cols = ['ffc_rank', 'name', 'team', 'pos', 'bye_week']
    rows = soup.find(attrs={'class': 'table table-striped mt-3'}).find("tbody").find_all("tr")

    # Read thru the table
    data = []
    for row in rows:
        data.append([val.text for val in row.find_all("td")])

    # Turn into dataframe
    ffc = pd.DataFrame(data, columns=cols)

    # Clean text and set types
    ffc['ffc_rank'] = ffc['ffc_rank'].str.replace('\.', '', regex=True).astype(int)
    ffc['bye_week'] = ffc['bye_week'].astype(int)
    for col in ['name', 'team', 'pos']:
        ffc[col] = ffc[col].str.strip()

    # Create pos rank
    pos_rank = ffc.groupby('pos')['ffc_rank'].rank().astype(int)
    ffc['ffc_pos'] = ffc['pos'] + pos_rank.astype(str)

    # Drop columns
    ffc = ffc.drop('pos', axis=1)

    # Replace common characters
    ffc['name'] = (ffc['name'].str.replace(r'\.', '', regex=True)
                              .str.replace(r'\s+Defense$', '', regex=True)
                              .str.strip())
    ffc = (ffc.replace(replacements)
              .replace(sn_teams)
              .replace(espn_teams, regex=True)
              .replace(ffc_teams, regex=True))
    
    return ffc

def gen_draft(ffc=ffc_df(), rt=rt_df(), espn=espn_df(), ds=ds_df()):
    # Merge
    df=(ffc.merge(rt, how='outer', on='name', suffixes=['_ffc', '_rt'])
        .merge(espn, how='outer', on='name', suffixes=[None, '_espn'])
        .merge(ds, how='outer', on='name', suffixes=[None, '_ds']))

    df['avg_rank'] = df[['ffc_rank', 'rt_rank', 'espn_rank', 'ds_rank']].mean(axis=1, numeric_only=True)
    df = df[['avg_rank', 'ffc_rank', 'rt_rank', 'espn_rank', 'ds_rank', 'name', 'ffc_pos', 'rt_pos', 'espn_pos', 'ds_pos',
            'team_ffc', 'team_rt', 'team', 'bye_week']].sort_values('avg_rank')

    # Clean cells
    int_cols = ['ffc_rank', 'rt_rank', 'espn_rank', 'ds_rank', 'bye_week']
    df[int_cols] = df[int_cols].astype('float').astype('Int32').astype(object)
    df = df.fillna('')  # Easier to read empty cells than NaNs

    def check_team(row):
        ffc = row['team_ffc']
        cbs = row['team_rt']
        espn = row['team']
        if len(ffc) > 1:
            return ffc
        if len(cbs) > 1:
            return cbs
        if len(espn) > 1:
            return espn

    def get_pos(row):
        ffc = row['ffc_pos']
        cbs = row['rt_pos']
        espn = row['espn_pos']
        sn = row['ds_pos']
        if len(ffc) > 1:
            return re.findall(r'(\D+)\d+', ffc)[0]
        if len(cbs) > 1:
            return re.findall(r'(\D+)\d+', cbs)[0]
        if len(espn) > 1:
            return re.findall(r'(\D+)\d+', espn)[0]
        else:
            return re.findall(r'(\D+)\d+', sn)[0]

    def replace_def(row):
        if row['pos'] == 'DST':
            row['team'] = row['name']

    # Clean up team columns
    df['team'] = df.apply(lambda row: check_team(row), axis=1)
    df = df.drop(['team_ffc', 'team_rt'], axis=1)

    # Clean up position columns
    pos_cols = ['ffc_pos', 'rt_pos', 'espn_pos', 'ds_pos']
    df[pos_cols] = df[pos_cols].replace({'DEF': 'DST', 'PK': 'K'}, regex=True)
    df['pos'] = df.apply(lambda row: get_pos(row), axis=1)

    # Clean up name column
    df['team'] = np.where(df['pos']=='DST', df['name'], df['team'])
    
    # Prepare for draft
    df['avg_rank'] = pd.to_numeric(df['avg_rank'], errors='coerce')
    df = df[['avg_rank', 'ffc_rank', 'rt_rank', 'espn_rank', 'ds_rank', 'pos', 'name', 'ffc_pos', 'rt_pos', 'espn_pos', 'ds_pos',
            'team', 'bye_week']].sort_values('avg_rank')

    df['avg_rank'] = df['avg_rank'].apply(lambda x: float("{:.1f}".format(x)))
    df['f_team'] = np.nan
    df['f_pick'] = np.nan

    # Reset Index
    df = df.reset_index(drop=True)

    return df

def find_player(name, df):
    name_list = list(df['name'])
    close_matches = difflib.get_close_matches(name, name_list, n=5)
    contains = [nm for nm in name_list if name in nm]
    print('Close Matches:\n{}'.format(close_matches))
    print('Contains:\n{}'.format(contains))

ffc = ffc_df()
rt = rt_df()
espn = espn_df()
ds = ds_df()

class draft_controller:
    def __init__(self, board, pick=0, auto_mode=False, rounds=20, backup=True):
        self.pick = int(pick)
        self.last_pick = {'name': '',
                          'fteam': '',
                          'pick': ''
                         }
        self.board = board
        self.auto_mode = auto_mode
        self.rounds = rounds
        self.backup = backup
        self.create_draft_list()

    def create_draft_list(self):
        d_list = fteams + list(reversed(fteams))
        d_list_reps = int(round(self.rounds / 2, 1))

        # Entire draft order fteams
        self.draft_order = d_list * d_list_reps

        # Entire draft order with tuples
        self.draft_order_tuples = [(ele, i) for i, ele in enumerate(self.draft_order, 1)]
        
        # Draft picks by owner
        d = {}
        for a, b in self.draft_order_tuples:
            if a in d:
                d[a].append(b)
            else:
                d[a] = [b]
        self.draft_picks = d

    def selection(self, name, pick=-1, fteam=''):
        # Copy board
        board = self.board.copy()

        if name.lower() not in list(self.board['name'].str.lower()):
            print('WARNING: name entered not in draft list')
            print('Possible Options: {}'.format(difflib.get_close_matches(name, list(self.board['name']))))
            return
        row = board[board['name'].str.match(name, case=False)]
        if len(row)>1:
            print('WARNING: Multiple name matches')
            return

        # If made it this far, look for auto_mode
        if self.auto_mode:
            self.pick += 1
            self.fteam = self.draft_order[self.pick-1]
        elif fteam not in fteams:
            raise ValueError('Team entered not in list')
        elif (pick < 0) | (pick > self.rounds*len(fteams)):
            raise ValueError('Please enter auto_mode of a valid pick number')
        elif fteam not in fteams:
            raise ValueError('Entered wrong fantasy team name')
        else:
            self.pick = pick
            self.fteam = fteam

        idx = board.index[board['name'].str.fullmatch(name, case=False)][0]  # index location
        board.loc[idx,'f_team'] = self.fteam
        board.loc[idx,'f_pick'] = self.pick

        self.board = board

        # Save file in case overwritten
        if self.backup:
            self.board.to_csv('stored_draft/2022_draft.csv')
        
        return

