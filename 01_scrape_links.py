import pickle
import pandas as pd

from utils import scrape_month

# Scrape the links for all competitions
all_links = {}
for year in range(2023, 2000, -1): # Loop through years
    for month in range(1, 13, 1): # Loop through months
        all_links[(year, month)] = scrape_month(year, month, verbose=True)

# Store the links
with open('data/all_links.pkl', 'wb') as out:
    pickle.dump(all_links, out)

# Filter categories
cats = [j[0] for i in all_links for j in all_links[i]]
new_cats = [i for i in set(cats) if any(j in i for j in ['A', 'B', 'C', 'D', 'E']) and all(k not in i for k in ['TPV', 'TL', 'profi', 'OPEN', 'Poh', 'Profi', 'IDSF', '21', 'MČ', 'Sou', 'Hobby'])]
pd.Series(list(new_cats), name='Categories').to_excel('Cats.xlsx', index=False)