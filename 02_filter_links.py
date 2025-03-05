import pickle
import pandas as pd

# Load the links
with open('data/all_links.pkl', 'rb') as handle:
    all_links = pickle.load(handle)

# Import filtered categories
cats = pd.read_excel('data/Cats_final.xlsx')
relevant_cats = set(cats['Categories']) # Make set of relevant categories for fast search

# Filter only relevant links
filtered_links = {}
for key in all_links: # Loop through all year/month combinations
    comp_month = [] # Initiate a placeholder for the filtered links in the current month
    
    # Loop through all competitions within the year and month
    for comp in all_links[key]: 
        if comp[0] in relevant_cats: # Append the comp if it is relevant
            comp_month.append(comp)
    
    # Make an entry in filtered links if there are some competitions left
    if len(comp_month) > 0: 
        filtered_links[key] = comp_month
    
    # Log
    with open('data/filtered_links.txt', 'a') as out:
        out.write(f'Filtered {len(comp_month)} out of {len(all_links[key])} competitions for year {key[0]} and month {key[1]}\n')

# Store info about dropped months
with open('data/filtered_links.txt', 'a') as out:
    out.write(f'Dropped {len(all_links) - len(filtered_links)} months')

# Store filtered links
with open('data/filtered_links.pkl', 'wb') as out:
    pickle.dump(filtered_links, out)