import pickle
import os

from utils import scrape_competition

# Load filtered links
with open('data/filtered_links.pkl', 'rb') as handle:
    links = pickle.load(handle)

# Get names of already scraped year/months
scraped_months = os.listdir('data/raw_data')

# Loop through all links and scrape the competition results
for key in links:
    file_name = f'{key[0]}_{key[1]}.pkl' # Create the file name
    if file_name not in scraped_months: # Scrape only unscraped months
        res_month = {} # Initiate a place holder to store the results for the current month
        
        # Loop through all competitions within a given month
        for comp in links[key]: 
            error = None # Indicator whether an error ocurred
            try:
                res = scrape_competition(comp[1]) # Scrape the competition
                if type(res) == type(''): # Check for unsuccessful requests
                    error = res
                else:
                    if res[0] not in res_month: # Check if the title of the competition exists and if not, make an entry
                        res_month[res[0]] = {}
                    res_month[res[0]][comp[0]] = (res[1], res[2]) # Store the results
            except Exception as e: # Catch unexpected errors
                error = f'Error while scraping {comp[1]}. Error: {e}'
                
            # Log errors
            if error:
                with open('data/raw data/Scrape errors.txt', 'a') as out:
                    out.write(error + '\n')
            
        # Store results for current month
        with open(f'data/raw data/{file_name}', 'wb') as out:
            pickle.dump(res_month, out)
        
        # Print info
        print(f'Year {key[0]} month {key[1]} finished.')