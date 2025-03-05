import pickle
import pandas as pd
import os

from utils import process_comp

# Load category mapping
cats = pd.read_excel('data/Cats_final.xlsx', index_col='Categories')['Map'].to_dict() # Convert to a dictionary

# Get names of raw data files
raw_files = os.listdir('data/raw data')
if 'Scrape errors.txt' in raw_files: # Do not consider log files
    raw_files.remove('Scrape errors.txt')

# Get names of already processed files
processed_files = os.listdir('data/processed data')

# Loop through all competitions
for file in raw_files: # Loop through all files (year/month)
    if file not in processed_files: # Process only unprocessed files
        # Load the data
        with open(f'data/raw data/{file}', 'rb') as handle:
            month_data = pickle.load(handle)

        # Get year and month
        year, month = file.replace('.pkl', '').split('_')

        # Get a placeholder for data from current month
        df = pd.DataFrame([])

        # Loop through the competitions within the given month
        for event in month_data: # Loop through all events
            for comp in month_data[event]: # Loop through all competitions within an event
                df_comp = process_comp(month_data[event][comp])
                if type(df_comp) == type(''): # Check if error ocurred
                    with open('data/processed data/Errors.txt', 'a', encoding='utf-8') as out:
                        out.write(f'Year {year} month {month} event {event} comp {comp}. ' + df_comp + '\n')
                else: # Otherwise append to the data obtained so far
                    df_comp['Category'] = cats[comp] # Map category to unified categories
                    df_comp['Event'] = event # Add title of the event
                    df = pd.concat([df, df_comp])

        # Store the data for current month year
        with open(f'data/processed data/{file}', 'wb') as out:
            pickle.dump(df.to_dict('list'), out)

        # Print info
        print(f'{df.shape[0]:,} total rows for year {year} month {month}'.replace(',', ' '))
