import pandas as pd

file_path = 'taxon.txt'
df = pd.read_csv(file_path, sep='\t', dtype=str)

# Search for Coleus decurrens
search_result = df[df['scientificName'].str.contains('Coleus decurrens', case=False, na=False)]
print("Matches for 'Coleus decurrens':")
print(search_result[['scientificName', 'iucnReference']].head())

# Search for any Coleus species
coleus_species = df[df['scientificName'].str.contains('Coleus', case=False, na=False)]
print(f"\nFound {len(coleus_species)} Coleus species")
print("First 5 Coleus species:")
print(coleus_species[['scientificName', 'iucnReference']].head())
