import pandas as pd
import numpy as np

# Write a docstring like this and let Copilot complete the code:
"""
Generate a synthetic dataset of 2,000 real estate rental listings with:
1. Structured columns: beds, baths, sqft, neighborhood, latitude, longitude, price
2. Unstructured 'description' column containing raw text with amenities 
   (e.g., 'Spacious 2bd with in-unit laundry and garage parking', 'Cozy studio, no pets, hardwood floors')
3. Introduce 5% missing values randomly in 'sqft' and 'baths' to simulate messy real-world data.
"""


rng = np.random.default_rng(42)

neighborhoods = {
    "Downtown": (40.7128, -74.0060, 1.15),
    "Midtown": (40.7580, -73.9855, 1.10),
    "Uptown": (40.7847, -73.9712, 1.05),
    "Old Town": (40.7306, -73.9352, 1.02),
    "Riverside": (40.8013, -73.9665, 1.08),
}

amenity_templates = [
    "Spacious {beds}bd with in-unit laundry and garage parking",
    "Cozy {beds}bd, no pets, hardwood floors",
    "Bright {beds}bd with balcony, dishwasher, and gym access",
    "Modern {beds}bd near transit with central AC and secure entry",
    "Family-friendly {beds}bd with backyard, washer/dryer, and storage",
    "Charming {beds}bd with fireplace, rooftop access, and pet-friendly policy",
]

n_rows = 2000
beds = rng.integers(1, 5, size=n_rows)

# Add a small amount of realism so rents trend with bedrooms and square footage.
base_sqft = 450 + beds * 180 + rng.normal(0, 40, size=n_rows)
base_sqft = np.clip(base_sqft, 300, 2400)

baths = np.clip(rng.normal(loc=1 + 0.5 * beds, scale=0.45, size=n_rows), 1, 4)
baths = np.round(baths).astype(float)

# Create a random neighborhood assignment and a geographically plausible coordinate jitter.
nb_choices = rng.choice(list(neighborhoods.keys()), size=n_rows, p=[0.22, 0.24, 0.18, 0.18, 0.18])
latitudes = []
longitudes = []
price_multipliers = []

for neighborhood in nb_choices:
    base_lat, base_lon, multiplier = neighborhoods[neighborhood]
    latitudes.append(base_lat + rng.normal(0, 0.012))
    longitudes.append(base_lon + rng.normal(0, 0.012))
    price_multipliers.append(multiplier)

latitudes = np.array(latitudes)
longitudes = np.array(longitudes)
price_multipliers = np.array(price_multipliers)

# Add structured and textual features.
price = (
    1100
    + beds * 350
    + base_sqft * 1.35
    + (price_multipliers - 1) * 700
    + rng.normal(0, 150, size=n_rows)
)
price = np.round(price).astype(int)

# Introduce 5% missing values in sqft and baths.
sqft = np.round(base_sqft).astype(float)
baths = baths.astype(float)

sqft_missing_mask = rng.random(n_rows) < 0.05
baths_missing_mask = rng.random(n_rows) < 0.05

sqft[sqft_missing_mask] = np.nan
baths[baths_missing_mask] = np.nan

# Build descriptions using the raw amenities style requested.
descriptions = []
for i in range(n_rows):
    beds_val = int(beds[i])
    template = rng.choice(amenity_templates)
    descriptions.append(template.format(beds=beds_val))

# Assemble final dataset.
df = pd.DataFrame({
    "beds": beds,
    "baths": baths,
    "sqft": sqft,
    "neighborhood": nb_choices,
    "latitude": latitudes,
    "longitude": longitudes,
    "price": price,
    "description": descriptions,
})

# Save the synthetic dataset.
df.to_csv("synthetic_rentals.csv", index=False)
print("Created synthetic_rentals.csv with 2,000 rows.")
print(df.head())