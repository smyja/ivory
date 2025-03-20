import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Create directories if they don't exist
os.makedirs("datasets/sample_dataset/train", exist_ok=True)

# Create sample data
data = {
    "text": [
        # Science Facts (10 items)
        "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
        "DNA is a double helix structure discovered by Watson and Crick in 1953.",
        "The human brain contains about 86 billion neurons.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The Earth completes one rotation on its axis in approximately 24 hours.",
        "Photosynthesis converts light energy into chemical energy.",
        "The periodic table has 118 known elements.",
        "Gravity on Earth is approximately 9.81 m/s².",
        "The human body has 206 bones.",
        "The Milky Way galaxy contains about 100-400 billion stars.",
        # Asian Politics (10 items)
        "China's Belt and Road Initiative aims to enhance global trade connectivity.",
        "Japan's constitution renounces war as a means of settling international disputes.",
        "India is the world's largest democracy by population.",
        "South Korea's economic development is known as the 'Miracle on the Han River'.",
        "Singapore's political system combines democracy with strong state intervention.",
        "Vietnam's economic reforms are known as 'Doi Moi'.",
        "Thailand's political system is a constitutional monarchy.",
        "Malaysia's government system is a federal constitutional monarchy.",
        "Indonesia is the world's largest archipelagic country.",
        "The Philippines has a presidential system of government.",
        # African Politics (10 items)
        "South Africa's transition from apartheid to democracy began in 1994.",
        "Nigeria is Africa's most populous country and largest economy.",
        "Egypt's political system is a semi-presidential republic.",
        "Kenya's government is a presidential representative democratic republic.",
        "Ethiopia is Africa's second-most populous country.",
        "Ghana was the first African country to gain independence from colonial rule.",
        "Morocco is a constitutional monarchy with an elected parliament.",
        "Tanzania's political system is a unitary presidential constitutional republic.",
        "Senegal is known for its stable democracy in West Africa.",
        "Rwanda has made significant progress in post-genocide reconstruction.",
        # History (10 items)
        "The Roman Empire fell in 476 CE.",
        "The Industrial Revolution began in Britain in the late 18th century.",
        "World War II ended in 1945.",
        "The French Revolution began in 1789.",
        "The American Civil War lasted from 1861 to 1865.",
        "The Renaissance period began in Italy in the 14th century.",
        "The Berlin Wall fell in 1989.",
        "The first successful powered flight was in 1903.",
        "The Great Depression began in 1929.",
        "The first computer was built in the 1940s.",
        # Mathematics (5 items)
        "The Pythagorean theorem states that a² + b² = c² in a right triangle.",
        "Pi (π) is approximately 3.14159.",
        "The Fibonacci sequence starts with 0, 1, 1, 2, 3, 5, 8...",
        "Euler's number (e) is approximately 2.71828.",
        "The golden ratio is approximately 1.618033988749895.",
        # Economics (5 items)
        "Supply and demand determine market prices in a free market.",
        "Inflation is the general increase in prices over time.",
        "GDP (Gross Domestic Product) measures a country's economic output.",
        "The law of diminishing returns states that adding more inputs eventually leads to smaller increases in output.",
        "Comparative advantage explains why countries trade with each other.",
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Create PyArrow Table
table = pa.Table.from_pandas(df)

# Write to Parquet file
pq.write_table(table, "datasets/sample_dataset/train/data.parquet")

print("Sample dataset created successfully!")
