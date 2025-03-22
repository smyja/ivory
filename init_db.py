# Create texts table
conn.execute(
    """
    CREATE TABLE IF NOT EXISTS texts (
        id INTEGER PRIMARY KEY,
        dataset_id INTEGER NOT NULL,
        text TEXT NOT NULL,
        FOREIGN KEY (dataset_id) REFERENCES dataset_metadata(id)
    )
    """
)
