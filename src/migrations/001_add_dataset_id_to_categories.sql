-- Add dataset_id column to categories table
ALTER TABLE categories ADD COLUMN dataset_id INTEGER;

-- Add foreign key constraint
ALTER TABLE categories ADD CONSTRAINT fk_categories_dataset 
    FOREIGN KEY (dataset_id) 
    REFERENCES dataset_metadata(id); 