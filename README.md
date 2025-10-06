# Ivory Project

This is a monorepo containing both the frontend and backend of the Ivory project.

## Prerequisites

- Docker
- Docker Compose
- Node.js 18+ (for local development)
- Python 3.11+ (for local development)

## Getting Started

### Using Docker (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ivory.git
cd ivory
```

2. Start the services:
```bash
docker-compose up -d
```

3. Access the applications:
- Frontend: http://localhost:3000
- Backend: http://localhost:8000/api/v1

The application uses DuckDB for data storage, which is automatically initialized when the container starts.

### Local Development

#### Frontend

1. Navigate to the web directory:
```bash
cd web
```

2. Install dependencies:
```bash
yarn install
```

3. Start the development server:
```bash
yarn dev
```

#### Backend

1. Navigate to the backend directory:
```bash
cd src
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize the database:
```bash
python init_db.py
```

5. Start the server:
```bash
python main.py
```

## Project Structure

- `web/` - Next.js frontend application
- `src/` - Python backend application
  - Uses DuckDB for data storage
  - Prepared for future PostgreSQL integration for authentication
- `docker-compose.yml` - Docker Compose configuration

## Database

The project currently uses DuckDB for data storage, with the database file located at `src/datasets.db`. The database is persisted using Docker volumes.

### Future Database Integration

The project is designed to support PostgreSQL integration in the future, particularly for:
- User authentication
- Session management
- Additional data storage needs

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Analytical Query API (v1)

This is the canonical way to query datasets. The backend reads Parquet files directly via DuckDB using a safe JSON query spec — no SQL from the frontend, no row-copying into relational tables.

- Enable persistent DuckDB cache (optional): set `IVORY_USE_TABLE_INDEX=1`
- API version endpoint: `GET /api/v1/meta/version`
- Preview dataset schema: `GET /api/v1/query/preview/{dataset}`
- Run a query: `POST /api/v1/query/run`

Example payload:

```
{
  "dataset": "my_dataset",
  "select": ["text", "_hf_split"],
  "where": [{"column": "text", "op": "contains", "value": "example"}],
  "order_by": {"column": "_hf_split", "direction": "asc"},
  "limit": 50
}
```

Labels are managed per dataset/label name and are stored in SQLite files under `datasets/<dataset>/labels/`.

- Upsert by text: `POST /api/v1/query/label/upsert`
- Upsert by row id (preferred): `POST /api/v1/query/label/upsert_row`

Notes:
- New ingests include a stable `__row_id` column in Parquet for consistent joins and label/embedding alignment.
- Legacy ORM-backed endpoints will be deprecated; use the JSON query API for reads.

### Backfill `__row_id` for existing datasets

If you have existing Parquet files without `__row_id`, run:

```
python tools/backfill_row_ids.py --root datasets
```

### Disable legacy ORM read endpoints (optional)

Set this env var on the backend to return 410 for ORM-backed read endpoints:

```
IVORY_DISABLE_ORM_READS=1
```
