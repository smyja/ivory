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
- Backend: http://localhost:8000

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

1. Navigate to the subnet directory:
```bash
cd src/subnet
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
- `src/subnet/` - Python backend application
  - Uses DuckDB for data storage
  - Prepared for future PostgreSQL integration for authentication
- `docker-compose.yml` - Docker Compose configuration

## Database

The project currently uses DuckDB for data storage, with the database file located at `src/subnet/datasets.db`. The database is persisted using Docker volumes.

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
