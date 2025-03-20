# Ivory Project

A full-stack application for clustering and dataset management, built with Next.js and FastAPI.

## Project Structure

The project is organized as a monorepo with two main components:

- `web/` - Frontend application built with Next.js and Mantine UI
- `src/subnet/` - Backend API built with FastAPI

## Backend (src/subnet/)

The backend is a FastAPI application that provides APIs for:
- Dataset management
- API endpoints for data processing

### Prerequisites

- Python 3.8+
- PostgreSQL
- Together API Key

### Setup

1. Create and activate a virtual environment:
```bash
cd src/subnet
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the development server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## Frontend (web/)

The frontend is a Next.js application with Mantine UI components.

### Prerequisites

- Node.js 18+
- Yarn

### Setup

1. Install dependencies:
```bash
cd web
yarn install
```

2. Run the development server:
```bash
yarn dev
```

The application will be available at `http://localhost:3000`

## Development Scripts

### Backend

- `python main.py` - Run the development server
- `python -m pytest` - Run tests
- `black .` - Format code
- `flake8` - Lint code

### Frontend

- `yarn dev` - Start development server
- `yarn build` - Build for production
- `yarn test` - Run tests
- `yarn lint` - Run ESLint
- `yarn typecheck` - Check TypeScript types
- `yarn storybook` - Start Storybook

## API Documentation

Once the backend server is running, you can access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.