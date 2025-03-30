FROM node:18-alpine

WORKDIR /app

# Enable Corepack and install correct Yarn version
RUN corepack enable && corepack prepare yarn@4.2.2 --activate

# Install dependencies
COPY package.json yarn.lock ./
RUN yarn install --frozen-lockfile

# Copy source code
COPY . .

# Build the application
RUN yarn build

# Expose the port
EXPOSE 3000

# Start the application
CMD ["yarn", "start"] 