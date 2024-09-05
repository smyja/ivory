"use client"; // Assuming you're using Next.js or a similar framework that requires this directive

import React, { useState } from 'react';
import { Title, Text, Container, Space, Box, Flex } from "@mantine/core";
import { ClusteringTable } from './table'; // Adjust the import path as needed
import SearchComponent from '../new/search';
import IndicatorBadge from './indicator'; // Adjust the import path for IndicatorBadge

const statuses = [
  { status: "Completed", color: "green" },
  { status: "Processing", color: "blue" },
  { status: "Failed", color: "red" },
  { status: "Queued", color: "yellow" },
];

export default function Cluster() {
  const [searchTerms, setSearchTerms] = useState<string[]>([]);
  const [selectedStatus, setSelectedStatus] = useState<string | null>(null);

  const handleSearch = (term: string) => {
    const terms = term.split(' ').filter(t => t.trim() !== '');
    setSearchTerms(terms);
  };

  const handleStatusFilter = (status: string) => {
    setSelectedStatus(selectedStatus === status ? null : status); // Toggle selection
  };

  return (
    <Container size="xl">
      <Title order={1} mt={20} mb={20} fw={300}>
        Clusters
      </Title>
      <Text mb={10} c="dimmed">
        Fast, Simple, Scalable and accelerated GPU clusters focused on
        simplicity, speed, & affordability.
      </Text>
      <SearchComponent 
  onSearch={handleSearch} 
  width="100%" 
  borderRadius="md" 
/>
<Space h="md" />

    

      {/* Filter section */}
      <Box mb={10}>
        <Flex justify="flex-start" align="center" gap="md">
          <Text fw={500} size="sm" mb={0}>
            Filter by Status:
          </Text>

          {/* Indicator badges for status filters */}
          {statuses.map(({ status, color }) => (
            <IndicatorBadge
              key={status}
              color={color}
              label={status}
              processing={status === "Processing"}
              textColor={selectedStatus === status ? color : 'gray'}
              onClick={() => handleStatusFilter(status)} // Handle click
              style={{ cursor: 'pointer' }} // Make the badge clickable
            />
          ))}
        </Flex>
      </Box>

      <Space h="md" />

      <ClusteringTable  selectedStatus={selectedStatus} />
    </Container>
  );
}
