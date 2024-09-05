import React, { useState } from 'react';
import { Table, Anchor, Box, Button, Text, Pagination } from '@mantine/core';
import { IconExternalLink } from '@tabler/icons-react';
import IndicatorBadge from './indicator';

// Sample data for demonstration
const allData = [
  {
    id: 1,
    clusterName: 'Customer Segmentation',
    status: 'Completed',
    timeCompleted: '2023-09-05T14:30:00',
  },
  {
    id: 2,
    clusterName: 'Product Categories',
    status: 'Processing',
    timeCompleted: null,
  },
  {
    id: 3,
    clusterName: 'User Behavior',
    status: 'Failed',
    timeCompleted: '2023-09-04T09:15:00',
  },
  {
    id: 4,
    clusterName: 'Text Classification',
    status: 'Completed',
    timeCompleted: '2023-09-03T22:45:00',
  },
  {
    id: 5,
    clusterName: 'Image Segmentation',
    status: 'Queued',
    timeCompleted: null,
  },
  {
    id: 6,
    clusterName: 'User Behavior',
    status: 'Failed',
    timeCompleted: '2023-09-04T09:15:00',
  },
  {
    id: 7,
    clusterName: 'User Behavior',
    status: 'Failed',
    timeCompleted: '2023-09-04T09:15:00',
  },
];

// Pagination chunk size
const CHUNK_SIZE = 5;

function formatDate(dateString) {
  if (!dateString) return '-';
  const date = new Date(dateString);
  return date.toLocaleString('en-US', { 
    year: 'numeric', 
    month: 'short', 
    day: 'numeric', 
    hour: '2-digit', 
    minute: '2-digit' 
  });
}

export function ClusteringTable() {
  const [activePage, setActivePage] = useState(1);

  // Chunk data for pagination
  const paginatedData = chunk(allData, CHUNK_SIZE);
  const currentPageData = paginatedData[activePage - 1] || [];

  const rows = currentPageData.map((row, index) => {
    const statusColor = 
      row.status === 'Completed' ? 'green' : 
      row.status === 'Processing' ? 'blue' : 
      row.status === 'Failed' ? 'red' : 'yellow';

    return (
      <Table.Tr
        key={row.id}
        style={index === currentPageData.length - 1 ? { borderBottom: '0px solid #ddd', borderRadius: '0 0 8px 8px' } : {}}
      >
        <Table.Td style={{ width: '20%' }}>
          <IndicatorBadge 
            color={statusColor} 
            label={row.status} 
            processing={row.status === 'Processing'}
            textColor={statusColor}
          />
        </Table.Td>
        <Table.Td style={{ width: '30%', paddingLeft: '10px' }}>
          <Anchor component="button" size="sm">
            {row.clusterName}
          </Anchor>
        </Table.Td>
        <Table.Td style={{ width: '30%', paddingLeft: '0px' }}>
          {formatDate(row.timeCompleted)}
        </Table.Td>
        <Table.Td style={{ width: '5%' }}>
          {row.id}
        </Table.Td>
        <Table.Td style={{ width: '15%', textAlign: 'right' }}>
          <Button
            variant="light"
            size="xs"
            rightSection={<IconExternalLink size={14} />}
            onClick={() => {
              console.log(`View clusters for ${row.clusterName}`);
            }}
          >
            View
          </Button>
        </Table.Td>
      </Table.Tr>
    );
  });

  return (
    <>
      <Box style={{ display: 'grid', gridTemplateColumns: '20% 30% 30% 10% 10%', padding: '0 16px', marginBottom: '6px' }}>
        <Text fw={500} size="sm" style={{ textAlign: 'left' }}>Status</Text>
        <Text fw={500} size="sm" style={{ textAlign: 'left' }}>Cluster Name</Text>
        <Text fw={500} size="sm" style={{ textAlign: 'left' }}>Time Completed</Text>
        <Text fw={500} size="sm" style={{ textAlign: 'left' }}>ID</Text>
        <Text fw={500} size="sm" style={{ textAlign: 'right' }}>View Clusters</Text>
      </Box>
      <Box style={{ borderRadius: '8px', overflow: 'hidden', border: '1px solid #ddd' }}>
        <Table verticalSpacing="xs" highlightOnHover>
          <Table.Tbody>{rows}</Table.Tbody>
        </Table>
        {/* Border at the bottom to separate the table from the pagination */}
        <Box style={{ borderTop: '1px solid #ddd', padding: '10px' }}>
          <Box style={{ display: 'flex', justifyContent: 'center' }}>
            <Pagination 
              total={paginatedData.length} 
              value={activePage} 
              onChange={setActivePage} 
              mt="sm" 
            />
          </Box>
        </Box>
      </Box>
    </>
  );
}

// Utility function to chunk data
function chunk<T>(array: T[], size: number): T[][] {
  if (!array.length) {
    return [];
  }
  const head = array.slice(0, size);
  const tail = array.slice(size);
  return [head, ...chunk(tail, size)];
}
