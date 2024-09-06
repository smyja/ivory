import React, { useState } from 'react';
import { Table, Anchor, Box, Button, Text, Pagination } from '@mantine/core';
import { IconExternalLink } from '@tabler/icons-react';
import IndicatorBadge from './indicator'; // Adjust the import path for IndicatorBadge

interface ClusteringTableProps {
  selectedStatus: string | null;
}

const data = [
  // Your data here
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
 
 
];

function formatDate(dateString: string | null) {
  if (!dateString) return '-';
  const date = new Date(dateString);
  return date.toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function ClusteringTable({ selectedStatus }: ClusteringTableProps) {
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 5;

  // Filter the data based on the selected status
  const filteredData = selectedStatus
    ? data.filter((row) => row.status === selectedStatus)
    : data;

  // Paginate the data
  const paginatedData = filteredData.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage);

  const rows = paginatedData.map((row, index) => {
    const statusColor =
      row.status === 'Completed'
        ? 'green'
        : row.status === 'Processing'
        ? 'blue'
        : row.status === 'Failed'
        ? 'red'
        : 'yellow';

    return (
      <Table.Tr key={row.id} style={index === paginatedData.length - 1 ? { borderRadius: '0 0 8px 8px' } : {}}>
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
        <Table.Td style={{ width: '30%', paddingLeft: '0px' }}>{formatDate(row.timeCompleted)}</Table.Td>
        <Table.Td style={{ width: '5%' }}>{row.id}</Table.Td>
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
        <Text fw={500} size="sm" style={{ textAlign: 'left' }}>
          Status
        </Text>
        <Text fw={500} size="sm" style={{ textAlign: 'left' }}>
          Cluster Name
        </Text>
        <Text fw={500} size="sm" style={{ textAlign: 'left' }}>
          Time Completed
        </Text>
        <Text fw={500} size="sm" style={{ textAlign: 'left' }}>
          ID
        </Text>
        <Text fw={500} size="sm" style={{ textAlign: 'right' }}>
          View Clusters
        </Text>
      </Box>
      <Box style={{ borderRadius: '8px', overflow: 'hidden', border: '1px solid #ddd' }}>
        <Table verticalSpacing="xs" highlightOnHover style={{backgroundColor:"white"}}>
          <Table.Tbody>{rows}</Table.Tbody>
        </Table>
        <Box
          style={{
            backgroundColor: '#f5f5f5', // Gray background color
            padding: '10px',
            display: 'flex',
            justifyContent: 'center',
            borderTop: '1px solid #ddd', // Border between table and footer
          }}
        >
          <Pagination
            total={Math.ceil(filteredData.length / itemsPerPage)}
            value={currentPage}
            onChange={setCurrentPage}
            size="sm"
          />
        </Box>
      </Box>
    </>
  );
}
