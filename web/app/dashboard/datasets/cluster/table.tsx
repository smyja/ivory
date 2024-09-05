import React from 'react';
import { Table, Anchor, Box, Button, Text } from '@mantine/core';
import { IconExternalLink } from '@tabler/icons-react';
import IndicatorBadge from './indicator';

const data = [
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
];

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
  const rows = data.map((row, index) => {
    const statusColor = 
      row.status === 'Completed' ? 'green' : 
      row.status === 'Processing' ? 'blue' : 
      row.status === 'Failed' ? 'red' : 'yellow';

    return (
      <Table.Tr key={row.id} style={index === data.length - 1 ? { borderRadius: '0 0 8px 8px' } : {}}>
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
        <Table.Td style={{ width: '5%', paddingleft: '-20px'  }}>
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
      <Box style={{ display: 'grid', gridTemplateColumns: '20% 30% 30% 10% 10%', padding: '0 16px', marginBottom: '6px', borderBottom: '1px' }}>
        <Text fw={500} size="sm" style={{ textAlign: 'left' }}>Status</Text>
        <Text fw={500} size="sm" style={{ textAlign: 'left' }}>Cluster Name</Text>
        <Text fw={500} size="sm" style={{ textAlign: 'left' }}>Time Completed</Text>
        <Text fw={500} size="sm" style={{ textAlign: 'left' }}>ID</Text>
        <Text fw={500} size="sm" style={{ textAlign: 'right' }}>View Clusters</Text>
      </Box>
      <Box style={{ borderRadius: '8px', overflow: 'hidden' }}>
        <Table verticalSpacing="xs" striped highlightOnHover>
          <Table.Tbody>{rows}</Table.Tbody>
        </Table>
      </Box>
    </>
  );
}
