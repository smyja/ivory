'use client';

import React, { useState } from 'react';
import { Table, Anchor, Box, Button, Text, Pagination, Group, Badge, Select } from '@mantine/core';
import { IconExternalLink } from '@tabler/icons-react';
import { useRouter } from 'next/navigation';
import IndicatorBadge from './indicator';

export interface ClusteringHistory {
  id: number;
  dataset_id: number;
  dataset_name: string;
  version: number;
  clustering_status: 'queued' | 'processing' | 'completed' | 'failed';
  created_at: string;
  completed_at: string | null;
  error_message?: string;
}

interface ClusteringTableProps {
  selectedStatus: string | null;
  history: ClusteringHistory[];
}

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

export function ClusteringTable({ selectedStatus, history }: ClusteringTableProps) {
  const [currentPage, setCurrentPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState<string | null>(selectedStatus);
  const itemsPerPage = 10;
  const router = useRouter();

  // Filter the data based on the selected status
  const filteredData = statusFilter
    ? history.filter((row) => row.clustering_status === statusFilter)
    : history;

  // Paginate the data
  const paginatedData = filteredData.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  const getStatusColor = (status: string | undefined | null): string => {
    // Handle undefined or null status gracefully
    if (!status) {
      return 'gray'; // Return a default color if status is missing
    }

    switch (status.toLowerCase()) {
      case 'completed':
        return 'green';
      case 'processing':
      case 'queued':
      case 'started':
      case 'embedding':
      case 'clustering_l1':
      case 'titling_l1':
      case 'clustering_l2':
      case 'naming_l2':
      case 'saving':
        return 'blue';
      case 'failed':
        return 'red';
      default:
        return 'gray';
    }
  };

  const rows = paginatedData.map((row) => (
    <Table.Tr key={row.id}>
      <Table.Td>
        <Anchor component="button" size="sm" onClick={() => router.push(`/dashboard/datasets/view?id=${row.dataset_id}`)}>
          {row.dataset_name}
        </Anchor>
      </Table.Td>
      <Table.Td>
        <Group gap="xs">
          <IndicatorBadge
            color={getStatusColor(row.clustering_status)}
            label={row.clustering_status}
            processing={row.clustering_status === 'processing'}
          />
          {row.error_message && (
            <Badge color="red" variant="dot" title={row.error_message}>
              Error
            </Badge>
          )}
        </Group>
      </Table.Td>
      <Table.Td>{formatDate(row.created_at)}</Table.Td>
      <Table.Td>{formatDate(row.completed_at)}</Table.Td>
      <Table.Td>
        <Button
          variant="filled"
          size="xs"
          color="black"
          radius="md"
          rightSection={<IconExternalLink size={14} />}
          onClick={() => router.push(`/dashboard/datasets/cluster?id=${row.dataset_id}&version=${row.version}`)}
          disabled={row.clustering_status !== 'completed'}
          styles={{
            root: {
              transition: 'background-color 0.2s ease',
              '&:hover': {
                backgroundColor: '#333',
              },
              '&:disabled': {
                backgroundColor: '#ccc',
                '&:hover': {
                  backgroundColor: '#ccc',
                },
              },
            },
          }}
        >
          View Results
        </Button>
      </Table.Td>
    </Table.Tr>
  ));

  return (
    <Box>
      <Group justify="space-between" mb="md">
        <Select
          label="Filter by status"
          placeholder="All statuses"
          value={statusFilter}
          onChange={setStatusFilter}
          data={[
            { value: 'queued', label: 'Queued' },
            { value: 'processing', label: 'Processing' },
            { value: 'completed', label: 'Completed' },
            { value: 'failed', label: 'Failed' },
          ]}
          clearable
        />
      </Group>

      <Box style={{ overflow: 'hidden', borderRadius: '8px' }}>
        <Group p="md" style={{ backgroundColor: '#f8f9fa' }}>
          <Text fw={500} c="dimmed" style={{ flex: 1 }}>Dataset</Text>
          <Text fw={500} c="dimmed" style={{ flex: 1 }}>Clustering Status</Text>
          <Text fw={500} c="dimmed" style={{ flex: 1 }}>Started At</Text>
          <Text fw={500} c="dimmed" style={{ flex: 1 }}>Completed At</Text>
          <Text fw={500} c="dimmed" style={{ flex: 1 }}>Actions</Text>
        </Group>
        <Table verticalSpacing="xs" highlightOnHover style={{ backgroundColor: "white" }}>
          <Table.Tbody>{rows}</Table.Tbody>
        </Table>

        <Box
          style={{
            backgroundColor: '#f5f5f5',
            padding: '10px',
            display: 'flex',
            justifyContent: 'center',
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
    </Box>
  );
}
