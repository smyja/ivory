'use client';

import { useEffect, useState } from 'react';
import {
  Container,
  Title,
  Table,
  Text,
  Button,
  Group,
  Box,
  Card,
  ActionIcon,
  Tooltip,
  rem,
  Alert,
  Select,
  Pagination,
  Badge,
} from '@mantine/core';
import { useRouter, useSearchParams } from 'next/navigation';
import { IconEye, IconTrash, IconRefresh, IconPlus } from '@tabler/icons-react';
import { notifications } from '@mantine/notifications';
import { useDownloads } from '@/components/DownloadNotifications/DownloadContext';

const API_URL = process.env.NEXT_PUBLIC_API_URL;

interface Dataset {
  id: number;
  name: string;
  subset: string | null;
  split: string | null;
  status: string;
  download_date?: string;
  created_at?: string;
  clustering_status?: string;
  is_clustered?: boolean;
  latest_version?: number | null;
}

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<number | null>(null);
  const [viewLoading, setViewLoading] = useState<number | null>(null);
  const [clusteringStatus, setClusteringStatus] = useState<Record<number, string>>({});
  const [currentPage, setCurrentPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState<string | null>(null);
  const itemsPerPage = 10;
  const router = useRouter();
  const searchParams = useSearchParams();
  const { addDownload } = useDownloads();

  // Handle tracking a newly created dataset from query params
  useEffect(() => {
    const trackId = searchParams?.get('track_id');
    const trackName = searchParams?.get('track_name');

    if (trackId && trackName) {
      // Add the dataset to our downloads tracker
      addDownload({
        id: parseInt(trackId),
        name: trackName,
        status: 'pending',
        message: 'Starting download...',
      });

      // Remove the tracking params from the URL to avoid readding on refresh
      const newUrl = window.location.pathname;
      window.history.replaceState({}, '', newUrl);
    }
  }, [searchParams, addDownload]);

  const fetchDatasets = async () => {
    try {
      const response = await fetch(`${API_URL}/datasets`);
      if (!response.ok) {
        throw new Error('Failed to fetch datasets');
      }
      const data = await response.json();
      setDatasets(data);
    } catch (error) {
      console.error('Error fetching datasets:', error);
      notifications.show({
        title: 'Error',
        message: 'Failed to fetch datasets',
        color: 'red',
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDatasets();
  }, []);

  const handleView = (datasetId: number) => {
    setViewLoading(datasetId);
    router.push(`/dashboard/datasets/view?id=${datasetId}`);
  };

  const handleDelete = async (datasetId: number) => {
    // TODO: Implement delete functionality
    notifications.show({
      title: 'Not implemented',
      message: 'Delete functionality coming soon',
      color: 'red',
    });
  };

  const handleCluster = async (datasetId: number) => {
    setActionLoading(datasetId);
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/datasets/${datasetId}/cluster`,
        {
          method: 'POST',
        }
      );
      if (!response.ok) {
        throw new Error('Failed to start clustering');
      }
      notifications.show({
        title: 'Success',
        message: 'Clustering started successfully',
        color: 'green',
      });
      // Start polling for clustering status
      pollClusteringStatus(datasetId);
    } catch (error: any) {
      notifications.show({
        title: 'Error',
        message: error.message || 'Failed to start clustering',
        color: 'red',
      });
    } finally {
      setActionLoading(null);
    }
  };

  const pollClusteringStatus = async (datasetId: number) => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/datasets/${datasetId}/status?status_type=clustering`
        );
        if (!response.ok) {
          throw new Error('Failed to fetch clustering status');
        }
        const data = await response.json();
        setClusteringStatus((prev) => ({
          ...prev,
          [datasetId]: data.status,
        }));
        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(pollInterval);
        }
      } catch (error) {
        console.error('Error polling clustering status:', error);
        clearInterval(pollInterval);
      }
    }, 2000); // Poll every 2 seconds

    // Clean up interval after 5 minutes
    setTimeout(() => clearInterval(pollInterval), 300000);
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed':
        return 'green';
      case 'in_progress':
        return 'blue';
      case 'failed':
        return 'red';
      case 'pending':
        return 'yellow';
      default:
        return 'gray';
    }
  };

  const formatDate = (dateString: string | undefined) => {
    if (!dateString) return 'N/A';

    try {
      const date = new Date(dateString);
      // Check if the date is valid (not NaN)
      if (isNaN(date.getTime())) {
        return 'Invalid date';
      }
      return date.toLocaleDateString();
    } catch (error) {
      console.error('Error formatting date:', error);
      return 'Invalid date';
    }
  };

  // Filter the data based on the selected status
  const filteredData = statusFilter
    ? datasets.filter((dataset) => dataset.status === statusFilter)
    : datasets;

  // Paginate the data
  const paginatedData = filteredData.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  return (
    <Container size="xl" py="xl">
      <Group justify="space-between" mb="md">
        <Title order={1} fw={300}>
          Datasets
        </Title>
        <Button
          onClick={() => router.push('/dashboard/datasets/create')}
          variant="filled"
          color="black"
          radius="md"
          styles={{
            root: {
              transition: 'background-color 0.2s ease',
              '&:hover': {
                backgroundColor: '#333',
              },
            },
          }}
        >
          Create Dataset
        </Button>
      </Group>

      <Box>
        <Group justify="space-between" mb="md">
          <Select
            label="Filter by status"
            placeholder="All statuses"
            value={statusFilter}
            onChange={setStatusFilter}
            data={[
              { value: 'pending', label: 'Pending' },
              { value: 'in_progress', label: 'In Progress' },
              { value: 'completed', label: 'Completed' },
              { value: 'failed', label: 'Failed' },
            ]}
            clearable
          />
        </Group>

        <Box style={{ overflow: 'hidden', borderRadius: '8px' }}>
          <Table verticalSpacing="xs" highlightOnHover>
            <Table.Thead style={{ backgroundColor: '#f8f9fa' }}>
              <Table.Tr>
                {[
                  { label: 'Name', width: '30%' },
                  { label: 'Status', width: '20%' },
                  { label: 'Clustering Status', width: '20%' },
                  { label: 'Download Date', width: '20%' },
                  { label: 'Actions', width: '10%', minWidth: '200px' },
                ].map((header) => (
                  <Table.Th
                    key={header.label}
                    style={{
                      borderBottom: 'none',
                      width: header.width,
                      minWidth: header.minWidth,
                      color: 'gray',
                      backgroundColor: '#f8f9fa',
                      fontSize: '16px',
                      fontWeight: 500,
                      padding: '16px',
                      textTransform: 'none',
                    }}
                  >
                    {header.label}
                  </Table.Th>
                ))}
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {loading ? (
                <Table.Tr>
                  <Table.Td colSpan={5}>
                    <Group justify="center" py="xl">
                      <Text c="dimmed">Loading datasets...</Text>
                    </Group>
                  </Table.Td>
                </Table.Tr>
              ) : filteredData.length === 0 ? (
                <Table.Tr>
                  <Table.Td colSpan={5}>
                    <Text ta="center" c="dimmed">
                      No datasets found. Create one to get started.
                    </Text>
                  </Table.Td>
                </Table.Tr>
              ) : (
                paginatedData.map((dataset) => (
                  <Table.Tr key={dataset.id}>
                    <Table.Td>
                      <Text
                        size="sm"
                        fw={500}
                        style={{
                          maxWidth: '300px',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {dataset.name}
                      </Text>
                    </Table.Td>
                    <Table.Td>
                      <Badge
                        color={getStatusColor(dataset.status)}
                        variant="light"
                        style={{ whiteSpace: 'normal', maxWidth: 'none' }}
                      >
                        {dataset.status}
                      </Badge>
                    </Table.Td>
                    <Table.Td>
                      {clusteringStatus[dataset.id] && (
                        <Badge
                          color={getStatusColor(clusteringStatus[dataset.id])}
                          variant="light"
                          style={{ whiteSpace: 'normal', maxWidth: 'none' }}
                        >
                          {clusteringStatus[dataset.id]}
                        </Badge>
                      )}
                    </Table.Td>
                    <Table.Td>{formatDate(dataset.download_date || dataset.created_at)}</Table.Td>
                    <Table.Td>
                      <Group gap="xs" wrap="nowrap">
                        <Tooltip label="View Dataset">
                          <ActionIcon
                            variant="light"
                            color="black"
                            onClick={() => handleView(dataset.id)}
                            loading={viewLoading === dataset.id}
                            styles={{
                              root: {
                                transition: 'background-color 0.2s ease',
                                '&:hover': {
                                  backgroundColor: '#f0f0f0',
                                },
                              },
                            }}
                          >
                            <IconEye style={{ width: rem(16), height: rem(16) }} />
                          </ActionIcon>
                        </Tooltip>
                        <Tooltip label="Cluster Dataset">
                          <ActionIcon
                            variant="light"
                            color="black"
                            onClick={() => handleCluster(dataset.id)}
                            loading={actionLoading === dataset.id}
                            styles={{
                              root: {
                                transition: 'background-color 0.2s ease',
                                '&:hover': {
                                  backgroundColor: '#f0f0f0',
                                },
                              },
                            }}
                          >
                            <IconPlus style={{ width: rem(16), height: rem(16) }} />
                          </ActionIcon>
                        </Tooltip>
                        <Tooltip label="Delete Dataset">
                          <ActionIcon
                            variant="light"
                            color="red"
                            onClick={() => handleDelete(dataset.id)}
                            loading={actionLoading === dataset.id}
                            styles={{
                              root: {
                                transition: 'background-color 0.2s ease',
                                '&:hover': {
                                  backgroundColor: '#ffebee',
                                },
                              },
                            }}
                          >
                            <IconTrash style={{ width: rem(16), height: rem(16) }} />
                          </ActionIcon>
                        </Tooltip>
                      </Group>
                    </Table.Td>
                  </Table.Tr>
                ))
              )}
            </Table.Tbody>
          </Table>

          {!loading && filteredData.length > 0 && (
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
          )}
        </Box>
      </Box>
    </Container>
  );
}
