'use client';

import { useEffect, useState } from 'react';
import {
  Container,
  Title,
  Table,
  Text,
  Button,
  Group,
  Badge,
  Card,
  ActionIcon,
  Tooltip,
  rem,
  Alert,
} from '@mantine/core';
import { useRouter } from 'next/navigation';
import { IconEye, IconTrash, IconRefresh, IconPlus } from '@tabler/icons-react';
import { notifications } from '@mantine/notifications';

const API_URL = process.env.NEXT_PUBLIC_API_URL;

interface Dataset {
  id: number;
  name: string;
  subset: string | null;
  split: string | null;
  status: string;
  download_date: string;
  clustering_status?: string;
}

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<number | null>(null);
  const [clusteringStatus, setClusteringStatus] = useState<Record<number, string>>({});
  const router = useRouter();

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
    router.push(`/dashboard/datasets/view?id=${datasetId}`);
  };

  const handleRefresh = async (datasetId: number) => {
    try {
      const response = await fetch(`${API_URL}/datasets/${datasetId}/verify`);
      if (!response.ok) {
        throw new Error('Failed to verify dataset');
      }
      await fetchDatasets(); // Refresh the list after verification
      notifications.show({
        title: 'Success',
        message: 'Dataset verified successfully',
        color: 'green',
      });
    } catch (error) {
      console.error('Error verifying dataset:', error);
      notifications.show({
        title: 'Error',
        message: 'Failed to verify dataset',
        color: 'red',
      });
    }
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
          `${process.env.NEXT_PUBLIC_API_URL}/datasets/${datasetId}/clustering_status`
        );
        if (!response.ok) {
          throw new Error('Failed to fetch clustering status');
        }
        const data = await response.json();
        setClusteringStatus(prev => ({
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

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  return (
    <Container size="xl" py="xl">
      <Card shadow="sm" padding="lg" radius="md" withBorder mb="xl">
        <Group justify="space-between" mb="md">
          <Title order={2}>Datasets</Title>
          <Button
            onClick={() => router.push('/dashboard/datasets/create')}
            variant="light"
            color="blue"
          >
            Create Dataset
          </Button>
        </Group>

        {error && (
          <Alert color="red" className="mb-4">
            {error}
          </Alert>
        )}

        <Table highlightOnHover>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Name</Table.Th>
              <Table.Th>Subset</Table.Th>
              <Table.Th>Split</Table.Th>
              <Table.Th>Status</Table.Th>
              <Table.Th>Clustering Status</Table.Th>
              <Table.Th>Download Date</Table.Th>
              <Table.Th>Actions</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {datasets.map((dataset) => (
              <Table.Tr key={dataset.id}>
                <Table.Td>
                  <Text size="sm" fw={500}>
                    {dataset.name}
                  </Text>
                </Table.Td>
                <Table.Td>
                  <Text size="sm">{dataset.subset || '-'}</Text>
                </Table.Td>
                <Table.Td>
                  <Text size="sm">{dataset.split || '-'}</Text>
                </Table.Td>
                <Table.Td>
                  <Badge color={getStatusColor(dataset.status)} variant="light">
                    {dataset.status}
                  </Badge>
                </Table.Td>
                <Table.Td>
                  {clusteringStatus[dataset.id] && (
                    <Badge
                      color={
                        clusteringStatus[dataset.id] === 'completed'
                          ? 'green'
                          : clusteringStatus[dataset.id] === 'failed'
                            ? 'red'
                            : 'black'
                      }
                    >
                      {clusteringStatus[dataset.id]}
                    </Badge>
                  )}
                </Table.Td>
                <Table.Td>
                  <Text size="sm">{formatDate(dataset.download_date)}</Text>
                </Table.Td>
                <Table.Td>
                  <Group gap="xs">
                    <Tooltip label="View Dataset">
                      <ActionIcon
                        variant="light"
                        color="blue"
                        onClick={() => handleView(dataset.id)}
                      >
                        <IconEye style={{ width: rem(16), height: rem(16) }} />
                      </ActionIcon>
                    </Tooltip>
                    <Tooltip label="Verify Dataset">
                      <ActionIcon
                        variant="light"
                        color="green"
                        onClick={() => handleRefresh(dataset.id)}
                      >
                        <IconRefresh style={{ width: rem(16), height: rem(16) }} />
                      </ActionIcon>
                    </Tooltip>
                    <Tooltip label="Cluster Dataset">
                      <ActionIcon
                        variant="light"
                        color="blue"
                        onClick={() => handleCluster(dataset.id)}
                        loading={actionLoading === dataset.id}
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
                      >
                        <IconTrash style={{ width: rem(16), height: rem(16) }} />
                      </ActionIcon>
                    </Tooltip>
                  </Group>
                </Table.Td>
              </Table.Tr>
            ))}
            {datasets.length === 0 && !loading && (
              <Table.Tr>
                <Table.Td colSpan={6}>
                  <Text ta="center" c="dimmed">
                    No datasets found. Create one to get started.
                  </Text>
                </Table.Td>
              </Table.Tr>
            )}
          </Table.Tbody>
        </Table>
      </Card>
    </Container>
  );
}
