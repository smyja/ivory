'use client';

import React, { useEffect, useState } from 'react';
import { Container, Title, Alert, Text, Center } from '@mantine/core';
import { ClusteringTable, ClusteringHistory } from '../table';

export default function ClusteringHistoryPage() {
  const [selectedStatus, setSelectedStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<ClusteringHistory[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchHistory = async () => {
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/datasets/clustering/history`
      );
      if (!response.ok) {
        throw new Error('Failed to fetch clustering history');
      }
      const data = await response.json();
      setHistory(data);
    } catch (error: any) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
    // Poll for updates every 10 seconds
    const interval = setInterval(fetchHistory, 10000);
    return () => clearInterval(interval);
  }, []);

  if (error) {
    return (
      <Container size="xl">
        <Alert color="red" title="Error">
          {error}
        </Alert>
      </Container>
    );
  }

  if (loading) {
    return (
      <Container size="xl" py="xl">
        <Center>
          <Text>Loading clustering history...</Text>
        </Center>
      </Container>
    );
  }

  return (
    <Container size="xl" py="xl">
      <Title order={1} mt={20} mb={20} fw={300}>
        Clustering History
      </Title>
      {history.length === 0 ? (
        <Center>
          <Text c="dimmed">No clustering history available.</Text>
        </Center>
      ) : (
        <ClusteringTable selectedStatus={selectedStatus} history={history} />
      )}
    </Container>
  );
}
