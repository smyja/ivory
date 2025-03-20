'use client';

import React, { useEffect, useState } from 'react';
import { Container, Title, Alert } from '@mantine/core';
import { ClusteringTable } from '../table';

interface ClusteringHistory {
    id: number;
    dataset_id: number;
    dataset_name: string;
    clustering_status: 'queued' | 'processing' | 'completed' | 'failed';
    titling_status: 'not_started' | 'in_progress' | 'completed' | 'failed';
    created_at: string;
    completed_at: string | null;
    error_message?: string;
}

export default function ClusteringHistoryPage() {
    const [selectedStatus, setSelectedStatus] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [history, setHistory] = useState<ClusteringHistory[]>([]);

    const fetchHistory = async () => {
        try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/datasets/clustering/history`);
            if (!response.ok) {
                throw new Error('Failed to fetch clustering history');
            }
            const data = await response.json();
            setHistory(data);
        } catch (error: any) {
            setError(error.message);
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

    return (
        <Container size="xl" py="xl">
            <Title order={2} mb="xl">Clustering History</Title>
            <ClusteringTable selectedStatus={selectedStatus} history={history} />
        </Container>
    );
} 