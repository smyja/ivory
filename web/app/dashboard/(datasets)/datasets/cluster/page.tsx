'use client';

import React, { useState, useEffect } from 'react';
import { Group, Paper, Text, Title, Container, Loader, Alert, Card, Pagination } from '@mantine/core';
import { useSearchParams, useRouter } from 'next/navigation';
import { IconArrowUpRight, IconArrowDownRight } from '@tabler/icons-react';
import { notifications } from '@mantine/notifications';
import classes from './ClusterView.module.css';

interface Category {
    id: number;
    name: string | null;
    total_rows: number;
    percentage: number;
    subclusters: Subcluster[];
}

interface Subcluster {
    id: number;
    title: string | null;
    row_count: number;
    percentage: number;
    texts: Text[];
}

interface Text {
    id: number;
    text: string;
    membership_score: number;
}

interface TitlingStatus {
    [key: string]: {
        status: 'pending' | 'in_progress' | 'completed';
        title?: string;
    };
}

export default function ClusterView() {
    const searchParams = useSearchParams();
    const router = useRouter();
    const datasetId = searchParams.get('id');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [categories, setCategories] = useState<Category[]>([]);
    const [titlingStatus, setTitlingStatus] = useState<TitlingStatus>({});
    const [currentPage, setCurrentPage] = useState<{ [key: number]: number }>({});
    const ITEMS_PER_PAGE = 6;

    const handleSubclusterClick = (subclusterId: number) => {
        router.push(`/dashboard/datasets/view?id=${datasetId}&subcluster=${subclusterId}`);
    };

    const requestTitling = async (clusterId: number, texts: Text[]) => {
        const key = `cluster-${clusterId}`;
        try {
            // Check if we're already titling this cluster
            if (titlingStatus[key]?.status === 'in_progress') {
                return;
            }

            setTitlingStatus(prev => ({
                ...prev,
                [key]: { status: 'in_progress' }
            }));

            // Send texts to titling endpoint
            const response = await fetch(
                `${process.env.NEXT_PUBLIC_API_URL}/datasets/${datasetId}/clusters/${clusterId}/title`,
                {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ texts: texts.map(t => t.text) }),
                }
            );

            if (!response.ok) {
                throw new Error('Failed to request titling');
            }

            const data = await response.json();

            if (data.status === 'in_progress') {
                // Poll for title status
                setTimeout(() => checkTitlingStatus(clusterId), 5000);
                return;
            }

            setTitlingStatus(prev => ({
                ...prev,
                [key]: { status: 'completed', title: data.title }
            }));

            // Update the cluster title in categories
            setCategories(prev => {
                return prev.map(category => ({
                    ...category,
                    subclusters: category.subclusters.map(subcluster =>
                        subcluster.id === clusterId
                            ? { ...subcluster, title: data.title }
                            : subcluster
                    )
                }));
            });

        } catch (error) {
            console.error('Error requesting titling:', error);
            setTitlingStatus(prev => ({
                ...prev,
                [key]: { status: 'pending' }
            }));
        }
    };

    const checkTitlingStatus = async (clusterId: number) => {
        const key = `cluster-${clusterId}`;
        try {
            const response = await fetch(
                `${process.env.NEXT_PUBLIC_API_URL}/datasets/${datasetId}/clusters/${clusterId}/title/status`
            );

            if (!response.ok) {
                throw new Error('Failed to check titling status');
            }

            const data = await response.json();

            if (data.status === 'in_progress') {
                setTimeout(() => checkTitlingStatus(clusterId), 5000);
                return;
            }

            if (data.status === 'completed') {
                setTitlingStatus(prev => ({
                    ...prev,
                    [key]: { status: 'completed', title: data.title }
                }));

                // Update the cluster title in categories
                setCategories(prev => {
                    return prev.map(category => ({
                        ...category,
                        subclusters: category.subclusters.map(subcluster =>
                            subcluster.id === clusterId
                                ? { ...subcluster, title: data.title }
                                : subcluster
                        )
                    }));
                });
            }
        } catch (error) {
            console.error('Error checking titling status:', error);
        }
    };

    const fetchClusters = async () => {
        try {
            const response = await fetch(
                `${process.env.NEXT_PUBLIC_API_URL}/datasets/${datasetId}/clusters`
            );
            if (!response.ok) {
                throw new Error('Failed to fetch clusters');
            }
            const data = await response.json();

            if (data.status === 'in_progress') {
                notifications.show({
                    title: 'In Progress',
                    message: 'Clustering is in progress. Please wait...',
                    color: 'yellow',
                });
                setTimeout(fetchClusters, 5000);
                return;
            }

            if (data.status === 'failed' || data.status === 'not_started') {
                setError(data.status === 'failed' ? 'Clustering failed' : 'Clustering not started');
                setLoading(false);
                return;
            }

            // Deduplicate categories and subclusters based on their IDs
            const uniqueCategories = data.categories.reduce((acc: Category[], curr: Category) => {
                if (!acc.find(c => c.id === curr.id)) {
                    // Deduplicate subclusters within the category
                    const uniqueSubclusters = curr.subclusters.reduce((subAcc: Subcluster[], subCurr: Subcluster) => {
                        if (!subAcc.find(s => s.id === subCurr.id)) {
                            subAcc.push(subCurr);
                        }
                        return subAcc;
                    }, []);

                    acc.push({
                        ...curr,
                        subclusters: uniqueSubclusters
                    });
                }
                return acc;
            }, []);

            console.log('Original categories count:', data.categories.length);
            console.log('Deduplicated categories count:', uniqueCategories.length);

            setCategories(uniqueCategories);

            // Request titling for each subcluster that doesn't have a title
            uniqueCategories.forEach((category: Category) => {
                category.subclusters.forEach((subcluster: Subcluster) => {
                    if (!subcluster.title && subcluster.texts.length > 0) {
                        requestTitling(subcluster.id, subcluster.texts);
                    }
                });
            });

        } catch (error: any) {
            setError(error.message);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (datasetId) {
            fetchClusters();
        }
    }, [datasetId]);

    const getPagedSubclusters = (subclusters: Subcluster[], categoryId: number) => {
        const page = currentPage[categoryId] || 1;
        const start = (page - 1) * ITEMS_PER_PAGE;
        const end = start + ITEMS_PER_PAGE;
        return subclusters.slice(start, end);
    };

    const handlePageChange = (categoryId: number, page: number) => {
        setCurrentPage(prev => ({ ...prev, [categoryId]: page }));
    };

    if (loading) {
        return (
            <Container size="xl">
                <div className="flex justify-center items-center h-screen">
                    <Loader size="xl" />
                </div>
            </Container>
        );
    }

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
            <div className={classes.root}>
                {categories.map((category) => {
                    const pagedSubclusters = getPagedSubclusters(category.subclusters, category.id);
                    const totalPages = Math.ceil(category.subclusters.length / ITEMS_PER_PAGE);

                    return (
                        <Card withBorder shadow="sm" radius="md" key={category.id} mb="lg" className={classes.categoryCard}>
                            <Group align="flex-start" wrap="nowrap">
                                <div className={classes.categoryInfo}>
                                    <div>
                                        <Title order={3} className={classes.categoryTitle}>
                                            {category.name || `Category ${category.id}`}
                                        </Title>
                                        <Text className={classes.categoryStats}>
                                            {category.subclusters.length} subclusters
                                        </Text>
                                    </div>
                                    <div>
                                        <Text size="lg" fw={700} className={classes.value}>
                                            {category.total_rows.toLocaleString()}
                                        </Text>
                                        <Text className={classes.categoryStats}>
                                            {category.percentage.toFixed(2)}% of total rows
                                        </Text>
                                    </div>
                                </div>

                                <div className={classes.subclusterContainer}>
                                    {pagedSubclusters.map((subcluster) => {
                                        const titleStatus = titlingStatus[`cluster-${subcluster.id}`];

                                        return (
                                            <Paper
                                                withBorder
                                                p="md"
                                                radius="md"
                                                key={subcluster.id}
                                                className={classes.subclusterCard}
                                                onClick={() => handleSubclusterClick(subcluster.id)}
                                                style={{ cursor: 'pointer' }}
                                            >
                                                <Text className={classes.subclusterTitle} lineClamp={2}>
                                                    {titleStatus?.status === 'in_progress' ? (
                                                        <Group gap="xs">
                                                            <Loader size="xs" />
                                                            <span>Generating title...</span>
                                                        </Group>
                                                    ) : (
                                                        subcluster.title || titleStatus?.title || `Subcluster ${subcluster.id}`
                                                    )}
                                                </Text>
                                                <div>
                                                    <Group align="flex-end" gap="xs">
                                                        <Text className={classes.value}>
                                                            {subcluster.row_count.toLocaleString()}
                                                        </Text>
                                                        <Text c={subcluster.percentage > 50 ? 'teal' : 'red'}
                                                            fz="sm"
                                                            fw={500}
                                                            className={classes.diff}>
                                                            <span>{subcluster.percentage.toFixed(1)}%</span>
                                                            {subcluster.percentage > 50 ?
                                                                <IconArrowUpRight size="1rem" stroke={1.5} /> :
                                                                <IconArrowDownRight size="1rem" stroke={1.5} />
                                                            }
                                                        </Text>
                                                    </Group>
                                                    <Text className={classes.categoryStats}>
                                                        rows in subcluster
                                                    </Text>
                                                </div>
                                            </Paper>
                                        );
                                    })}
                                    {totalPages > 1 && (
                                        <div style={{ gridColumn: '1/-1', display: 'flex', justifyContent: 'center', marginTop: '1rem' }}>
                                            <Pagination
                                                total={totalPages}
                                                value={currentPage[category.id] || 1}
                                                onChange={(page) => handlePageChange(category.id, page)}
                                            />
                                        </div>
                                    )}
                                </div>
                            </Group>
                        </Card>
                    );
                })}
            </div>
        </Container>
    );
} 