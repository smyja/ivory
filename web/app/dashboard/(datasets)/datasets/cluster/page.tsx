'use client';

import React, { useState, useEffect } from 'react';
import { Group, Paper, Text, Title, Container, Loader, Alert, Card, Pagination, Badge, Grid, Select } from '@mantine/core';
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
    const versionParam = searchParams.get('version');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [categories, setCategories] = useState<Category[]>([]);
    const [titlingStatus, setTitlingStatus] = useState<TitlingStatus>({});
    const [currentPage, setCurrentPage] = useState<{ [key: number]: number }>({});
    const [subclusterPages, setSubclusterPages] = useState<{ [key: number]: number }>({});
    const [versions, setVersions] = useState<{ id: number; version: number; created_at: string }[]>([]);
    const [selectedVersion, setSelectedVersion] = useState<number | null>(versionParam ? parseInt(versionParam) : null);
    const ITEMS_PER_PAGE = 6;
    const SUBCLUSTERS_PER_PAGE = 5;

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
            const url = new URL(`${process.env.NEXT_PUBLIC_API_URL}/datasets/${datasetId}/clusters`);
            if (selectedVersion) {
                url.searchParams.append('version', selectedVersion.toString());
            }

            const response = await fetch(url.toString());
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

    const fetchVersions = async () => {
        try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/datasets/${datasetId}/clustering/versions`);
            if (!response.ok) {
                throw new Error('Failed to fetch clustering versions');
            }
            const data = await response.json();
            setVersions(data);

            // If no version is selected, select the latest one
            if (!selectedVersion && data.length > 0) {
                const latestVersion = data[0].version;
                setSelectedVersion(latestVersion);
                router.push(`/dashboard/datasets/cluster?id=${datasetId}&version=${latestVersion}`);
            }
        } catch (error) {
            console.error('Error fetching versions:', error);
        }
    };

    useEffect(() => {
        if (datasetId) {
            fetchVersions();
            fetchClusters();
        }
    }, [datasetId, selectedVersion]);

    const getPagedSubclusters = (subclusters: Subcluster[], categoryId: number) => {
        const page = currentPage[categoryId] || 1;
        const start = (page - 1) * ITEMS_PER_PAGE;
        const end = start + ITEMS_PER_PAGE;
        return subclusters.slice(start, end);
    };

    const handlePageChange = (categoryId: number, page: number) => {
        setCurrentPage(prev => ({ ...prev, [categoryId]: page }));
    };

    const handleSubclusterPageChange = (categoryId: number, page: number) => {
        setSubclusterPages(prev => ({ ...prev, [categoryId]: page }));
    };

    const renderSubclusters = (subclusters: Subcluster[], categoryId: number) => {
        const currentPage = subclusterPages[categoryId] || 1;
        const startIndex = (currentPage - 1) * SUBCLUSTERS_PER_PAGE;
        const endIndex = startIndex + SUBCLUSTERS_PER_PAGE;
        const paginatedSubclusters = subclusters.slice(startIndex, endIndex);
        const totalPages = Math.ceil(subclusters.length / SUBCLUSTERS_PER_PAGE);

        return (
            <>
                {paginatedSubclusters.map((subcluster) => {
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
                    <Group justify="center" mt="md">
                        <Pagination
                            total={totalPages}
                            value={currentPage}
                            onChange={(page) => handleSubclusterPageChange(categoryId, page)}
                            size="sm"
                        />
                    </Group>
                )}
            </>
        );
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
                {versions.length > 0 && (
                    <Group mb="md" justify="space-between">
                        <Text fw={500}>Clustering Version:</Text>
                        <Select
                            value={selectedVersion?.toString() || ''}
                            onChange={(value) => {
                                if (value) {
                                    setSelectedVersion(parseInt(value));
                                    router.push(`/dashboard/datasets/cluster?id=${datasetId}&version=${value}`);
                                }
                            }}
                            data={versions.map(v => ({ value: v.version.toString(), label: `Version ${v.version} (${new Date(v.created_at).toLocaleString()})` }))}
                            style={{ width: '300px' }}
                        />
                    </Group>
                )}

                {categories.length === 0 ? (
                    <Alert color="blue" title="No Clusters Found">
                        There are no clusters available for this dataset. This could mean either:
                        <ul style={{ marginTop: '10px' }}>
                            <li>The clustering process has not been started yet</li>
                            <li>The clustering process is still in progress</li>
                            <li>No clusters were found in the dataset</li>
                        </ul>
                    </Alert>
                ) : (
                    categories.map((category) => {
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
                                        {renderSubclusters(category.subclusters, category.id)}
                                    </div>
                                </Group>
                            </Card>
                        );
                    })
                )}
            </div>
        </Container>
    );
} 