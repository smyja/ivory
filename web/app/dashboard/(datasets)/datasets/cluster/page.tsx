'use client';

import React, { useState, useEffect } from 'react';
import { Group, Paper, Text, Title, Container, Loader, Alert, Card, Pagination, Badge, Grid, Select } from '@mantine/core';
import { useSearchParams, useRouter } from 'next/navigation';
import { IconArrowUpRight, IconArrowDownRight } from '@tabler/icons-react';
import { notifications } from '@mantine/notifications';
import classes from './ClusterView.module.css';

interface TextDBResponse {
    id: number;
    text: string;
    // Add other fields if they exist in your TextDBResponse model
}

interface Level1ClusterResponse {
    id: number;
    l1_cluster_id: number;
    title: string | null;
    texts: TextDBResponse[];
    text_count: number;
}

interface CategoryResponse {
    id: number;
    name: string | null;
    l2_cluster_id?: number; // Make optional or adjust based on your model
    level1_clusters: Level1ClusterResponse[];
    category_text_count: number;
}

interface DatasetDetailResponse {
    // Inherited fields from DatasetMetadataResponse
    id: number;
    name: string;
    // ... other metadata fields
    latest_version: number | null;
    categories: CategoryResponse[] | null;
    dataset_total_texts: number | null;
    // Add status fields if needed
    status?: string;
    clustering_status?: string;
}

interface TitlingStatus {
    [key: string]: {
        status: 'pending' | 'in_progress' | 'completed';
        title?: string;
    };
}

function calculatePercentage(part: number | null | undefined, total: number | null | undefined): number {
    if (total && total > 0 && part != null) {
        return (part / total) * 100;
    }
    return 0; // Return 0 if calculation isn't possible
}

export default function ClusterView() {
    const searchParams = useSearchParams();
    const router = useRouter();
    const datasetId = searchParams.get('id');
    const versionParam = searchParams.get('version');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [datasetDetails, setDatasetDetails] = useState<DatasetDetailResponse | null>(null);
    const [titlingStatus, setTitlingStatus] = useState<TitlingStatus>({});
    const [currentPage, setCurrentPage] = useState<{ [key: number]: number }>({});
    const [versions, setVersions] = useState<number[]>([]);
    const [selectedVersion, setSelectedVersion] = useState<number | null>(versionParam ? parseInt(versionParam) : null);
    const SUBCLUSTERS_PER_PAGE = 5;

    const handleSubclusterClick = (subclusterId: number) => {
        router.push(`/dashboard/datasets/view?id=${datasetId}&subcluster=${subclusterId}`);
    };

    const requestTitling = async (clusterId: number, texts: TextDBResponse[]) => {
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
            setDatasetDetails(prevDetails => {
                if (!prevDetails || !prevDetails.categories) return prevDetails;
                const updatedCategories = prevDetails.categories.map(category => ({
                    ...category,
                    level1_clusters: category.level1_clusters.map(level1Cluster =>
                        level1Cluster.id === clusterId
                            ? { ...level1Cluster, title: data.title }
                            : level1Cluster
                    )
                }));
                return { ...prevDetails, categories: updatedCategories };
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
                setDatasetDetails(prevDetails => {
                    if (!prevDetails || !prevDetails.categories) return prevDetails;
                    const updatedCategories = prevDetails.categories.map(category => ({
                        ...category,
                        level1_clusters: category.level1_clusters.map(level1Cluster =>
                            level1Cluster.id === clusterId
                                ? { ...level1Cluster, title: data.title }
                                : level1Cluster
                        )
                    }));
                    return { ...prevDetails, categories: updatedCategories };
                });
            }
        } catch (error) {
            console.error('Error checking titling status:', error);
        }
    };

    const fetchClusters = async () => {
        if (!datasetId) return;
        setLoading(true);
        setError(null);
        try {
            const url = new URL(`${process.env.NEXT_PUBLIC_API_URL}/datasets/${datasetId}/details`);
            if (selectedVersion) {
                url.searchParams.append('version', selectedVersion.toString());
            }

            const response = await fetch(url.toString());
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Failed to fetch clusters' }));
                throw new Error(errorData.detail || 'Failed to fetch clusters');
            }
            const data: DatasetDetailResponse = await response.json();

            console.log('API Response for clusters/details:', {
                url: url.toString(),
                data,
            });

            if (data.clustering_status === 'processing') {
                notifications.show({
                    title: 'In Progress',
                    message: 'Clustering is in progress. Refreshing...',
                    color: 'yellow',
                });
                setTimeout(fetchClusters, 5000);
                return;
            }

            if (data.clustering_status === 'failed') {
                setError('Clustering failed for this version.');
                setDatasetDetails(data);
                setLoading(false);
                return;
            }

            if (!data.categories || !Array.isArray(data.categories)) {
                console.warn(`No categories found or invalid format for dataset ${datasetId}, version ${selectedVersion}`);
                setDatasetDetails({ ...data, categories: [] });
            } else {
                setDatasetDetails(data);
                data.categories.forEach((category: CategoryResponse) => {
                    category.level1_clusters.forEach((level1Cluster: Level1ClusterResponse) => {
                        if (!level1Cluster.title && level1Cluster.texts.length > 0 && !titlingStatus[`cluster-${level1Cluster.id}`]) {
                            requestTitling(level1Cluster.id, level1Cluster.texts);
                        }
                    });
                });
            }
        } catch (error: any) {
            console.error("Fetch Clusters Error:", error);
            setError(error.message || 'An unexpected error occurred.');
            setDatasetDetails(null);
        } finally {
            setLoading(false);
        }
    };

    const fetchVersions = async () => {
        if (!datasetId) return;
        try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/datasets/${datasetId}/clustering/versions`);
            if (!response.ok) {
                throw new Error('Failed to fetch clustering versions');
            }
            const data: number[] = await response.json();
            setVersions(data);

            if (selectedVersion === null && data.length > 0) {
                const latestVersion = Math.max(...data);
                setSelectedVersion(latestVersion);
                router.replace(`/dashboard/datasets/cluster?id=${datasetId}&version=${latestVersion}`);
            } else if (data.length === 0) {
                setError("No completed clustering versions found for this dataset.");
            }
        } catch (error: any) {
            console.error('Error fetching versions:', error);
            setError(error.message || "Failed to load clustering versions.");
            setVersions([]);
        }
    };

    useEffect(() => {
        if (datasetId) {
            fetchVersions();
        }
    }, [datasetId]);

    useEffect(() => {
        if (datasetId && selectedVersion !== null) {
            fetchClusters();
        }
        if (selectedVersion === null) {
            setDatasetDetails(null);
        }
    }, [datasetId, selectedVersion]);

    const handleSubclusterPageChange = (categoryId: number, page: number) => {
        setCurrentPage(prev => ({ ...prev, [categoryId]: page }));
    };

    const renderLevel1Clusters = (
        level1Clusters: Level1ClusterResponse[],
        categoryId: number,
        categoryTextCount: number
    ) => {
        const currentSubclusterPage = currentPage[categoryId] || 1;
        const startIndex = (currentSubclusterPage - 1) * SUBCLUSTERS_PER_PAGE;
        const endIndex = startIndex + SUBCLUSTERS_PER_PAGE;
        const paginatedLevel1Clusters = level1Clusters.slice(startIndex, endIndex);
        const totalSubclusterPages = Math.ceil(level1Clusters.length / SUBCLUSTERS_PER_PAGE);

        return (
            <>
                {paginatedLevel1Clusters.map((level1Cluster) => {
                    const titleStatus = titlingStatus[`cluster-${level1Cluster.id}`];
                    const subclusterPercentage = calculatePercentage(level1Cluster.text_count, categoryTextCount);

                    return (
                        <Paper
                            withBorder
                            p="md"
                            radius="md"
                            key={level1Cluster.id}
                            className={classes.subclusterCard}
                            onClick={() => handleSubclusterClick(level1Cluster.id)}
                            style={{ cursor: 'pointer' }}
                        >
                            <Text className={classes.subclusterTitle} lineClamp={2}>
                                {titleStatus?.status === 'in_progress' ? (
                                    <Group gap="xs">
                                        <Loader size="xs" />
                                        <span>Generating title...</span>
                                    </Group>
                                ) : (
                                    level1Cluster.title || titleStatus?.title || `Subcluster ${level1Cluster.l1_cluster_id}`
                                )}
                            </Text>
                            <div>
                                <Group align="flex-end" gap="xs">
                                    <Text className={classes.value}>
                                        {level1Cluster.text_count.toLocaleString()}
                                    </Text>
                                    <Text
                                        c={subclusterPercentage > 50 ? 'teal' : 'red'}
                                        fz="sm" fw={500} className={classes.diff}
                                    >
                                        <span>{subclusterPercentage.toFixed(1)}%</span>
                                        {subclusterPercentage > 50 ?
                                            <IconArrowUpRight size="1rem" stroke={1.5} /> :
                                            <IconArrowDownRight size="1rem" stroke={1.5} />
                                        }
                                    </Text>
                                </Group>
                                <Text fz="xs" c="dimmed" mt={2} className={classes.categoryStats}>
                                    rows in subcluster
                                </Text>
                            </div>
                        </Paper>
                    );
                })}
                {totalSubclusterPages > 1 && (
                    <Group justify="center" mt="md">
                        <Pagination
                            total={totalSubclusterPages}
                            value={currentSubclusterPage}
                            onChange={(page) => handleSubclusterPageChange(categoryId, page)}
                            size="sm"
                        />
                    </Group>
                )}
            </>
        );
    };

    if (loading && !datasetDetails) {
        return (
            <Container size="xl" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh' }}>
                <Loader size="xl" />
            </Container>
        );
    }

    if (error) {
        return (
            <Container size="xl" py="xl">
                <Alert color="red" title="Error Loading Clusters">
                    {error} - Try selecting a different version or refreshing.
                </Alert>
                {versions.length > 0 && (
                    <Group mt="md" justify="center">
                        <Text fw={500}>Select Version:</Text>
                        <Select
                            value={selectedVersion?.toString() || ''}
                            onChange={(value) => {
                                if (value) {
                                    const newVersion = parseInt(value);
                                    setSelectedVersion(newVersion);
                                    router.replace(`/dashboard/datasets/cluster?id=${datasetId}&version=${newVersion}`);
                                } else {
                                    setSelectedVersion(null);
                                    router.replace(`/dashboard/datasets/cluster?id=${datasetId}`);
                                }
                            }}
                            data={versions.map(version => ({
                                value: version.toString(),
                                label: `Version ${version}`
                            }))}
                            placeholder="Select version"
                            style={{ width: '200px' }}
                        />
                    </Group>
                )}
            </Container>
        );
    }

    return (
        <Container size="xl" py="xl">
            <div className={classes.root}>
                {versions.length > 0 && (
                    <Group mb="xl" justify="space-between" align="center">
                        <Title order={2}>Dataset Clusters</Title>
                        <Group>
                            <Text fw={500}>Clustering Version:</Text>
                            <Select
                                value={selectedVersion?.toString() || ''}
                                onChange={(value) => {
                                    if (value) {
                                        const newVersion = parseInt(value);
                                        setSelectedVersion(newVersion);
                                        router.replace(`/dashboard/datasets/cluster?id=${datasetId}&version=${newVersion}`);
                                    } else {
                                        setSelectedVersion(null);
                                        router.replace(`/dashboard/datasets/cluster?id=${datasetId}`);
                                    }
                                }}
                                data={versions.map(version => ({
                                    value: version.toString(),
                                    label: `Version ${version}`
                                }))}
                                placeholder="Select version"
                                style={{ width: '200px' }}
                            />
                        </Group>
                    </Group>
                )}

                {loading && <Loader style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }} />}

                {!loading && (!datasetDetails || !datasetDetails.categories || datasetDetails.categories.length === 0) && (
                    <Alert color="blue" title="No Clusters Found">
                        No cluster data available for Version {selectedVersion}. This could be due to an incomplete clustering process or no clusters being generated.
                    </Alert>
                )}

                {datasetDetails && datasetDetails.categories && datasetDetails.categories.length > 0 && (
                    datasetDetails.categories.map((category) => {
                        const categoryPercentage = calculatePercentage(
                            category.category_text_count,
                            datasetDetails.dataset_total_texts
                        );

                        return (
                            <Card withBorder shadow="sm" radius="md" key={category.id} mb="lg" className={classes.categoryCard}>
                                <Group align="flex-start" wrap="nowrap">
                                    <div className={classes.categoryInfo}>
                                        <div>
                                            <Title order={3} className={classes.categoryTitle}>
                                                {category.name || `Category ${category.id}`}
                                            </Title>
                                            <Text className={classes.categoryStats}>
                                                {category.level1_clusters.length} subclusters
                                            </Text>
                                        </div>
                                        <div>
                                            <Text size="lg" fw={700} className={classes.value}>
                                                {category.category_text_count.toLocaleString()}
                                            </Text>
                                            <Text className={classes.categoryStats}>
                                                {categoryPercentage.toFixed(2)}% of total rows
                                            </Text>
                                        </div>
                                    </div>

                                    <div className={classes.subclusterContainer}>
                                        {renderLevel1Clusters(
                                            category.level1_clusters,
                                            category.id,
                                            category.category_text_count
                                        )}
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